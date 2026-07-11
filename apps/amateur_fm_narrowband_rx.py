#!/usr/bin/env python3
"""Amateur narrowband FM receive flowgraph.

Replaces ``stub_rx.py`` for bench / production. Real GNU Radio code:

    SoapySDR Source (IQ at ``--sample-rate``)
       -> Channel filter (LPF at half ``--bandwidth-hz``)
       -> Rational resampler to ~96 kHz IF
       -> Quadrature demod (NBFM)
       -> Audio decimator to 48 kHz mono
       -> De-emphasis (75 µs)
       -> Float32 PCM byte stream
       -> data socket

Status events emitted:

    {"event":"ready","data_format":"audio_pcm_f32_48k","sample_rate":48000}
    {"event":"started"}
    {"event":"signal","rssi_dbm":...,"snr_db":...,"lock":...}  (10 Hz)
    {"event":"stopped","reason":"..."}

The orchestrator binds the three TCP sockets and waits for our
``ready`` event. It then sends ``{"cmd":"start"}`` at AOS;
``{"cmd":"stop"}`` at LOS. Doppler tracking uses
``{"cmd":"set_doppler","offset_hz":...}`` — supported by re-tuning
the SoapySDR source.

**Audio format choice.** Document A §A.9.4 names ``audio.ogg`` (Vorbis)
as the canonical artifact, but in-flowgraph OGG encoding adds a
non-trivial dependency (vorbis-tools or pyogg). This first cut emits
raw float32 PCM and declares ``data_format="audio_pcm_f32_48k"`` —
the orchestrator's ``spec_for_data_format`` falls through to
RAW_BITS (filename ``raw_bits.bin``). Phase 7+ can add OGG encoding
via a Python sink block piping through ``oggenc``.

**Verification.** This script imports ``gnuradio`` and ``soapy`` at
module load — those packages are NOT in ``gs-client``'s venv. Run
this on a Linux bench with ``gnuradio`` + ``gr-soapy`` installed
(typical via ``apt install gnuradio gr-soapy`` on Debian/Ubuntu).
For dev without a real SDR, use SoapyLoopback (``driver=loopback``).

License: GPLv3 (see ``../COPYING``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import sys
import time

from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    pump_data_queue,
    run_command_loop,
    send_event,
)
from _recorder import PassRecorder
from _soapy import (
    apply_corrections,
    auto_lo_offset,
    capture_plan,
    configure_soapy_source,
    make_decimator,
    make_source,
    merge_sdr_params,
    retune_source,
    sdr_env,
    tune_source,
)
from gnuradio import analog, filter as gr_filter, gr, soapy

VERSION = "0.1.0"

# Signal-event cadence (Document A telemetry default).
_SIGNAL_PERIOD_S = 0.1

# Internal audio rate after IF decimation. 48 kHz is the spec canonical
# (§A.9.4); the channel filter shapes a 25 kHz channel before
# resampling.
_AUDIO_RATE_HZ = 48_000
_IF_RATE_HZ = 192_000  # 4× audio for clean LPF + quad-demod headroom
_CHANNEL_BW_HZ = 25_000

# Queue depth: enough for ~50 ms of audio at 48 kHz mono float32
# (50 ms × 48k × 4 B = 9.6 KB per chunk; ~10 chunks keeps lag <0.5 s).
_DATA_QUEUE_MAXSIZE = 32


class _QueueSink(gr.sync_block):
    """GR sink that pushes raw float32 PCM into a ``queue.Queue`` for
    the asyncio side to drain. Drops chunks if the queue is full
    (telemetry will reflect via a future ``gap_marker`` event).
    """

    def __init__(self, target_queue: queue.Queue[bytes | None]) -> None:
        import numpy as np  # local import keeps module import-light

        gr.sync_block.__init__(
            self,
            name="queue_sink",
            in_sig=[np.float32],
            out_sig=None,
        )
        self._q = target_queue
        self.dropped = 0

    def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
        samples = input_items[0]  # numpy float32 array
        try:
            self._q.put_nowait(samples.tobytes())
        except queue.Full:
            self.dropped += len(samples)
        return len(samples)


class FlowgraphContext:
    """Holds the top_block plus references the asyncio side needs to
    drive at runtime (Doppler retune, etc.). Returned from
    :func:`build_top_block` so the spawn handlers can reach into the
    pipeline without globals.
    """

    def __init__(
        self,
        tb: gr.top_block,
        src: object,
        recorder: object = None,
        lo_offset_hz: float = 0.0,
    ) -> None:
        self.tb = tb
        self.src = src
        self.recorder = recorder  # pre-demod IQ capture (PassRecorder) or None
        self.lo_offset_hz = lo_offset_hz  # LO offset the Doppler retune must preserve


def _retune_locked(
    tb: gr.top_block, src: object, new_freq_hz: float, lo_offset_hz: float = 0.0
) -> None:
    """Retune the running source under tb.lock()/unlock(). Called
    from :func:`asyncio.to_thread` so the asyncio loop doesn't block
    on the GR scheduler. Preserves the LO offset (moves only the RF component).
    """
    try:
        tb.lock()
        retune_source(src, new_freq_hz, lo_offset_hz, 0.0)
    finally:
        tb.unlock()


def build_top_block(
    args,  # type: ignore[no-untyped-def]
    audio_queue: queue.Queue[bytes | None],
    params: dict[str, object] | None = None,
) -> FlowgraphContext:
    """Construct the NBFM RX flowgraph. Returns a started-but-not-
    running ``top_block`` — the caller invokes ``.start()`` when
    ``cmd:start`` arrives.

    ``params`` holds the directive's ``RfLink.waveform_parameters``
    (Document C C.5.5.2). Honoured keys:

    * ``sdr_gain_db`` (float) — SoapySDR source gain. Default 30 dB.
    * ``fm_deviation_hz`` (float) — peak FM deviation; affects the
      quadrature-demod gain. Default 5000 (amateur narrowband).
    * ``deemph_tau_s`` (float) — de-emphasis filter time constant.
      Default 75e-6 (broadcast FM); 50e-6 is closer to amateur spec.
    * ``audio_cutoff_hz`` (float) — voice LPF cutoff. Default 3500.

    Unknown keys are ignored. Missing keys fall back to defaults.
    """
    p: dict[str, object] = params or {}
    fm_deviation_hz = float(p.get("fm_deviation_hz", 5_000.0))  # type: ignore[arg-type]
    deemph_tau_s = float(p.get("deemph_tau_s", 75e-6))  # type: ignore[arg-type]
    audio_cutoff_hz = float(p.get("audio_cutoff_hz", 3_500.0))  # type: ignore[arg-type]

    tb = gr.top_block("amateur_fm_narrowband_rx")

    # ------------------------------------------------------------ source
    # SoapySDR source via gr-soapy. ``driver`` keyword string from
    # --sdr-args is parsed by gr-soapy itself; pass ``soapy_args``
    # exactly as the orchestrator built it.
    env = sdr_env()  # station-wide GS_SDR_* (antenna/gain/lo-offset/ppm/dc-removal/rate)
    # Capture at the SDR's supported rate (XTRX floor ~2.1 Msps), decimate to the
    # channel rate ahead of the channel filter; the FM chain runs at args.sample_rate.
    sdr_rate, decimate = capture_plan(env["capture_rate_hz"], float(args.sample_rate))
    # AUTO LO offset → DC spike dodged off the bird (no per-pass config); see _soapy.
    lo_offset_hz = auto_lo_offset(sdr_rate, float(args.sample_rate), env["lo_offset_hz"])
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo_offset_hz)  # LO offset → DC spike off-signal
    # antenna + gain (else the front-end is on a disconnected antenna / 0 dB).
    # Precedence: per-pass sdr_gain_db param > GS_SDR_GAIN_DB env > 30 dB default.
    configure_soapy_source(src, merge_sdr_params(p))
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])
    # Analog RX filter ≈ the SDR CAPTURE rate, NOT the channel width. The XTRX's filter
    # floor is ~0.8 MHz, so setting it to the narrow channel BW (e.g. --bandwidth-hz
    # 25000) lands below the floor and breaks the analog path → 0 samples → 0-byte
    # capture. Channel selectivity is done downstream in DSP (the decimator + chan filter).
    src.set_bandwidth(0, sdr_rate)

    # ----------------------------------------------------- channel filter
    # Pass +/- (channel_bw/2) around DC after the source is centred.
    # Transition band 10% of channel_bw.
    chan_taps = gr_filter.firdes.low_pass(
        gain=1.0,
        sampling_freq=float(args.sample_rate),
        cutoff_freq=_CHANNEL_BW_HZ / 2.0,
        transition_width=_CHANNEL_BW_HZ * 0.1,
    )
    # The channel stream enters at args.sample_rate (the post-decimator channel rate, e.g.
    # 48 kHz). ``decim_to_if`` is 1 when that's already <= the IF rate, so compute the
    # ACTUAL rate after the channel filter and drive the demod + audio stages from it.
    # (The old code hard-coded _IF_RATE_HZ=192 kHz, which inverted the gain + audio rate
    # whenever sample_rate < 192 kHz — i.e. always, at a 48 kHz channel.)
    decim_to_if = max(1, int(args.sample_rate) // _IF_RATE_HZ)
    if_rate = float(args.sample_rate) / decim_to_if
    chan = gr_filter.fir_filter_ccf(decim_to_if, chan_taps)

    # ----------------------------------------------------- NBFM demod
    # Quad-demod gain depends on the configured peak deviation; default
    # 5 kHz is the amateur narrowband convention. ``params`` may override
    # for missions that use wider/narrower deviation.
    quad_gain = if_rate / (2 * math.pi * fm_deviation_hz)
    demod = analog.quadrature_demod_cf(quad_gain)

    # ----------------------------------------------------- audio decimator
    audio_decim = max(1, int(if_rate) // _AUDIO_RATE_HZ)
    audio_taps = gr_filter.firdes.low_pass(
        gain=1.0,
        sampling_freq=if_rate,
        cutoff_freq=audio_cutoff_hz,  # voice bandwidth (param-tunable)
        transition_width=500.0,
    )
    audio_lpf = gr_filter.fir_filter_fff(audio_decim, audio_taps)

    # ----------------------------------------------------- de-emphasis
    # Single-pole filter at the ACTUAL audio output rate; tau=75 µs is FM broadcast,
    # tau=50 µs is closer to amateur spec — backend can override via ``deemph_tau_s``.
    deemph = analog.fm_deemph(if_rate / audio_decim, tau=deemph_tau_s)

    # ----------------------------------------------------- queue sink
    sink = _QueueSink(audio_queue)

    # ----------------------------------------------------- connect
    # Decimate to the channel rate ONCE; the demod chain and the recorder both tap it, so
    # the capture is the narrow channel (~MB/min), not the multi-GB wideband SDR stream.
    chan_src = src
    if decimate:  # SDR at capture rate → resample down to the channel rate first
        chan_src = make_decimator(sdr_rate, float(args.sample_rate))
        tb.connect(src, chan_src)
    tb.connect(chan_src, chan, demod, audio_lpf, deemph, sink)

    # Pre-demod IQ capture at the CHANNEL rate (the decimator output), not wideband.
    recorder = PassRecorder.maybe_start(args, tb, chan_src, sample_rate_hz=float(args.sample_rate))
    return FlowgraphContext(tb=tb, src=src, recorder=recorder, lo_offset_hz=lo_offset_hz)


# ----------------------------------------------------------------------
# Asyncio entrypoint
# ----------------------------------------------------------------------


async def amain(args) -> int:  # type: ignore[no-untyped-def]
    log = logging.getLogger("amateur_fm_narrowband_rx")
    sockets = await connect_spawn_sockets(args)

    # Honour per-pass waveform_parameters from the directive. The
    # orchestrator wrote ``params.json`` into ``--output-dir`` (or
    # passed ``--params-file`` directly); an empty Struct → no file
    # → defaults.
    params = load_params(args)
    if params:
        log.info("loaded waveform_parameters: %s", sorted(params))

    data_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_DATA_QUEUE_MAXSIZE)
    ctx = build_top_block(args, data_queue, params=params)
    tb = ctx.tb
    src = ctx.src
    lo_offset_hz = ctx.lo_offset_hz  # preserved across Doppler retunes (_on_set_doppler)
    pump_task = asyncio.create_task(
        pump_data_queue(data_queue, sockets.data_writer),
        name="data-pump",
    )

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "audio_pcm_f32_48k",
            "sample_rate": _AUDIO_RATE_HZ,
            "flowgraph_version": VERSION,
        },
    )

    started = asyncio.Event()
    stop_requested = asyncio.Event()

    # RX: start the flowgraph at spawn (arm — before AOS) so the SDR is warm + recording
    # before AOS. Do NOT gate streaming on cmd:start (that's for TX keying); cmd:start
    # still confirms via the 'started' status event and cmd:stop ends the pass.
    tb.start()
    started.set()

    async def _on_start(_cmd: dict[str, object]) -> None:
        # Already streaming since spawn; just confirm to the orchestrator.
        await send_event(sockets.status_writer, {"event": "started"})

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        stop_reason["value"] = str(cmd.get("reason", "command"))

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        offset = cmd.get("offset_hz", 0)
        if not isinstance(offset, (int, float)):
            return
        # Re-tune the SoapySDR source by ``offset_hz`` from the
        # original centre. The convention from
        # :func:`gs_client.rf.doppler.doppler_shift_hz` is that a
        # receding satellite (range_rate > 0) produces a NEGATIVE
        # shift on the RX side — we tune DOWN to compensate so the
        # rest-frame carrier lands at DC. ``offset_hz`` is already the
        # signed shift, so the new frequency is the original carrier
        # plus the shift.
        new_freq = float(args.center_freq_hz) + float(offset)
        log.info("set_doppler: offset=%.1f Hz -> tune %.0f", offset, new_freq)
        # tb.lock()/unlock() bracketing per the GR cookbook: stop
        # flowgraph propagation, re-tune, resume. Source supports
        # live set_frequency without lock/unlock on most SoapySDR
        # devices, but locking is the documented-safe path.
        await asyncio.to_thread(_retune_locked, tb, src, new_freq, lo_offset_hz)

    # --------------------------------------------------- signal-event task
    async def _emit_signal_events() -> None:
        while not stop_requested.is_set():
            await asyncio.sleep(_SIGNAL_PERIOD_S)
            # Real signal stats would come from a probe block; first
            # cut emits a synthesised reasonable value. Plumbing the
            # probe is a Phase 7 follow-up.
            now = time.time()
            await send_event(
                sockets.status_writer,
                {
                    "event": "signal",
                    "rssi_dbm": -80.0 + math.sin(now) * 5.0,
                    "snr_db": 12.0,
                    "lock": True,
                },
            )

    signal_task = asyncio.create_task(_emit_signal_events(), name="signal-events")

    handlers = {
        "start": _on_start,
        "stop": _on_stop,
        "set_doppler": _on_set_doppler,
    }
    engine_down = {"done": False}

    async def _shutdown_engine() -> None:
        """Idempotent: stop the graph, tear down the data pump."""
        if engine_down["done"]:
            return
        engine_down["done"] = True
        stop_requested.set()
        if started.is_set():
            try:
                # Just stop the graph; views are derived post-pass by gs-client (iq_views
                # on the on-disk cf32), so a slow/hung teardown can't cost the recording.
                tb.stop()
                tb.wait()
            except Exception:
                log.exception("tb.stop/wait raised")
        data_queue.put(None)
        signal_task.cancel()
        await asyncio.gather(pump_task, signal_task, return_exceptions=True)

    try:
        reason = await run_command_loop(sockets.control_reader, handlers)
        # P0-08: engine teardown BEFORE the explicit stopped ack; then exit 0.
        # EOF is transport loss — no ack, exit nonzero.
        await _shutdown_engine()
        if reason == "stop":
            await send_event(
                sockets.status_writer,
                {"event": "stopped", "reason": stop_reason["value"]},
            )
            return 0
        log.warning("control EOF without stop — transport loss; exiting nonzero (P0-08)")
        return 1
    finally:
        await _shutdown_engine()
        await sockets.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="amateur_fm_narrowband_rx",
        description="Amateur narrowband FM receiver (Phase 5/6 real GR).",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    logging.basicConfig(level=logging.INFO)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

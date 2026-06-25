#!/usr/bin/env python3
"""Amateur narrowband FM transmit flowgraph (test-tone first cut).

Replaces ``stub_tx.py`` for bench / production. Real GNU Radio code.

**First cut scope.** This emits a continuous 1 kHz test tone over
NBFM while keyed. It does NOT yet accept audio data via the data
socket (Document A §A.7.4 makes that flowgraph-specific; future
voice / packet variants will add an audio source). The point of the
first cut is to verify the full RF path end-to-end on the bench:
spawn -> SDR transmit -> dummy load -> forward-power sensor reads
non-zero -> normal disarm. After Phase 6/7 verifies this works,
follow-up phases add proper modulation (voice via data socket,
packet via control-socket ``transmit_frame`` commands).

Pipeline::

    1 kHz tone source
       -> NBFM modulator (frequency_modulator_fc)
       -> Resampler from audio rate to SDR rate
       -> SoapySDR Sink

Status events emitted::

    {"event":"ready","data_format":"none","sample_rate":48000}
    {"event":"started"}
    {"event":"transmit_started"}      ← drives KEYED_READY -> KEYED in safety FSM
    {"event":"transmit_complete","frame_id":"tone"}
    {"event":"stopped","reason":"..."}

License: GPLv3 (see ``../COPYING``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys

from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)
from _soapy import apply_corrections, configure_soapy_source, make_sink, merge_sdr_params, sdr_env
from gnuradio import analog, filter as gr_filter, gr, soapy

VERSION = "0.1.0"

_AUDIO_RATE_HZ = 48_000
_TONE_HZ = 1000.0
_FM_DEVIATION_HZ = 2500.0


class FlowgraphContext:
    """Holds the top_block + the SoapySDR sink reference the asyncio
    side needs for live Doppler retune."""

    def __init__(self, tb: gr.top_block, sink: object) -> None:
        self.tb = tb
        self.sink = sink


def _retune_locked(tb: gr.top_block, sink: object, new_freq_hz: float) -> None:
    """Retune the running sink under tb.lock()/unlock()."""
    try:
        tb.lock()
        sink.set_frequency(0, new_freq_hz)
    finally:
        tb.unlock()


def build_top_block(
    args,  # type: ignore[no-untyped-def]
    params: dict[str, object] | None = None,
) -> FlowgraphContext:
    """Construct the NBFM TX flowgraph.

    Honoured ``params`` keys (Document C C.5.5.2):

    * ``tone_hz`` (float) — sinusoid frequency. Default 1000.
    * ``fm_deviation_hz`` (float) — peak FM deviation. Default 2500.
    * ``sdr_gain_db`` (float) — SoapySDR sink gain. Default 30 dB.
    """
    p: dict[str, object] = params or {}
    tone_hz = float(p.get("tone_hz", _TONE_HZ))  # type: ignore[arg-type]
    fm_deviation_hz = float(p.get("fm_deviation_hz", _FM_DEVIATION_HZ))  # type: ignore[arg-type]

    tb = gr.top_block("amateur_fm_narrowband_tx")

    # ------------------------------------------------------------ tone
    # Sinusoid at the audio rate; amplitude scaled so post-modulation
    # peak deviation matches ``fm_deviation_hz``.
    tone = analog.sig_source_f(
        _AUDIO_RATE_HZ,
        analog.GR_SIN_WAVE,
        tone_hz,
        1.0,
        0.0,
    )

    # ----------------------------------------------------- FM modulate
    # ``frequency_modulator_fc`` is the inverse of the RX path's
    # quadrature_demod_cf. ``sensitivity`` matches deviation per
    # sample.
    sensitivity = (2 * math.pi * fm_deviation_hz) / _AUDIO_RATE_HZ
    fm_mod = analog.frequency_modulator_fc(sensitivity)

    # ----------------------------------------------------- interpolate
    # Audio -> SDR rate.
    interp = max(1, args.sample_rate // _AUDIO_RATE_HZ)
    interp_taps = gr_filter.firdes.low_pass(
        gain=float(interp),
        sampling_freq=float(args.sample_rate),
        cutoff_freq=_AUDIO_RATE_HZ / 2.0,
        transition_width=_AUDIO_RATE_HZ * 0.1,
        window=gr_filter.firdes.WIN_HAMMING,
    )
    interp_filter = gr_filter.interp_fir_filter_ccf(interp, interp_taps)

    # ----------------------------------------------------- soapy sink
    sink = make_sink(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    sink.set_sample_rate(0, float(args.sample_rate))
    sink.set_frequency(0, float(args.center_freq_hz))  # TX: no LO offset (modulator at baseband 0)
    # antenna + PA gain. Precedence: sdr_gain_db param > GS_SDR_GAIN_DB env > 30 dB.
    configure_soapy_source(sink, merge_sdr_params(p))
    apply_corrections(sink, ppm=sdr_env()["ppm"], dc_removal=False)
    sink.set_bandwidth(0, float(args.bandwidth_hz) if args.bandwidth_hz else 200_000.0)

    tb.connect(tone, fm_mod, interp_filter, sink)
    return FlowgraphContext(tb=tb, sink=sink)


async def amain(args) -> int:  # type: ignore[no-untyped-def]
    log = logging.getLogger("amateur_fm_narrowband_tx")
    sockets = await connect_spawn_sockets(args)

    params = load_params(args)
    if params:
        log.info("loaded waveform_parameters: %s", sorted(params))

    ctx = build_top_block(args, params=params)
    tb = ctx.tb
    sink = ctx.sink

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "none",
            "sample_rate": _AUDIO_RATE_HZ,
            "flowgraph_version": VERSION,
        },
    )

    started = asyncio.Event()
    stop_requested = asyncio.Event()

    async def _on_start(_cmd: dict[str, object]) -> None:
        if started.is_set():
            return
        tb.start()
        started.set()
        await send_event(sockets.status_writer, {"event": "started"})
        # Real RF is now on the air. Emit transmit_started so the
        # orchestrator can flip the safety FSM KEYED_READY -> KEYED
        # (Document A §A.6.1 + the on_transmit_started handler in
        # ``gs_client.core.orchestrator``).
        await send_event(sockets.status_writer, {"event": "transmit_started"})

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        if started.is_set():
            tb.stop()
            tb.wait()
        await send_event(
            sockets.status_writer,
            {"event": "stopped", "reason": str(cmd.get("reason", "command"))},
        )

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        offset = cmd.get("offset_hz", 0)
        if not isinstance(offset, (int, float)):
            return
        # On TX, a receding satellite means our uplink will appear
        # at a LOWER frequency at the spacecraft — we pre-tune
        # UP by ``|offset_hz|`` so the spacecraft sees rest-frame.
        # ``offset_hz`` is the signed shift (negative for receding),
        # so the corrected TX frequency is ``carrier - offset``.
        new_freq = float(args.center_freq_hz) - float(offset)
        log.info("set_doppler: offset=%.1f Hz -> tune %.0f", offset, new_freq)
        await asyncio.to_thread(_retune_locked, tb, sink, new_freq)

    async def _on_transmit_frame(cmd: dict[str, object]) -> None:
        # First-cut TX is tone-only; future variants accept payload
        # bytes via this command. We acknowledge the frame so the
        # orchestrator's frame ledger doesn't time out.
        fid = cmd.get("frame_id", "")
        await send_event(
            sockets.status_writer,
            {"event": "transmit_complete", "frame_id": fid},
        )

    handlers = {
        "start": _on_start,
        "stop": _on_stop,
        "set_doppler": _on_set_doppler,
        "transmit_frame": _on_transmit_frame,
    }
    try:
        await run_command_loop(sockets.control_reader, handlers)
    finally:
        stop_requested.set()
        if started.is_set():
            try:
                tb.stop()
                tb.wait()
            except Exception:
                log.exception("tb.stop/wait raised")
        await sockets.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="amateur_fm_narrowband_tx",
        description="Amateur narrowband FM transmitter (Phase 5/6 real GR, test-tone).",
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

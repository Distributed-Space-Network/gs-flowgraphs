"""Multi-mission decode bridge: gr-satellites -> the spawn contract (bench engine).

Makes the ground station "prepared for everything" by delegating to **gr-satellites**
(GPLv3) — the canonical library of public satellite framers/deframers (AX.25,
AX.100, Mobitex, CCSDS, GOMspace, EnduroSat, …) across AFSK/FSK/GFSK/BPSK/GMSK/…,
SatYAML-driven. We integrate it rather than reimplement it (see Document F).

This module is imported ONLY when a pass selects a gr-satellites `satellite`, so it
may import ``satellites`` (gr-satellites) at load. It builds a GNU Radio flowgraph
that demodulates + deframes the selected satellite and forwards each decoded frame
(PDU) to the data/status sockets, exactly like ``cubesat_gfsk_ax25_rx.py`` emits
``frame_received``.

Licensing: gr-satellites is GPLv3 (compatible with this repo). Do NOT pull in
gr-satnogs / satnogs-client (AGPLv3). Depend on gr-satellites (apt/pip, pinned);
do not vendor its source.

Status: BENCH-PENDING — not runnable on the dev box (needs GNU Radio + gr-satellites
+ gr-soapy). The gr-satellites embedding API has shifted across versions; the
construction call below is the documented shape and must be confirmed against the
installed version on the bench (as with ``gnuradio_gfsk.py``).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import logging
import queue
import re

from _fallback_select import (  # pure, testable, no GNU Radio
    CHANNEL_OVERSAMPLE,
    channel_rate_for,
    fallback_modes,
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
from gnuradio import gr

# gr-satellites flowgraph component. Import name/shape may vary by version
# (e.g. ``satellites.core.gr_satellites_flowgraph``); confirm on the bench.
from satellites.core import gr_satellites_flowgraph

_log = logging.getLogger("gr_satellites_rx")
# When gr-satellites can't decode a bird, fallback demods run in parallel (all tap one
# decimator) — frames come from whichever locks. The SET is chosen by
# ``_fallback_select.fallback_modes`` (backend symbol_rate/modulation when known, else the
# full bank); GS_FALLBACK_DEMODS overrides. Covers GFSK/FSK/GMSK, BPSK/QPSK/PSK, AFSK.


class _FrameSink(gr.basic_block):
    """Collects decoded frame PDUs (gr-satellites' message output) into a queue."""

    def __init__(self) -> None:
        gr.basic_block.__init__(self, name="frame_sink", in_sig=None, out_sig=None)
        self._q: queue.Queue[bytes] = queue.Queue()
        self.message_port_register_in(gr.pmt.intern("in"))
        self.set_msg_handler(gr.pmt.intern("in"), self._on_msg)

    def _on_msg(self, msg) -> None:  # type: ignore[no-untyped-def]
        # gr-satellites emits a PDU: (metadata, u8-vector). Extract the bytes.
        payload = gr.pmt.cdr(msg)
        data = bytes(gr.pmt.u8vector_elements(payload))
        self._q.put(data)

    def drain(self) -> list[bytes]:
        out: list[bytes] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class _SatContext:
    def __init__(
        self,
        tb: gr.top_block,
        src,
        sink: _FrameSink,
        center_hz: float,
        recorder=None,
        lo_offset_hz: float = 0.0,
        *,
        fallbacks=None,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._recorder = recorder
        # When gr-satellites has no decoder for this bird, ``fallbacks`` is a list of
        # _FallbackDemod tapping the channel in parallel; frames come from any of them. The
        # real-time graph runs only the backend's TARGETED demod(s) — it must keep up with
        # the SDR or drop samples. The exhaustive bank runs POST-PASS on the recorded .cf32
        # (``decode_file``), which has no real-time deadline. (Earlier we escalated to the
        # full bank on the live graph; on a constrained SoC that overran the RX DMA →
        # BUF_OVF + starved Doppler retuning. Post-pass decode replaces it.)
        self._fallbacks = list(fallbacks or [])

    @property
    def framing(self) -> str:
        return "fallback" if self._fallbacks else "grsatellites"

    def start(self) -> None:
        self.tb.start()

    def stop(self) -> None:
        # Just stop the graph. The cf32 is on disk (unbuffered sink); the view artifacts
        # are derived AFTER the pass by gs-client (iq_views on the .cf32), so a slow/hung
        # gr-soapy teardown can't cost us the recording or the views.
        self.tb.stop()
        self.tb.wait()

    def wait(self) -> None:
        self.tb.wait()

    def drain_frames(self) -> list[bytes]:
        # gr-satellites PDUs (empty when the sink is unconnected in fallback mode) PLUS
        # every targeted fallback demod — frames come from whichever locks.
        frames: list[bytes] = list(self._sink.drain())
        for fb in list(self._fallbacks):
            frames.extend(fb.drain_frames())
        return frames

    def set_doppler(self, offset_hz: float) -> None:
        retune_source(self.src, self._center, self._lo_offset, offset_hz)


class _FallbackDemod:
    """One fallback demod tapping the SDR source: a demodulator chain + a deframer.
    ``drain_frames`` returns the bytes of any frames recovered since the last call."""

    def __init__(self, name: str, sink, deframe) -> None:
        self.name = name
        self._sink = sink
        self._deframe = deframe

    def drain_frames(self) -> list[bytes]:
        return self._deframe(self._sink)


def _bits_deframe(bit_sink) -> list[bytes]:
    """Modulation-agnostic deframer: turn a demod's hard bits into frames. Tries AX.25
    both descrambled (G3RUH) and plain, then the EnduroSat chip packet (both bit
    polarities). The HDLC flag + FCS/CRC gate false positives, so trying everything is
    safe — only CRC-valid frames come back. Used by every fallback demod."""
    import numpy as np  # noqa: PLC0415

    from gfsk_ax25 import endurosat_link, framing  # noqa: PLC0415

    bits = bit_sink.drain()
    if not len(bits):
        return []
    frames: list[bytes] = []
    for scramble in (True, False):  # G3RUH-scrambled AX.25, then plain AX.25
        frames.extend(framing.decode(bits, scramble=scramble, nrzi=True))
    if frames:
        return frames
    arr = np.asarray(bits, dtype=np.uint8)
    return endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)


# Modulation kind -> demod builder. ``_build_fallbacks`` parses "<kind><rate>"
# (e.g. gfsk9600, gmsk4800, bpsk1200, qpsk9600, afsk1200) and dispatches here.
_PSK_ORDER = {"bpsk": 2, "psk": 2, "qpsk": 4}


def _build_fallbacks(
    tb, demod_src, sample_rate: float, params=None, modes=None
) -> list[_FallbackDemod]:
    """Build the selected fallback demods, all tapping ``demod_src`` (already at the
    channel rate, so they fan out from a single decimator). The set is ``modes`` if given
    (an explicit ``"<kind><rate>"`` list, used by runtime escalation), else chosen by
    ``_fallback_select.fallback_modes`` (backend mode if known, else the full bank). Covers
    the frame-producing modulations of 401 MHz LEO cubesats: GFSK / FSK / GMSK, BPSK /
    QPSK / PSK, AFSK. All share the modulation-agnostic ``_bits_deframe``."""
    from gnuradio_gfsk import (  # noqa: PLC0415
        connect_afsk_demod,
        connect_gfsk_demod,
        connect_psk_demod,
    )

    from gfsk_ax25 import endurosat  # noqa: PLC0415

    if modes is None:
        modes = fallback_modes(params)
    out: list[_FallbackDemod] = []
    for raw in modes:
        mode = raw.strip().lower()
        if not mode:
            continue
        m = re.match(r"([a-z]+)(\d*)", mode)
        kind = m.group(1) if m else ""
        rate = float(m.group(2)) if (m and m.group(2)) else 0.0
        # Each builder is guarded: a demod that can't be built for this channel (e.g.
        # symbol_sync needs sps>1, so the rate exceeds ~sample_rate/2) must NOT crash the
        # engine or cost us the IQ recording — skip it and keep the others.
        try:
            if kind in ("gfsk", "fsk", "gmsk"):  # 2-FSK family (GMSK = h≈0.5)
                mod_index = 0.5 if kind == "gmsk" else endurosat.LinkProfile().mod_index
                profile = endurosat.LinkProfile(symbol_rate_hz=rate or 9600.0, mod_index=mod_index)
                sink = connect_gfsk_demod(
                    tb, demod_src, sample_rate, profile, decimate=False, sdr_rate=sample_rate
                )
            elif kind in _PSK_ORDER:  # BPSK / QPSK / PSK (coherent, differential)
                sink = connect_psk_demod(
                    tb, demod_src, sample_rate, rate or 1200.0, order=_PSK_ORDER[kind]
                )
            elif kind == "afsk":  # Bell-202 1200/2200 Hz
                sink = connect_afsk_demod(tb, demod_src, sample_rate, baud=rate or 1200.0)
            else:
                _log.warning("fallback demod %r not implemented; skipping", mode)
                continue
        except Exception as e:  # noqa: BLE001 — one bad demod must not sink the rest/recording
            _log.warning("fallback demod %r failed to build (%s); skipping", mode, e)
            continue
        out.append(_FallbackDemod(mode, sink, _bits_deframe))
    return out


def _gr_satellites_selector(satellite) -> dict | None:
    """The gr-satellites SatYAML key for ``satellite`` — a NORAD id (canonical,
    unambiguous) when it is purely numeric, else a name. Returns None for an empty /
    non-numeric-garbage id so we never hand gr-satellites a bogus string."""
    s = str(satellite or "").strip()
    if not s:
        return None
    if s.isdigit():
        return {"norad": int(s)}
    return {"name": s}


def build_satellites_rx(
    args, satellite: str, sample_rate: float, params: dict | None = None
) -> _SatContext:
    """Build an RX flowgraph for ``satellite``: gr-satellites if it has a SatYAML
    decoder for the bird, otherwise the configured fallback demods. Either way the
    wideband IQ is recorded (the priority), so an unknown bird still yields a capture.

    ``satellite`` is normally the pass's NORAD id (``satellite.noradId``); we pass it to
    gr-satellites as a clean ``norad=`` int (or ``name=`` for a non-numeric id) — never
    a bogus string. If gr-satellites raises (not in its catalog / API drift), we switch
    to the fallback demods (GS_FALLBACK_DEMODS; default GFSK 9k6 + 4k8).

    BENCH-PENDING: confirm the gr_satellites_flowgraph constructor signature and the
    decoded-frame message port name against the installed gr-satellites version.
    """
    env = sdr_env()  # station-wide GS_SDR_* (antenna/gain/lo-offset/ppm/dc-removal/rate)
    # The SDR samples at the capture rate (XTRX can't stream the narrow channel rate), so
    # decimate to the CHANNEL rate ONCE and feed everything (recorder, gr-satellites, the
    # fallback demods) from it. The channel must be wide enough for the bird's symbol rate
    # (≥ a few samples/symbol) — a 50 kBd bird needs more than the 48 kHz default, else
    # symbol_sync gets sps<1 — so size it from the backend's symbol_rate_hz, capped at the
    # capture rate. Low-baud birds stay at the requested --sample-rate (~MB/min recording).
    sym = float((params or {}).get("symbol_rate_hz") or 0.0)
    want_channel = max(float(sample_rate), CHANNEL_OVERSAMPLE * sym)
    sdr_rate, _ = capture_plan(env["capture_rate_hz"], want_channel)
    channel_rate = channel_rate_for(float(sample_rate), sym, sdr_rate)
    decimate = channel_rate < sdr_rate
    # AUTO LO offset: dodge the DC/LO spike off the bird (no per-pass config — we know the
    # frequency). tune_source keeps the signal at DC and the spike at +offset, which the
    # decimator filters out. Honors an explicit GS_SDR_LO_OFFSET.
    lo = auto_lo_offset(sdr_rate, channel_rate, env["lo_offset_hz"])
    tb = gr.top_block("gr_satellites_rx")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo)  # LO offset → DC spike off-signal
    configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])

    chan = src
    if decimate:
        chan = make_decimator(sdr_rate, channel_rate)
        tb.connect(src, chan)

    # Pre-demod IQ capture FIRST (the priority): it taps the channel independently of the
    # decoder, so a decoder problem never costs us the recording. At the CHANNEL rate.
    recorder = PassRecorder.maybe_start(args, tb, chan, sample_rate_hz=channel_rate)

    # gr-satellites decoder if the bird is in its catalog; else fall back. ``flowgraph``
    # is instantiated DIRECTLY (no ``.make()``) as a hier block exposing an 'out' PDU.
    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    selector = _gr_satellites_selector(satellite)
    try:
        if selector is None:
            raise ValueError(f"no usable satellite id {satellite!r}")
        flowgraph = gr_satellites_flowgraph(
            samp_rate=channel_rate,  # the channel rate; gr-satellites resamples to the bird
            iq=True,
            grc_block=True,
            **selector,
        )
        tb.connect(chan, flowgraph)
        tb.msg_connect(flowgraph, "out", sink, "in")
        _log.info("gr-satellites: decoding %s (%r) @ %.0f Hz", satellite, selector, channel_rate)
    except Exception as e:  # noqa: BLE001 — any build failure → fall back, keep the IQ
        # Not in gr-satellites' catalog (or API drift): fan the fallback demods out from
        # the same channel decimator.
        _log.warning("gr-satellites: no decoder for %s (%s) → fallback demods", satellite, e)
        fallbacks = _build_fallbacks(tb, chan, channel_rate, params)
        _log.info(
            "fallback demods @ %.0f Hz (symbol_rate=%s): %s",
            channel_rate,
            (params or {}).get("symbol_rate_hz", "?"),
            ", ".join(f.name for f in fallbacks) or "(none)",
        )
    return _SatContext(
        tb,
        src,
        sink,
        float(args.center_freq_hz),
        recorder,
        lo_offset_hz=lo,
        fallbacks=fallbacks,
    )


def decode_file(
    cf32_path,
    *,
    sample_rate_hz: float,
    satellite: str = "",
    params: dict | None = None,
    modes: list[str] | None = None,
) -> list[tuple[str, bytes]]:
    """POST-PASS decode of a recorded ``.cf32``: gr-satellites (if the bird is in its
    catalog) plus the demod bank, run over the whole capture off a file source.

    This is where decode belongs — NOT the real-time graph. A file source runs to EOF as
    fast as the CPU allows, with no SDR to keep up with, so there is no BUF_OVF / dropped
    samples / starved Doppler. gs-client runs it after teardown (the SDR is freed), like
    ``iq_views`` for the views.

    The demods are the backend's TARGETED set by default (``fallback_modes(params)`` — the
    optimal path: we use the symbol rate/modulation we were given, not a brute-force sweep,
    so a targeted decode at 48 ksps finishes well inside the gap before the next pass). Pass
    ``modes`` to force a specific set (e.g. a wider sweep when there is idle time). Returns
    ``[(demod_name, frame_bytes)]`` for every CRC-valid frame. BENCH-PENDING: needs GNU
    Radio + gr-satellites (bench only)."""
    from gnuradio import blocks  # noqa: PLC0415 — bench-only

    tb = gr.top_block("iq_decode")
    src = blocks.file_source(gr.sizeof_gr_complex, str(cf32_path), repeat=False)
    sink = _FrameSink()
    selector = _gr_satellites_selector(satellite)
    if selector is not None:
        try:
            flowgraph = gr_satellites_flowgraph(
                samp_rate=float(sample_rate_hz), iq=True, grc_block=True, **selector
            )
            tb.connect(src, flowgraph)
            tb.msg_connect(flowgraph, "out", sink, "in")
        except Exception as e:  # noqa: BLE001 — not in catalog / API drift → bank only
            _log.warning("post-pass: gr-satellites has no decoder for %s (%s)", satellite, e)
    pick = modes if modes is not None else list(dict.fromkeys(fallback_modes(params)))
    fallbacks = _build_fallbacks(tb, src, float(sample_rate_hz), modes=pick)
    _log.info(
        "post-pass decode: %d demods over %s @ %.0f Hz",
        len(fallbacks), cf32_path, float(sample_rate_hz),
    )
    tb.run()  # blocks until the file is exhausted — offline, no SDR, no real-time deadline
    results: list[tuple[str, bytes]] = [("grsatellites", f) for f in sink.drain()]
    for fb in fallbacks:
        results.extend((fb.name, fr) for fr in fb.drain_frames())
    return results


__all__ = ["build_satellites_rx", "decode_file"]

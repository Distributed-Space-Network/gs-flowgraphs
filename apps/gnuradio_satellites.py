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

from _fallback_select import fallback_modes  # pure mode-selection (testable, no GNU Radio)
from _recorder import PassRecorder
from _soapy import (
    apply_corrections,
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
        chan=None,
        sample_rate: float = 0.0,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._recorder = recorder
        # When gr-satellites has no decoder for this bird, ``fallbacks`` is a list of
        # _FallbackDemod tapping ``chan`` in parallel; frames come from any of them.
        self._fallbacks = list(fallbacks or [])
        self._chan = chan if chan is not None else src  # demod tap point for escalation
        self._sample_rate = sample_rate
        self._escalated = False

    @property
    def framing(self) -> str:
        return "fallback" if self._fallbacks else "grsatellites"

    def escalate_fallbacks(self) -> None:
        """No frames decoded → add the full fallback bank to the RUNNING graph (the
        backend's targeted mode / gr-satellites didn't lock). Idempotent + best-effort:
        wrapped so a reconfiguration hiccup can't break the recording or the pass; the
        native cf32 sink is independent of this. BENCH-PENDING: runtime tb.lock()/unlock()
        reconfiguration — validate on hardware."""
        if self._escalated:
            return
        self._escalated = True
        try:
            running = {f.name for f in self._fallbacks}
            modes = [m for m in fallback_modes(None) if m.strip().lower() not in running]
            if not modes:
                return
            self.tb.lock()
            try:
                added = _build_fallbacks(self.tb, self._chan, self._sample_rate, modes=modes)
            finally:
                self.tb.unlock()
            self._fallbacks.extend(added)
            _log.warning(
                "no frames yet → escalated to fallback bank (+%d demods: %s)",
                len(added),
                ", ".join(f.name for f in added) or "(none)",
            )
        except Exception:
            _log.exception("fallback escalation failed; continuing with current demods")

    def start(self) -> None:
        self.tb.start()

    def stop(self) -> None:
        self.tb.stop()
        # Finalize BEFORE tb.wait(): the native cf32 sink has already flushed to disk, and
        # gr-soapy's source can hang tb.wait() (→ SIGTERM), which would otherwise skip the
        # PNG/CSV derivation. finalize reads the on-disk cf32, so it needs no running graph.
        if self._recorder is not None:
            self._recorder.finalize()
        self.tb.wait()

    def wait(self) -> None:
        self.tb.wait()

    def drain_frames(self) -> list[bytes]:
        # gr-satellites PDUs (empty when the sink is unconnected in fallback mode) PLUS
        # every fallback demod — so escalation works whichever was primary. Snapshot the
        # list: escalate_fallbacks() may extend it from a worker thread.
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
    lo = env["lo_offset_hz"]
    # The SDR samples at the capture rate (XTRX can't stream the narrow channel rate), so
    # decimate to the CHANNEL rate ONCE and feed everything (recorder, gr-satellites, the
    # fallback demods) from it. This is what makes the recording the narrow channel
    # (~MB/min) instead of the multi-GB wideband capture.
    sdr_rate, decimate = capture_plan(env["capture_rate_hz"], float(sample_rate))
    tb = gr.top_block("gr_satellites_rx")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo)  # LO offset → DC spike off-signal
    configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])

    chan = src
    if decimate:
        chan = make_decimator(sdr_rate, float(sample_rate))
        tb.connect(src, chan)

    # Pre-demod IQ capture FIRST (the priority): it taps the channel independently of the
    # decoder, so a decoder problem never costs us the recording. At the CHANNEL rate.
    recorder = PassRecorder.maybe_start(args, tb, chan, sample_rate_hz=float(sample_rate))

    # gr-satellites decoder if the bird is in its catalog; else fall back. ``flowgraph``
    # is instantiated DIRECTLY (no ``.make()``) as a hier block exposing an 'out' PDU.
    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    selector = _gr_satellites_selector(satellite)
    try:
        if selector is None:
            raise ValueError(f"no usable satellite id {satellite!r}")
        flowgraph = gr_satellites_flowgraph(
            samp_rate=float(sample_rate),  # the channel rate; gr-satellites resamples to the bird
            iq=True,
            grc_block=True,
            **selector,
        )
        tb.connect(chan, flowgraph)
        tb.msg_connect(flowgraph, "out", sink, "in")
        _log.info("gr-satellites: decoding %s (%r) @ %.0f Hz", satellite, selector, sample_rate)
    except Exception as e:  # noqa: BLE001 — any build failure → fall back, keep the IQ
        # Not in gr-satellites' catalog (or API drift): fan the fallback demods out from
        # the same channel decimator.
        _log.warning("gr-satellites: no decoder for %s (%s) → fallback demods", satellite, e)
        fallbacks = _build_fallbacks(tb, chan, float(sample_rate), params)
        _log.info(
            "fallback demods @ %.0f Hz (symbol_rate=%s): %s",
            sample_rate,
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
        chan=chan,
        sample_rate=float(sample_rate),
    )


__all__ = ["build_satellites_rx"]

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

import contextlib
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
        valve_ours=None,
        valve_grsat=None,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._recorder = recorder
        # Frames come from gr-satellites (``sink``) and/or our own demod (``fallbacks``, one
        # demod for the bird's known mode). With demod params present AND the bird catalogued
        # we run BOTH (each behind a valve), and the FIRST to produce a CRC-valid frame wins:
        # we gate off the loser's valve so its chain starves (the SDR stream is shared — one
        # open, fanned out — so there is no hardware conflict, only CPU, and two chains is
        # cheap). Frames are deduped across both.
        self._fallbacks = list(fallbacks or [])
        self._valve_ours = valve_ours
        self._valve_grsat = valve_grsat
        self._winner: str | None = None
        self._seen: set[bytes] = set()

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
        # gr-satellites PDUs and our demod, kept separate so we can tell who decoded FIRST.
        gr_frames = list(self._sink.drain())
        our_frames: list[bytes] = []
        for fb in list(self._fallbacks):
            our_frames.extend(fb.drain_frames())
        # First CRC-valid frame wins; gate off the loser's valve so its chain starves. Only
        # when both ran (both valves present). Idempotent.
        if self._winner is None and (self._valve_ours or self._valve_grsat):
            if gr_frames:
                self._winner = "grsatellites"
                self._gate_off(self._valve_ours, "our engine")
            elif our_frames:
                self._winner = "ours"
                self._gate_off(self._valve_grsat, "gr-satellites")
        fresh: list[bytes] = []
        for f in (*gr_frames, *our_frames):
            if f not in self._seen:
                self._seen.add(f)
                fresh.append(f)
        return fresh

    def _gate_off(self, valve, name: str) -> None:
        if valve is None:
            return
        with contextlib.suppress(Exception):  # bench-pending: blocks.copy disabled drains input
            valve.set_enabled(False)
        _log.info("%s won; gated off %s for the rest of the pass", self._winner, name)

    def set_doppler(self, offset_hz: float) -> None:
        retune_source(self.src, self._center, self._lo_offset, offset_hz)


class _FallbackDemod:
    """One fallback demod tapping the SDR source: a demodulator chain + a deframer.
    ``drain_frames`` returns the bytes of any frames recovered since the last call."""

    def __init__(self, name: str, sink, deframe, framing: str | None = None) -> None:
        self.name = name
        self._sink = sink
        self._deframe = deframe
        self._framing = (framing or "").strip().lower() or None  # backend hint (locks the protocol)
        self._locked: str | None = None  # framing discovered this pass when no hint was given

    def drain_frames(self) -> list[bytes]:
        # Backend gave the framing → use only that. Otherwise try all, and once one matches,
        # LOCK to it for the rest of the pass (stop frame-matching).
        use = self._framing or self._locked
        frames, matched = self._deframe(self._sink, use)
        if matched and self._framing is None and self._locked is None:
            self._locked = matched
        return frames


_FRAMINGS = ("ax25", "endurosat")  # link layers our engine knows; gr-satellites does its own


def _bits_deframe(bit_sink, framing_name: str | None = None) -> tuple[list[bytes], str | None]:
    """Deframe a demod's hard bits → ``(frames, matched_framing)``. ``framing_name`` runs ONLY
    that link layer (the backend told us). When None, try every known framing — we don't know
    the link layer — and report which one matched so the caller can lock onto it. Every framing
    is CRC/FCS-gated, so trying several is safe (a wrong one has a ~1/65536-per-flag spurious
    chance, which is why locking once matched is worth it)."""
    import numpy as np  # noqa: PLC0415

    from gfsk_ax25 import endurosat_link, framing  # noqa: PLC0415

    bits = bit_sink.drain()  # consume once; try framings against the same buffer
    if not len(bits):
        return [], None
    order = [framing_name.strip().lower()] if framing_name else list(_FRAMINGS)
    for name in order:
        if name == "endurosat":
            arr = np.asarray(bits, dtype=np.uint8)
            frames = endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)
        elif name == "ax25":  # both G3RUH-descrambled and plain — same framing, just descrambling
            frames = []
            for scramble in (True, False):
                frames.extend(framing.decode(bits, scramble=scramble, nrzi=True))
        else:
            continue
        if frames:
            return frames, name
    return [], None


# Modulation kind -> demod builder. ``_build_fallbacks`` parses "<kind><rate>"
# (e.g. gfsk9600, gmsk4800, bpsk1200, qpsk9600, afsk1200) and dispatches here.
_PSK_ORDER = {"bpsk": 2, "psk": 2, "qpsk": 4}


def _build_fallbacks(
    tb, demod_src, sample_rate: float, params=None, modes=None, framing=None
) -> list[_FallbackDemod]:
    """Build the demod(s) tapping ``demod_src`` (already at the channel rate). Normally this
    is the ONE demod the backend specified — ``modes=["<kind><rate>"]`` from the transmitter
    record (modulation + symbol_rate) — deframed with the backend ``framing``. Falls back to
    ``fallback_modes(params)`` only when no explicit mode is given. Covers the frame-producing
    modulations of 401 MHz LEO cubesats: GFSK / FSK / GMSK, BPSK / QPSK / PSK, AFSK."""
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
        out.append(_FallbackDemod(mode, sink, _bits_deframe, framing))
    return out


def _backend_mode(params: dict | None) -> str | None:
    """The single "<kind><rate>" the backend specified, from the transmitter record's
    modulation + symbol_rate_hz (e.g. {"modulation":"gfsk","symbol_rate_hz":2400} ->
    "gfsk2400"). None when either is absent — caller then runs gr-satellites only."""
    p = params or {}
    kind = str(p.get("modulation") or "").strip().lower()
    rate = p.get("symbol_rate_hz")
    if not kind or not rate:
        return None
    return f"{kind}{int(float(rate))}"


def _build_grsatellites(selector, channel_rate: float, satellite):
    """Instantiate the gr-satellites flowgraph for ``satellite`` (by NORAD) or return None if
    it has no decoder (not catalogued / API drift) — non-fatal; our engine + the recording
    carry on. The caller wires it (so it can insert a valve first for the parallel race)."""
    if selector is None:
        return None
    try:
        fg = gr_satellites_flowgraph(
            samp_rate=channel_rate, iq=True, grc_block=True, **selector  # gr-satellites resamples
        )
        _log.info("gr-satellites: decoder for %s (%r) @ %.0f Hz", satellite, selector, channel_rate)
        return fg
    except Exception as e:  # noqa: BLE001 — not catalogued / API drift
        _log.info("gr-satellites: no decoder for %s (%s)", satellite, e)
        return None


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

    # Engine selection (per backend params). Both tap the SAME channel stream (one SDR open,
    # fanned out — no hardware conflict; the dynamic SDR control, Doppler, is ephemeris-driven
    # on the shared source, identical for both). CPU is the only shared cost, and two chains is
    # cheap (the 12-demod bank is what overran the RX DMA, not two):
    #   * demod params present AND bird catalogued → BOTH, each behind a valve; the first to
    #     produce a CRC-valid frame wins and the loser's valve is gated off (see drain_frames).
    #   * demod params present, not catalogued → OUR engine only.
    #   * only a NORAD / a demod param missing → gr-satellites only.
    from gnuradio import blocks  # noqa: PLC0415 — bench-only

    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    valve_ours = valve_grsat = None
    selector = _gr_satellites_selector(satellite)
    framing = (params or {}).get("framing")
    mode = _backend_mode(params)  # "<kind><rate>" when modulation+symbol_rate both present
    fg = _build_grsatellites(selector, channel_rate, satellite)  # None if not catalogued
    if mode and fg is not None:  # race both
        valve_ours = blocks.copy(gr.sizeof_gr_complex)
        valve_grsat = blocks.copy(gr.sizeof_gr_complex)
        tb.connect(chan, valve_ours)
        tb.connect(chan, valve_grsat, fg)
        tb.msg_connect(fg, "out", sink, "in")
        fallbacks = _build_fallbacks(tb, valve_ours, channel_rate, modes=[mode], framing=framing)
        _log.info("racing: our engine %s + gr-satellites %r (first frame wins)", mode, selector)
    elif mode:  # our engine only (not catalogued)
        fallbacks = _build_fallbacks(tb, chan, channel_rate, modes=[mode], framing=framing)
        _log.info("our engine: %s @ %.0f Hz (framing=%s)", mode, channel_rate, framing or "auto")
    elif fg is not None:  # gr-satellites only (no demod params)
        tb.connect(chan, fg)
        tb.msg_connect(fg, "out", sink, "in")
        _log.info("gr-satellites only for %s (no demod params)", satellite)
    else:
        _log.warning("no decode: no demod params and no gr-satellites decoder for %r", satellite)
    return _SatContext(
        tb,
        src,
        sink,
        float(args.center_freq_hz),
        recorder,
        lo_offset_hz=lo,
        fallbacks=fallbacks,
        valve_ours=valve_ours,
        valve_grsat=valve_grsat,
    )


__all__ = ["build_satellites_rx"]

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
import os
import queue
import tempfile

import compose  # decode composer (plan + race decision); numpy-only, import-safe
import framings  # framing registry (deframe dispatch); numpy-only, import-safe
import numpy as np
import pmt  # PMT is a standalone top-level module in GNU Radio 3.10 (NOT gr.pmt)
from _fallback_select import (  # pure, testable, no GNU Radio
    CHANNEL_OVERSAMPLE,
    channel_rate_for,
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
# Decode is fully backend-driven: demod params present -> the ONE backend-specified demod
# (built via the modem registry); only a NORAD -> gr-satellites alone. There is no
# brute-force fallback bank (GS_FALLBACK_DEMODS is deprecated and unused).


class _FrameSink(gr.basic_block):
    """Collects decoded frame PDUs (gr-satellites' message output) into a queue."""

    def __init__(self) -> None:
        gr.basic_block.__init__(self, name="frame_sink", in_sig=None, out_sig=None)
        self._q: queue.Queue[bytes] = queue.Queue()
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self._on_msg)

    def _on_msg(self, msg) -> None:  # type: ignore[no-untyped-def]
        # gr-satellites emits a PDU: (metadata, u8-vector). Extract the bytes.
        payload = pmt.cdr(msg)
        data = bytes(pmt.u8vector_elements(payload))
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

    def drain_frames(self) -> list[tuple[str, bytes]]:
        # Each frame tagged with the engine that produced it (kept separate so we know who
        # decoded). ``our_frames`` carry the demod name (e.g. "gfsk2400"); gr-satellites PDUs
        # are "gr-satellites".
        our_frames: list[tuple[str, bytes]] = []
        our_matched: list[str] = []  # framings that produced our NEW frames (race gating input)
        for fb in list(self._fallbacks):
            got = fb.drain_frames()
            our_frames.extend((fb.name, f) for f in got)
            if fb.race_framing is not None:
                our_matched.append(fb.race_framing)
        gr_frames: list[tuple[str, bytes]] = [("gr-satellites", f) for f in self._sink.drain()]
        # Race: the first to produce a CRC-valid frame wins; gate off the loser. Only while
        # both ran (both valves set). The decision is compose.race_winner (pure, unit-tested):
        # only a CRC/FCS/RS-gated framing may declare OUR win — checksum-less KISS "frames"
        # are products but never gate off gr-satellites (docs/10 MED-1). Ties within one
        # drain go to OUR engine (the backend-specified primary). Idempotent.
        if self._winner is None and self._valve_ours is not None and self._valve_grsat is not None:
            winner = compose.race_winner(our_matched, bool(gr_frames))
            if winner == "ours":
                self._winner = "ours"
                self._gate_off(self._valve_grsat, "gr-satellites")
            elif winner == "grsatellites":
                self._winner = "grsatellites"
                self._gate_off(self._valve_ours, "our engine")
        # Dedup WITHIN this drain only (a frame both engines decoded in the same window) — NOT
        # across drains, so genuine repeat beacons (identical payloads over time) are kept.
        fresh: list[tuple[str, bytes]] = []
        seen: set[bytes] = set()
        for source, f in (*our_frames, *gr_frames):
            if f not in seen:
                seen.add(f)
                fresh.append((source, f))
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

    _LOCK_AFTER = 2  # matches of the SAME framing before locking (one CRC hit can be spurious)
    _TAIL_BITS = 4096  # carry-over so a frame straddling a drain boundary isn't lost (~2 AX.25)

    def __init__(self, name: str, sink, framing: str | None = None) -> None:
        self.name = name
        self._sink = sink
        # Backend framing hint, VERBATIM (SatYAML label or local token) — framings.deframe
        # normalizes it to a local deframer; unknown labels deframe upstream (gr-satellites).
        self._framing = (framing or "").strip() or None
        self._locked: str | None = None  # framing discovered this pass when no hint was given
        self._hits: dict[str, int] = {}  # per-framing match count, to lock only on a confident one
        self._tail = np.empty(0, dtype=np.uint8)  # bits carried across drain boundaries
        # The LOCAL framing that produced the last drain's NEW frames (None if none). The race
        # gate feeds this to compose.race_winner: only a CRC-gated framing may win (MED-1).
        self.race_framing: str | None = None

    def drain_frames(self) -> list[bytes]:
        # Backend gave the framing → use only that. Otherwise try all; once ONE framing has
        # matched _LOCK_AFTER times (a single CRC hit can be a ~1/65536 fluke), lock to it for
        # the rest of the pass so a spurious early match can't strand the real framing.
        use = self._framing or self._locked
        fresh = self._sink.drain()
        prev_tail = self._tail
        bits = np.concatenate([prev_tail, fresh]) if prev_tail.size else fresh
        self._tail = bits[-self._TAIL_BITS:].copy() if bits.size else self._tail
        frames, matched = framings.deframe(bits, use)
        # POSITIONAL dedup of the carry-over: frames decodable from the carried tail ALONE were
        # already returned last drain — subtract exactly those, WITH multiplicity. (A payload-set
        # dedup would permanently suppress genuine repeat beacons: an identical beacon re-decodes
        # out of the tail every drain, refreshing the set forever.)
        out = list(frames)
        if prev_tail.size:
            already, _ = framings.deframe(prev_tail, use)
            for f in already:
                if f in out:
                    out.remove(f)  # one occurrence per tail re-decode
        self.race_framing = matched if out else None  # what produced the NEW frames (race input)
        # Lock-counting uses only NEW frames: a single CRC fluke re-decoding out of the tail
        # must not count twice and defeat the two-independent-hits guard.
        if matched and out and self._framing is None and self._locked is None:
            self._hits[matched] = self._hits.get(matched, 0) + 1
            if self._hits[matched] >= self._LOCK_AFTER:
                self._locked = matched
        return out


# Deframing (``framings.deframe``) and the modulation→demod dispatch (``modem.build_demod``)
# live in the framing/modem registries now (docs/08 — universal modem + framing).
# ``_build_fallbacks`` just composes them: (modulation, rate) → build demod → wrap with the
# deframer. New modulations/framings register in modem.py/framings.py, not here.


def _build_fallbacks(
    tb, demod_src, sample_rate: float, modes=None, framing=None, differential=None
) -> list[_FallbackDemod]:
    """Build the demod(s) tapping ``demod_src`` (already at the channel rate). ``modes`` is a
    list of ``(modulation, symbol_rate)`` tuples — normally the ONE the backend specified from
    the transmitter record — deframed with the backend ``framing`` (verbatim label; the framing
    registry normalizes). ``differential`` (bool | None) threads the backend's DxPSK flag to the
    PSK demod. Modulation coverage comes from the modem registry (``modem.build_demod``)."""
    import modem  # noqa: PLC0415 — lazy: pulls in gnuradio_gfsk (GNU Radio) only at decode time

    out: list[_FallbackDemod] = []
    for kind, rate in modes or []:
        kind = str(kind or "").strip().lower()
        if not kind:
            continue
        # Guarded: a demod that can't be built for this channel (e.g. symbol_sync needs sps>1,
        # so the rate exceeds ~sample_rate/2) must NOT crash the engine or cost us the IQ
        # recording — skip it and keep the others.
        try:
            sink = modem.build_demod(
                kind, tb, demod_src, sample_rate, float(rate or 0.0), differential=differential)
        except Exception as e:  # noqa: BLE001 — one bad demod must not sink the rest/recording
            _log.warning("fallback demod %s@%s failed to build (%s); skipping", kind, rate, e)
            continue
        if sink is None:
            _log.warning("fallback demod %s@%s not implemented; skipping", kind, rate)
            continue
        out.append(_FallbackDemod(f"{kind}{int(rate or 0)}", sink, framing))
    return out


def _backend_mode(params: dict | None) -> tuple[str, float] | None:
    """The single ``(modulation, symbol_rate)`` the backend specified from the transmitter
    record's modulation + symbol_rate_hz. None when either is absent — caller then runs
    gr-satellites only. A tuple (not a concatenated string) so digit-bearing modulation names
    (``2fsk``, ``8psk``, ``qam16``) stay unambiguous."""
    p = params or {}
    kind = str(p.get("modulation") or "").strip().lower()
    try:
        rate = float(p.get("symbol_rate_hz") or 0)
    except (TypeError, ValueError):  # a non-numeric symbol rate must not crash the engine
        return None
    if not kind or rate <= 0:
        return None
    return (kind, rate)


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


def _synthetic_satyaml_path(satellite, params: dict | None, frequency_hz: float) -> str | None:
    """Write a synthetic gr-satellites SatYAML from the backend's ``(modulation, baud, framing)``
    for a bird gr-satellites doesn't catalog, and return its path (to pass as
    ``gr_satellites_flowgraph(file=...)``) — or None when gr-satellites can't demodulate the
    modulation (QAM/APSK/OFDM/QPSK → our own modem) or a field is missing. This reuses
    gr-satellites' full demod + ~50-deframer library for NON-catalogued birds (docs/08 Ph1).
    The caller removes the temp file after the flowgraph has parsed it."""
    import grsat_synth  # noqa: PLC0415 — lazy, numpy/PyYAML only (no GNU Radio)

    p = params or {}
    s = str(satellite or "").strip()
    norad = int(s) if s.isdigit() else 0
    fd, path = tempfile.mkstemp(prefix="grsat_synth_", suffix=".yml")
    os.close(fd)
    out = grsat_synth.write_synthetic_satyaml(
        path, norad, p.get("modulation"), p.get("symbol_rate_hz"),
        p.get("framing"), frequency_hz, name=(s or None),
    )
    if out is None:
        with contextlib.suppress(OSError):
            os.remove(path)
        return None
    return out


def build_satellites_rx(
    args, satellite: str, sample_rate: float, params: dict | None = None
) -> _SatContext:
    """Build an RX flowgraph for ``satellite``: gr-satellites if it has a SatYAML
    decoder for the bird, otherwise the configured fallback demods. Either way the
    wideband IQ is recorded (the priority), so an unknown bird still yields a capture.

    ``satellite`` is normally the pass's NORAD id (``satellite.noradId``); we pass it to
    gr-satellites as a clean ``norad=`` int (or ``name=`` for a non-numeric id) — never
    a bogus string. If gr-satellites has no decoder (not catalogued and not synthesizable),
    the ONE backend-specified demod (modulation + symbol_rate from params) runs alone.

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
    #   * demod params present AND a gr-satellites decoder exists → BOTH, each behind a valve;
    #     the first to produce a CRC-valid frame wins and the loser's valve is gated off.
    #   * demod params present, no gr-satellites decoder → OUR engine only.
    #   * only a NORAD / a demod param missing → gr-satellites only.
    # The gr-satellites decoder is the catalogued SatYAML when the bird is known; otherwise, if
    # the backend gave (modulation, framing, baud), a SYNTHETIC SatYAML so we still get its full
    # ~50-deframer library for a non-catalogued bird (docs/08 Ph1).
    from gnuradio import blocks  # noqa: PLC0415 — bench-only

    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    valve_ours = valve_grsat = None
    selector = _gr_satellites_selector(satellite)
    framing = (params or {}).get("framing")
    differential = (params or {}).get("differential")
    if not isinstance(differential, bool):
        differential = None  # absent/garbage → PSK demod keeps its robust default
    mode = _backend_mode(params)  # (modulation, symbol_rate) when both present
    fg = _build_grsatellites(selector, channel_rate, satellite)  # None if not catalogued
    # Compose the registries into a decode plan (docs/08 Phase 4) for observability — which path(s)
    # the backend rfLink implies. The construction below still drives the graph; the plan is the
    # single explanation of the choice (and the seam a future satellite_rx composition builds on).
    try:
        _log.info("decode plan: %s",
                  compose.plan_decode(params, catalogued=fg is not None).describe())
    except Exception as e:  # noqa: BLE001 — planning must never block decoding
        _log.debug("decode-plan compose failed (non-fatal): %s", e)
    if fg is None and mode:  # not catalogued → synthesize a SatYAML from the backend rfLink
        synth = _synthetic_satyaml_path(satellite, params, float(args.center_freq_hz))
        if synth is not None:
            try:
                fg = _build_grsatellites({"file": synth}, channel_rate, satellite)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(synth)  # gr-satellites parsed it in __init__; safe to remove
            if fg is not None:
                _log.info("gr-satellites via synthetic SatYAML for %s (not catalogued)", satellite)
    # Spectral inversion (rfLink ``invert``): conjugate the DECODE tap only — the recorder keeps
    # the raw channel so the .cf32 is always what was actually received.
    demod_tap = chan
    if (params or {}).get("invert") is True:
        demod_tap = blocks.conjugate_cc()
        tb.connect(chan, demod_tap)
        _log.info("spectral inversion: conjugating the decode tap (recorder stays raw)")
    if mode and fg is not None:  # race both
        valve_ours = blocks.copy(gr.sizeof_gr_complex)
        valve_grsat = blocks.copy(gr.sizeof_gr_complex)
        tb.connect(demod_tap, valve_ours)
        tb.connect(demod_tap, valve_grsat, fg)
        tb.msg_connect(fg, "out", sink, "in")
        fallbacks = _build_fallbacks(
            tb, valve_ours, channel_rate, modes=[mode], framing=framing, differential=differential)
        if not fallbacks:  # demod failed to build → valve_ours must not dangle (start() aborts)
            tb.connect(valve_ours, blocks.null_sink(gr.sizeof_gr_complex))
        _log.info("racing: our engine %s@%.0f + gr-satellites %r (first frame wins)",
                  mode[0], mode[1], selector)
    elif mode:  # our engine only (no gr-satellites decoder — not catalogued, un-synthesizable)
        fallbacks = _build_fallbacks(
            tb, demod_tap, channel_rate, modes=[mode], framing=framing, differential=differential)
        _log.info("our engine: %s@%.0f on %.0f Hz channel (framing=%s)",
                  mode[0], mode[1], channel_rate, framing or "auto")
    elif fg is not None:  # gr-satellites only (no demod params)
        tb.connect(demod_tap, fg)
        tb.msg_connect(fg, "out", sink, "in")
        _log.info("gr-satellites only for %s (no demod params)", satellite)
    else:
        _log.warning("no decode: no demod params and no gr-satellites decoder for %r", satellite)
    # GNU Radio validates ALL stream ports at start(); a consumer-less tap would abort the whole
    # graph and cost us the recording. Terminate any tap that ended up without a consumer:
    #   * demod_tap: no decoder built (no-decode branch, or every fallback failed to build) — and
    #     when demod_tap is the conjugate block it ALWAYS needs a consumer (it's fed from chan);
    #   * chan: nothing at all downstream (decoders on a dead branch AND recording disabled).
    decode_consumers = bool(fallbacks) or fg is not None or valve_ours is not None
    if not decode_consumers and demod_tap is not chan:
        tb.connect(demod_tap, blocks.null_sink(gr.sizeof_gr_complex))
    elif not decode_consumers and recorder is None:
        tb.connect(chan, blocks.null_sink(gr.sizeof_gr_complex))
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

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
import tempfile
from dataclasses import dataclass

import compose  # decode composer (plan + race decision); numpy-only, import-safe
import framings  # framing registry (deframe dispatch); numpy-only, import-safe
import numpy as np
import pmt  # PMT is a standalone top-level module in GNU Radio 3.10 (NOT gr.pmt)
from _fallback_select import (  # pure, testable, no GNU Radio
    CHANNEL_OVERSAMPLE,
    channel_rate_for,
    no_decode_reason,
    symbol_rate_hz_of,
)
from _recorder import PassRecorder
from _soapy import (
    DEFAULT_LO_OFFSET_HZ,
    apply_corrections,
    auto_lo_offset,
    capture_plan,
    configure_soapy_source,
    lo_phase_inc,
    make_decimator,
    make_lo_rotator,
    make_source,
    merge_sdr_params,
    open_analog_bandwidth,
    sdr_env,
    tune_below,
)
from gnuradio import gr
from native_framing.modem_matrix import RxExecution, plan_native_rx_pairing
from native_framing.registry import build_decoder, resolve_profile
from native_framing.runtime_queue import BoundedQueue, QueueStats, require_lossless
from native_framing.shadow_runtime import ShadowReconciler, ShadowStats
from native_framing.types import FrameResult, IntegrityStatus, Polarity, SymbolInput

# gr-satellites flowgraph component. Import name/shape may vary by version
# (e.g. ``satellites.core.gr_satellites_flowgraph``); confirm on the bench.
from satellites.core import gr_satellites_flowgraph

_log = logging.getLogger("gr_satellites_rx")
_FRAME_QUEUE_CAPACITY_FRAMES = 1024
_FRAME_QUEUE_CAPACITY_BYTES = 16 * 1024 * 1024
# Decode is fully backend-driven: demod params present -> the ONE backend-specified demod
# (built via the modem registry); only a NORAD -> gr-satellites alone. There is no
# brute-force fallback bank (GS_FALLBACK_DEMODS is deprecated and unused).


@dataclass(frozen=True)
class _DecodedFrame:
    """One live frame plus the source-domain metadata the decoder can prove."""

    source: str
    payload: bytes
    framing: str = ""
    source_start: int | None = None
    source_end: int | None = None
    source_offset_kind: str = ""
    integrity: str = ""
    polarity: str = ""
    sync_distance: float | None = None
    corrected_symbols: int | None = None


def _decoded_from_result(source: str, frame: FrameResult) -> _DecodedFrame:
    offsets_available = bool(frame.metadata.get("source_offsets_available", True))
    return _DecodedFrame(
        source=source,
        payload=frame.payload,
        framing=frame.canonical_framing,
        source_start=frame.source_start if offsets_available else None,
        source_end=frame.source_end if offsets_available else None,
        source_offset_kind="demodulated_symbol" if offsets_available else "",
        integrity=frame.integrity.value,
        polarity=frame.polarity.value,
        sync_distance=frame.sync_distance,
        corrected_symbols=frame.corrected_symbols,
    )


class _FrameSink(gr.basic_block):
    """Collects decoded frame PDUs (gr-satellites' message output) into a queue."""

    def __init__(self) -> None:
        gr.basic_block.__init__(self, name="frame_sink", in_sig=None, out_sig=None)
        self._q = BoundedQueue[bytes](
            capacity_items=_FRAME_QUEUE_CAPACITY_FRAMES,
            capacity_units=_FRAME_QUEUE_CAPACITY_BYTES,
        )
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self._on_msg)

    def _on_msg(self, msg) -> None:  # type: ignore[no-untyped-def]
        # gr-satellites emits a PDU: (metadata, u8-vector). Extract the bytes.
        payload = pmt.cdr(msg)
        data = bytes(pmt.u8vector_elements(payload))
        self._q.offer(data, units=len(data))

    def drain(self) -> list[bytes]:
        stats = self._q.stats()
        require_lossless(stats, label="gr-satellites frame", unit_name="bytes")
        return self._q.drain()

    @property
    def queue_stats(self) -> QueueStats:
        return self._q.stats()


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
        rotator=None,
        sdr_rate_hz: float = 0.0,
        fallbacks=None,
        valve_ours=None,
        valve_grsat=None,
        sdr_applied: dict | None = None,
        no_decode_reason: str = "",
        shadow_enabled: bool = False,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._rotator = rotator  # software LO+Doppler NCO (Phase 1); None ⇒ no retune
        self._sdr_rate = sdr_rate_hz
        self.recorder = recorder  # public: the app's R-11 first-sample probe reads it
        self.sdr_applied = dict(sdr_applied or {})  # R-21: what configure/corrections applied
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
        self._shadow = (
            ShadowReconciler[_DecodedFrame](key=lambda frame: frame.payload)
            if shadow_enabled
            else None
        )
        # R2-02: did we build ANY decoder at all? A pass with no demod params (the backend
        # transmitter has a null/zero baud) and gr-satellites gated off degrades to a
        # RECORDER-ONLY graph — it captures IQ and produces exactly zero frames. That is a
        # legitimate outcome (the .cf32 can be decoded offline), but it must never be
        # reported as an ordinary successful decode pass: the operator sees a green pass and
        # no frames, with nothing saying why. The app puts these on the `ready` event.
        self._no_decode_reason = no_decode_reason

    @property
    def framing(self) -> str:
        return "fallback" if self._fallbacks else "grsatellites"

    @property
    def decode_built(self) -> bool:
        """False when the graph is RECORDER-ONLY (no decoder was constructed)."""
        return not self._no_decode_reason

    @property
    def no_decode_reason(self) -> str:
        """Why no decoder exists (empty when one does). Rides the `ready` event."""
        return self._no_decode_reason

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

    def drain_frames(self) -> list[_DecodedFrame]:
        # Each frame tagged with the engine that produced it (kept separate so we know who
        # decoded). ``our_frames`` carry the demod name (e.g. "gfsk2400"); gr-satellites PDUs
        # are "gr-satellites".
        our_frames: list[_DecodedFrame] = []
        our_matched: list[str] = []  # framings that produced our NEW frames (race gating input)
        for fb in list(self._fallbacks):
            got = fb.drain_frames()
            our_frames.extend(_decoded_from_result(fb.name, frame) for frame in got)
            if fb.race_framing is not None:
                our_matched.append(fb.race_framing)
        gr_frames = [_DecodedFrame(source="gr-satellites", payload=f) for f in self._sink.drain()]
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
        if self._shadow is not None:
            return self._shadow.reconcile(our_frames, gr_frames)
        return [*our_frames, *gr_frames]

    def flush_frames(self) -> list[_DecodedFrame]:
        """Finalize streaming decoders after the stopped graph has been drained."""

        output: list[_DecodedFrame] = []
        for fb in list(self._fallbacks):
            output.extend(_decoded_from_result(fb.name, frame) for frame in fb.flush_frames())
        if self._shadow is not None:
            return self._shadow.reconcile(output, [])
        return output

    def finalize_shadow(self) -> ShadowStats | None:
        """Finalize bounded comparison accounting after the decoder flush."""

        return self._shadow.finalize() if self._shadow is not None else None

    def _gate_off(self, valve, name: str) -> None:
        if valve is None:
            return
        with contextlib.suppress(Exception):  # bench-pending: blocks.copy disabled drains input
            valve.set_enabled(False)
        _log.info("%s won; gated off %s for the rest of the pass", self._winner, name)

    def set_doppler(self, offset_hz: float) -> None:
        # Software NCO retune (gs-orbitd ephemeris → rotator), NOT a hardware LO retune: the
        # rotator shifts the +lo_offset+doppler carrier to DC. No PLL settle glitch, and it
        # composes with the fixed lo_offset. No rotator (shouldn't happen) ⇒ no-op.
        if self._rotator is not None and self._sdr_rate:
            self._rotator.set_phase_inc(lo_phase_inc(self._sdr_rate, self._lo_offset, offset_hz))


class _FallbackDemod:
    """One fallback demod tapping the SDR source: a demodulator chain + a deframer.
    ``drain_frames`` returns typed results recovered since the last call."""

    _LOCK_AFTER = 2  # matches of the SAME framing before locking (one CRC hit can be spurious)
    _TAIL_BITS = 4096  # carry-over so a frame straddling a drain boundary isn't lost (~2 AX.25)

    def __init__(
        self,
        name: str,
        sink,
        framing: str | None = None,
        framing_parameters: dict | None = None,
        native_enabled: bool = False,
    ) -> None:
        self.name = name
        self._sink = sink
        # Backend framing hint, VERBATIM (SatYAML label or local token) — framings.deframe
        # normalizes it to a local deframer; unknown labels deframe upstream (gr-satellites).
        self._framing = (framing or "").strip() or None
        self._native_profile = resolve_profile(self._framing) if self._framing else None
        self._native_decoder = None
        if (
            native_enabled
            and self._native_profile is not None
            and self._native_profile.decoder_available
            and self._native_profile.symbol_input is SymbolInput.HARD_BITS
        ):
            supplied = framing_parameters or {}
            profile_parameters = {
                key: supplied[key] for key in self._native_profile.parameters if key in supplied
            }
            self._native_decoder = build_decoder(self._framing, profile_parameters)
        self._locked: str | None = None  # framing discovered this pass when no hint was given
        self._hits: dict[str, int] = {}  # per-framing match count, to lock only on a confident one
        self._tail = np.empty(0, dtype=np.uint8)  # bits carried across drain boundaries
        # The LOCAL framing that produced the last drain's NEW frames (None if none). The race
        # gate feeds this to compose.race_winner: only a CRC-gated framing may win (MED-1).
        self.race_framing: str | None = None

    def drain_frames(self) -> list[FrameResult]:
        # Backend gave the framing → use only that. Otherwise try all; once ONE framing has
        # matched _LOCK_AFTER times (a single CRC hit can be a ~1/65536 fluke), lock to it for
        # the rest of the pass so a spurious early match can't strand the real framing.
        use = self._framing or self._locked
        fresh = self._sink.drain()
        if self._native_decoder is not None:
            out = self._native_decoder.push(fresh)
            self.race_framing = (
                self._native_profile.canonical
                if any(frame.integrity is IntegrityStatus.PASSED for frame in out)
                else None
            )
            return out
        prev_tail = self._tail
        bits = np.concatenate([prev_tail, fresh]) if prev_tail.size else fresh
        self._tail = bits[-self._TAIL_BITS :].copy() if bits.size else self._tail
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
        # Legacy deframers have no source-coordinate contract. Wrap their payloads so the live
        # writer can preserve that distinction rather than inventing offsets or timestamps.
        return [
            FrameResult(
                canonical_framing=matched or str(use or "legacy"),
                payload=frame,
                integrity=(
                    IntegrityStatus.PASSED
                    if framings.is_crc_gated(matched or use)
                    else IntegrityStatus.NOT_PRESENT
                ),
                source_start=0,
                source_end=0,
                polarity=Polarity.AMBIGUOUS,
                metadata={"source_offsets_available": False},
            )
            for frame in out
        ]

    def flush_frames(self) -> list[FrameResult]:
        if self._native_decoder is None:
            return []
        return self._native_decoder.flush()


# Deframing (``framings.deframe``) and the modulation→demod dispatch (``modem.build_demod``)
# live in the framing/modem registries now (docs/08 — universal modem + framing).
# ``_build_fallbacks`` just composes them: (modulation, rate) → build demod → wrap with the
# deframer. New modulations/framings register in modem.py/framings.py, not here.


def _build_fallbacks(
    tb,
    demod_src,
    sample_rate: float,
    modes=None,
    framing=None,
    differential=None,
    channel_bw_hz=None,
    framing_parameters=None,
    native_enabled=False,
) -> tuple[list[_FallbackDemod], object]:
    """Build the demod(s) tapping ``demod_src`` (already at the channel rate) and return
    ``(fallbacks, soft_tap)``: the list of our numpy-deframer fallbacks, plus the FLOAT
    soft-symbol tap of the FSK demod (or ``None``) that Phase 3 feeds to the decoupled
    gr-satellites deframers. ``modes`` is a list of ``(modulation, symbol_rate)`` tuples —
    normally the ONE the backend specified from the transmitter record — deframed with the backend
    ``framing`` (verbatim label; the framing registry normalizes). ``differential`` (bool | None)
    threads the backend's DxPSK flag to the PSK demod. Modulation coverage comes from the modem
    registry (``modem.build_demod``)."""
    import modem  # noqa: PLC0415 — lazy: pulls in gnuradio_gfsk (GNU Radio) only at decode time

    out: list[_FallbackDemod] = []
    soft_tap = None  # FSK float soft-symbol tap (last one built) for the gr-satellites deframers
    for kind, rate in modes or []:
        kind = str(kind or "").strip().lower()
        if not kind:
            continue
        # Guarded: a demod that can't be built for this channel (e.g. symbol_sync needs sps>1,
        # so the rate exceeds ~sample_rate/2) must NOT crash the engine or cost us the IQ
        # recording — skip it and keep the others.
        try:
            sink, soft = modem.build_demod(
                kind,
                tb,
                demod_src,
                sample_rate,
                float(rate or 0.0),
                differential=differential,
                channel_bw_hz=channel_bw_hz,
            )
        except Exception as e:  # noqa: BLE001 — one bad demod must not sink the rest/recording
            _log.warning("fallback demod %s@%s failed to build (%s); skipping", kind, rate, e)
            continue
        if sink is None:
            _log.warning("fallback demod %s@%s not implemented; skipping", kind, rate)
            continue
        native_for_demod = native_enabled
        if native_enabled and framing:
            pairing = plan_native_rx_pairing(
                framing,
                kind,
                sample_rate_hz=sample_rate,
                symbol_rate_hz=float(rate or 0.0),
                capture_rate_hz=sample_rate,
                execution=RxExecution.LIVE,
                evaluation=True,
            )
            native_for_demod = pairing.accepted
            if not pairing.accepted:
                _log.warning(
                    "native live pairing rejected before decoder construction: %s/%s (%s)",
                    kind,
                    framing,
                    pairing.reason,
                )
        native_sink = sink
        native_profile = resolve_profile(framing) if framing else None
        if (
            native_for_demod
            and native_profile is not None
            and native_profile.symbol_input is SymbolInput.SOFT_SYMBOLS
        ):
            if soft is None:
                native_for_demod = False
                _log.warning(
                    "native live soft profile %s has no demodulator soft tap; rejecting",
                    framing,
                )
            else:
                from gnuradio_gfsk import SoftSymbolSink  # noqa: PLC0415 - GNU Radio path only

                native_sink = SoftSymbolSink()
                tb.connect(soft, native_sink)
        out.append(
            _FallbackDemod(
                f"{kind}{int(rate or 0)}",
                native_sink,
                framing,
                framing_parameters=framing_parameters,
                native_enabled=native_for_demod,
            )
        )
        if soft is not None:
            soft_tap = soft
    return out, soft_tap


def _backend_mode(params: dict | None) -> tuple[str, float] | None:
    """The single ``(modulation, symbol_rate)`` the backend specified from the transmitter
    record's modulation + symbol rate. None when either is absent — caller then runs
    gr-satellites only. A tuple (not a concatenated string) so digit-bearing modulation names
    (``2fsk``, ``8psk``, ``qam16``) stay unambiguous.

    The symbol rate is read via :func:`symbol_rate_hz_of`, so ``baud`` / ``baudrate`` are accepted
    interchangeably with ``symbol_rate_hz`` — the demod builds whenever the rate is present under
    ANY of its names (baud == symbol rate), never dark because of a key-name mismatch."""
    p = params or {}
    kind = str(p.get("modulation") or "").strip().lower()
    rate = symbol_rate_hz_of(p)  # accepts baud/baudrate/symbol_rate_hz (all the same quantity)
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
            samp_rate=channel_rate,
            iq=True,
            grc_block=True,
            **selector,  # gr-satellites resamples
        )
        _log.info("gr-satellites: decoder for %s (%r) @ %.0f Hz", satellite, selector, channel_rate)
        return fg
    except Exception as e:  # noqa: BLE001 — not catalogued / API drift
        _log.info("gr-satellites: no decoder for %s (%s)", satellite, e)
        return None


def make_grsat_deframers(framing) -> list:
    """Map a framing label → a LIST of gr-satellites DEFRAMER hier-blocks
    (``satellites.components.deframers.*``) that consume the FLOAT soft-symbol tap of OUR demod and
    emit frame PDUs on their ``out`` message port. This is the SatNOGS-robust decouple (docs/12
    Phase 3): we demodulate ONCE (connect_gfsk_demod) and reuse gr-satellites' proven deframers on
    the soft tap — NOT the monolithic ``gr_satellites_flowgraph`` that did its own demod+resample on
    raw IQ and buffer-deadlocked, starving the recorder.

    GUARDED at every step: if gr-satellites isn't importable, or a deframer ctor drifts / rejects
    its args, that entry is skipped (logged) and the rest still build — a decoder problem must never
    crash the engine or cost the IQ recording. Returns ``[]`` for a framing with no gr-satellites
    deframer (decoded by our numpy deframers, or record-only). AX.25 races both G3RUH scramblings
    (mirrors ``framings.deframe``). Do NOT NRZI/descramble upstream — each deframer does its own."""
    plan = framings.grsat_deframer_plan(framing)
    if not plan:
        return []
    try:
        from satellites.components.deframers import (  # noqa: PLC0415 — bench-only (gr-satellites)
            ax25_deframer,
            ax100_deframer,
            endurosat_deframer,
            usp_deframer,
        )
    except Exception as e:  # noqa: BLE001 — no gr-satellites here → numpy deframers only
        _log.info("gr-satellites deframers unavailable (%s); our numpy deframers only", e)
        return []

    builders = {
        "ax25": lambda scramble: ax25_deframer(scramble, options=None),
        "ax100": lambda mode: ax100_deframer(mode, options=None),
        "usp": lambda: usp_deframer(options=None),
        "endurosat": lambda: endurosat_deframer(options=None),
    }
    out = []
    for kind, *args in plan:
        try:
            out.append(builders[kind](*args))
        except Exception as e:  # noqa: BLE001 — API drift / bad ctor args → skip this deframer
            _log.warning("gr-satellites deframer %s%r failed (%s); skipping", kind, args, e)
    return [d for d in out if d is not None]


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
        path,
        norad,
        p.get("modulation"),
        symbol_rate_hz_of(p) or None,
        p.get("framing"),
        frequency_hz,
        name=(s or None),
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
    sym = symbol_rate_hz_of(params)  # baud/baudrate/symbol_rate_hz — interchangeable
    want_channel = max(float(sample_rate), CHANNEL_OVERSAMPLE * sym)
    sdr_rate, _ = capture_plan(env["capture_rate_hz"], want_channel)
    channel_rate = channel_rate_for(float(sample_rate), sym, sdr_rate)
    decimate = channel_rate < sdr_rate
    # AUTO LO offset: dodge the DC/LO spike off the bird (no per-pass config — we know the
    # frequency). tune_below puts the carrier at +lo_offset at baseband; the software rotator
    # (make_lo_rotator) shifts it to DC and the decimator's LPF rejects the spike left at
    # -lo_offset. Honors an explicit GS_SDR_LO_OFFSET, else a 100 kHz default (docs/12 Phase 1).
    lo = auto_lo_offset(
        sdr_rate, channel_rate, env["lo_offset_hz"], default_offset_hz=DEFAULT_LO_OFFSET_HZ
    )
    tb = gr.top_block("gr_satellites_rx")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    open_analog_bandwidth(src, sdr_rate)  # widen analog BW so the +lo_offset carrier survives
    tune_below(src, float(args.center_freq_hz), lo)  # LO to center-lo_offset (plain; no BB CORDIC)
    sdr_applied = configure_soapy_source(src, merge_sdr_params(params))  # antenna+gain (else deaf)
    sdr_applied.update(apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"]))
    # Front-end plan — the ONE line that says what the RX actually did this pass (so a
    # mis-deployed offset / mis-sized channel is never silent again). lo != 0 now means the
    # SOFTWARE rotator dodges the spike (works on the XTRX, unlike the old hardware BB offset).
    _log.info(
        "front-end: center=%.0f Hz lo_offset=%.0f Hz (%s, sw-rotator) | capture=%.0f Hz "
        "channel=%.0f Hz decimate=%s | dc_removal=%s",
        float(args.center_freq_hz),
        lo,
        "ON-CENTER" if not lo else "OFFSET",
        sdr_rate,
        channel_rate,
        decimate,
        env["dc_removal"],
    )

    # Software LO+Doppler rotator right after the source (at the capture rate): brings the
    # +lo_offset carrier to DC and is the mid-pass Doppler NCO (set_doppler → set_phase_inc).
    rotator = make_lo_rotator(sdr_rate, lo, 0.0)
    tb.connect(src, rotator)
    chan = rotator
    if decimate:
        chan = make_decimator(sdr_rate, channel_rate)
        tb.connect(rotator, chan)

    # Pre-demod IQ capture FIRST (the priority): it taps the channel independently of the
    # decoder, so a decoder problem never costs us the recording. At the CHANNEL rate.
    recorder = PassRecorder.maybe_start(args, tb, chan, sample_rate_hz=channel_rate)

    # Engine selection (per backend params). Everything taps the SAME channel stream (one SDR
    # open, fanned out — no hardware conflict). Phase 3 (docs/12) DECOUPLES gr-satellites from the
    # recording the SatNOGS-robust way: we demodulate ONCE (our connect_gfsk_demod) and feed
    #   * the hard-bit sink → our numpy deframers, and
    #   * the FLOAT soft-symbol tap → gr-satellites' own DEFRAMER components (make_grsat_deframers),
    # so BOTH decode libraries run off one demod. The frames are collected + deduped in
    # _SatContext.drain_frames (both are cheap → no valve gating needed; a CRC-less KISS hit is a
    # product but never suppresses the others). The monolithic gr_satellites_flowgraph — which did
    # its OWN demod+resample on raw IQ and buffer-DEADLOCKED, starving the recorder (cmd_70) —
    # survives ONLY for the no-demod-params catalogued case, and stays GATED (GS_GRSAT_LIVE). The
    # decoupled deframers are gated too for now, pending bench proof they can't backpressure the
    # recorder; default stays our-numpy-only + a bulletproof recording.
    from gnuradio import blocks  # noqa: PLC0415 — bench-only

    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    grsat_deframers: list = []
    selector = _gr_satellites_selector(satellite)
    framing = (params or {}).get("framing")
    differential = (params or {}).get("differential")
    if not isinstance(differential, bool):
        differential = None  # absent/garbage → PSK demod keeps its robust default
    mode = _backend_mode(params)  # (modulation, symbol_rate) when both present
    grsat_live = os.environ.get("GS_GRSAT_LIVE", "").strip().lower() in ("1", "true", "yes", "on")
    native_live = os.environ.get("GS_NATIVE_FRAMING_LIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not grsat_live:
        _log.info(
            "gr-satellites decode gated off (GS_GRSAT_LIVE unset → the recorder can never be "
            "starved); our numpy engine only. Set GS_GRSAT_LIVE=1 to also run gr-satellites."
        )
    if native_live:
        _log.warning(
            "native framing live path enabled for bench/shadow evaluation; exact raw-sample UTC "
            "mapping remains unavailable"
        )
    # The monolithic flowgraph (own demod on raw IQ, deadlock-prone) is ONLY for the no-mode
    # catalogued case — with demod params we use the decoupled deframers below instead.
    fg = None
    if grsat_live and not mode:
        fg = _build_grsatellites(selector, channel_rate, satellite)
    # Spectral inversion (rfLink ``invert``): conjugate the DECODE tap only — the recorder keeps
    # the raw channel so the .cf32 is always what was actually received.
    demod_tap = chan
    if (params or {}).get("invert") is True:
        demod_tap = blocks.conjugate_cc()
        tb.connect(chan, demod_tap)
        _log.info("spectral inversion: conjugating the decode tap (recorder stays raw)")
    if mode:
        # Our demod ONCE → (numpy-deframer fallbacks, FSK float soft tap).
        fallbacks, soft = _build_fallbacks(
            tb,
            demod_tap,
            channel_rate,
            modes=[mode],
            framing=framing,
            differential=differential,
            channel_bw_hz=float(args.bandwidth_hz or 0) or None,
            framing_parameters=params,
            native_enabled=native_live,
        )
        # Decoupled gr-satellites deframers on the soft tap (FLOAT valve, message out → the SAME
        # sink), racing our numpy deframers off one demod. Gated; empty for a framing they don't
        # cover (our numpy engine / record-only carries it).
        if grsat_live and soft is not None:
            grsat_deframers = make_grsat_deframers(framing)
            for d in grsat_deframers:
                tb.connect(soft, blocks.copy(gr.sizeof_float), d)
                tb.msg_connect(d, "out", sink, "in")
        # No decoupled deframer covers this framing → fall back to the (gated, deadlock-prone)
        # monolithic gr_satellites_flowgraph: catalogued SatYAML, else a SYNTHETIC one from the
        # backend (modulation, baud, framing) so the FULL ~50-deframer library still applies
        # (docs/08 Ph1). The common framings (AX.25/AX100/USP/EnduroSat) never reach here — they
        # are covered above, so the target birds never touch the deadlock path.
        if grsat_live and not grsat_deframers:
            fg = _build_grsatellites(selector, channel_rate, satellite)
            if fg is None:
                synth = _synthetic_satyaml_path(satellite, params, float(args.center_freq_hz))
                if synth is not None:
                    try:
                        fg = _build_grsatellites({"file": synth}, channel_rate, satellite)
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(synth)  # gr-satellites parsed it in __init__; safe to remove
            if fg is not None:
                tb.connect(demod_tap, fg)
                tb.msg_connect(fg, "out", sink, "in")
                _log.info(
                    "gr-satellites monolithic (gated) for %s framing=%s (no decoupled deframer)",
                    satellite,
                    framing or "?",
                )
        if not fallbacks and not grsat_deframers and fg is None:
            # Nothing consumes demod_tap (demod failed to build) → terminate so start() can't abort
            # the graph (and cost the recording). connect_gfsk_demod itself connects-last, so a
            # partial chain never dangles; this covers the "build returned None" case.
            tb.connect(demod_tap, blocks.null_sink(gr.sizeof_gr_complex))
        _log.info(
            "our demod %s@%.0f on %.0f Hz channel (framing=%s)%s",
            mode[0],
            mode[1],
            channel_rate,
            framing or "auto",
            f" + {len(grsat_deframers)} gr-satellites deframer(s)" if grsat_deframers else "",
        )
    elif fg is not None:  # no demod params → catalogued monolithic gr-satellites only (gated)
        tb.connect(demod_tap, fg)
        tb.msg_connect(fg, "out", sink, "in")
        _log.info("gr-satellites monolithic only for %s (no demod params)", satellite)
    else:
        _log.error(
            "NO DECODER BUILT for %r — this pass is RECORDER-ONLY and will produce ZERO "
            "frames. Cause: the backend sent no usable demod params (a transmitter with a "
            "null/zero baud yields no modulation+symbol_rate) and GS_GRSAT_LIVE is unset, so "
            "gr-satellites could not supply one either. The .cf32 is still captured — decode "
            "it offline (iq_decode.py) — but the pass must not be read as a successful decode.",
            satellite,
        )
    # Compose the registries into a decode plan (docs/08 Phase 4) for observability — which path(s)
    # the backend rfLink implies. Construction above drives the graph; the plan is the explanation.
    try:
        catalogued = fg is not None or bool(grsat_deframers)
        _log.info("decode plan: %s", compose.plan_decode(params, catalogued=catalogued).describe())
    except Exception as e:  # noqa: BLE001 — planning must never block decoding
        _log.debug("decode-plan compose failed (non-fatal): %s", e)
    # GNU Radio validates ALL stream ports at start(); a consumer-less tap would abort the whole
    # graph and cost us the recording. Terminate any tap that ended up without a consumer:
    #   * demod_tap: no decoder built (no-decode branch) — and when demod_tap is the conjugate block
    #     it ALWAYS needs a consumer (it's fed from chan);
    #   * chan: nothing at all downstream (decoders on a dead branch AND recording disabled).
    decode_consumers = bool(fallbacks) or fg is not None or bool(grsat_deframers)
    if not decode_consumers and demod_tap is not chan:
        tb.connect(demod_tap, blocks.null_sink(gr.sizeof_gr_complex))
    elif not decode_consumers and recorder is None:
        tb.connect(chan, blocks.null_sink(gr.sizeof_gr_complex))
    # R2-02: a graph with NO decode consumer is recorder-only. Say so out loud, on the
    # `ready` event, so the pass result cannot read as a successful decode (a green pass
    # with zero frames and no explanation is indistinguishable from a bird that was silent).
    # The reason itself is computed by a PURE helper so it is testable without GNU Radio.
    # A demod with NOTHING that can deframe it is decode-dead too: a backend framing
    # outside our local vocabulary (AX.100 / USP / Mobitex / CCSDS Concatenated…) is
    # deframable only by gr-satellites, so with it gated off every drain returns
    # nothing while the graph looks perfectly healthy.
    native_profile = resolve_profile(framing) if framing else None
    native_pairing_available = False
    if native_live and native_profile is not None and mode is not None:
        native_pairing_available = plan_native_rx_pairing(
            framing,
            mode[0],
            sample_rate_hz=channel_rate,
            symbol_rate_hz=mode[1],
            capture_rate_hz=channel_rate,
            execution=RxExecution.LIVE,
            evaluation=True,
        ).accepted
    deframer_available = (
        bool(grsat_deframers)
        or fg is not None
        or (
            native_pairing_available
            or (framings.normalize_framing(framing) is not None if framing else True)
        )
    )
    reason = no_decode_reason(
        has_decode_consumer=decode_consumers,
        mode=mode,
        grsat_live=grsat_live,
        framing=framing,
        deframer_available=deframer_available,
    )
    # Decoupled model: both decode libraries run off our one demod, so there are no valves to gate
    # (valve_ours/valve_grsat stay None → drain_frames just collects + dedups; the race_winner
    # gating it still carries is dormant unless a future path re-introduces valves).
    return _SatContext(
        tb,
        src,
        sink,
        float(args.center_freq_hz),
        recorder,
        lo_offset_hz=lo,
        rotator=rotator,
        sdr_rate_hz=sdr_rate,
        fallbacks=fallbacks,
        sdr_applied=sdr_applied,
        no_decode_reason=reason,
        shadow_enabled=bool(fallbacks) and (bool(grsat_deframers) or fg is not None),
    )


__all__ = ["build_satellites_rx"]

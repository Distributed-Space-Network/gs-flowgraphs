"""Decode composer — backend rfLink ``(modulation, fec, framing, baud)`` → a decode plan (docs/08
Phase 4).

The registries (:mod:`modem`, :mod:`fec`, :mod:`framings`) describe *what* can be demodulated,
error-corrected, and deframed. This module composes them into the *decision* the engine acts on:
given the backend's transmitter description (and whether gr-satellites catalogs the bird), which
path(s) can decode it —

  * **our engine**  — our modem demodulates the modulation AND a local (numpy) deframer handles the
    framing;
  * **gr-satellites** — the bird is catalogued, OR a synthetic SatYAML can be built (FSK/BPSK/AFSK
    + a gr-satellites framing + baud) to reuse its ~50 deframers;
  * **race** — both apply, so the engine runs them in parallel and the first CRC-valid frame wins.

Pure (numpy/stdlib), so the whole decision layer is unit-testable without GNU Radio. The bench
engine (``gnuradio_satellites.build_satellites_rx``) consults this to select + log its path.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import framings
import grsat_synth
import modem
from _fallback_select import symbol_rate_hz_of


@dataclass(frozen=True)
class DecodePlan:
    modulation: str | None
    symbol_rate: float
    framing: str | None
    fec: str | None
    tier: int | None            # modem tier (1/2/3) or None if the modulation is unrecognized
    our_modem: bool             # our modem registry recognizes the modulation
    our_framing: bool           # a local numpy deframer handles the framing (no label →
                                # the CRC-gated autodetect set, matching the engine)
    our_crc_gated: bool         # that deframer has a real integrity check (may win the race)
    grsat_catalogued: bool      # gr-satellites has a SatYAML for this bird
    grsat_synthesizable: bool   # a synthetic SatYAML can be built (FSK/BPSK/AFSK + framing + baud)

    @property
    def our_engine(self) -> bool:
        """Our demod + a local deframer can decode it in-process."""
        return self.our_modem and self.our_framing

    @property
    def grsatellites(self) -> bool:
        """gr-satellites can decode it (catalogued or via a synthetic SatYAML)."""
        return self.grsat_catalogued or self.grsat_synthesizable

    @property
    def race(self) -> bool:
        """Both paths apply → run in parallel, first valid frame wins."""
        return self.our_engine and self.grsatellites

    @property
    def race_ours_can_win(self) -> bool:
        """In a race, our engine may declare the win ONLY when its framing carries a real
        integrity check (``framings.is_crc_gated`` — docs/10 MED-1). Otherwise (KISS) our
        frames are products but never gate off gr-satellites; :func:`race_winner` agrees."""
        return self.race and self.our_crc_gated

    @property
    def decodable(self) -> bool:
        return self.our_engine or self.grsatellites

    def describe(self) -> str:
        if self.race:
            paths = ("race(ours+gr-satellites)" if self.our_crc_gated
                     else "race(ours+gr-satellites; ours ungated — no CRC)")
        elif self.our_engine:
            paths = "our-engine"
        elif self.grsatellites:
            paths = "gr-satellites" + ("(catalogued)" if self.grsat_catalogued else "(synthetic)")
        else:
            paths = "none(record-only)"
        return (f"{self.modulation or '?'}@{self.symbol_rate:.0f} framing={self.framing or '?'}"
                f" fec={self.fec or '-'} → {paths}")


def plan_decode(params: dict | None, *, catalogued: bool = False) -> DecodePlan:
    """Build a :class:`DecodePlan` from the backend transmitter ``params`` (``modulation``,
    ``symbol_rate_hz``, ``framing``, optional ``fec``). ``catalogued`` is whether gr-satellites has
    a SatYAML for the bird (the caller knows this from the NORAD lookup)."""
    p = params or {}
    modulation = str(p.get("modulation") or "").strip().lower() or None
    framing = str(p.get("framing")).strip() if p.get("framing") not in (None, "") else None
    fec = str(p.get("fec")).strip() if p.get("fec") not in (None, "") else None
    baud = symbol_rate_hz_of(p)  # baud/baudrate/symbol_rate_hz — interchangeable
    spec = modem.modulation_spec(modulation) if modulation else None
    # A framing is "ours" when the registry normalizes it to a LOCAL deframer — this accepts
    # backend/SatYAML labels verbatim ("AX.25 G3RUH" → ax25), not just local tokens.
    if framing is not None:
        our_framing = framings.normalize_framing(framing) is not None
        our_crc_gated = framings.is_crc_gated(framing)
    else:
        # No framing label: the engine still builds modulation fallbacks and
        # framings.deframe AUTODETECTS across the registry's autodetect set, so the plan
        # must report the same — the race exists, and ours can win exactly when every
        # autodetected framing is CRC-gated (it is, by the registry's construction).
        # Derived from the same registry tuple the engine consumes (docs/J LOW-2).
        auto = framings.autodetect_framings()
        our_framing = bool(auto)
        our_crc_gated = bool(auto) and all(framings.is_crc_gated(f) for f in auto)
    return DecodePlan(
        modulation=modulation,
        symbol_rate=baud,
        framing=framing,
        fec=fec,
        tier=spec.tier if spec else None,
        our_modem=spec is not None,
        our_framing=our_framing,
        our_crc_gated=our_crc_gated,  # single source: the framings registry (see above)
        grsat_catalogued=bool(catalogued),
        grsat_synthesizable=grsat_synth.can_synthesize(modulation, baud, framing),
    )


def race_winner(our_matched_framings: Iterable[str | None], grsat_produced: bool) -> str | None:
    """The engine-race decision for ONE drain window — pure, so the GNU-Radio engine's valve
    logic is unit-testable without GNU Radio (``gnuradio_satellites._SatContext`` calls this).

    ``our_matched_framings``: the framing labels (any vocabulary) that produced OUR engine's new
    frames this window; ``grsat_produced``: gr-satellites emitted at least one frame. Returns
    ``"ours"`` / ``"grsatellites"`` / ``None`` (no winner yet — keep racing).

    Only a framing with a REAL integrity check may declare our win (``framings.is_crc_gated`` —
    docs/10 MED-1): KISS has no checksum and decodes ~2 garbage frames per noise drain, so a
    KISS "hit" must never gate off the real gr-satellites decoder for the pass (its frames are
    still emitted as products). gr-satellites deframers all validate CRC/FEC, so any PDU from it
    counts. On a tie within one window OUR engine wins (it's the backend-specified primary),
    but only when CRC-gated."""
    if any(framings.is_crc_gated(f) for f in our_matched_framings):
        return "ours"
    if grsat_produced:
        return "grsatellites"
    return None

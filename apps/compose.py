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

from dataclasses import dataclass

import framings
import grsat_synth
import modem


@dataclass(frozen=True)
class DecodePlan:
    modulation: str | None
    symbol_rate: float
    framing: str | None
    fec: str | None
    tier: int | None            # modem tier (1/2/3) or None if the modulation is unrecognized
    our_modem: bool             # our modem registry recognizes the modulation
    our_framing: bool           # a local numpy deframer handles the framing
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
    def decodable(self) -> bool:
        return self.our_engine or self.grsatellites

    def describe(self) -> str:
        if self.race:
            paths = "race(ours+gr-satellites)"
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
    try:
        baud = float(p.get("symbol_rate_hz") or 0.0)
    except (TypeError, ValueError):
        baud = 0.0
    spec = modem.modulation_spec(modulation) if modulation else None
    # A framing is "ours" when the registry normalizes it to a LOCAL deframer — this accepts
    # backend/SatYAML labels verbatim ("AX.25 G3RUH" → ax25), not just local tokens.
    our_framing = framings.normalize_framing(framing) is not None
    return DecodePlan(
        modulation=modulation,
        symbol_rate=baud,
        framing=framing,
        fec=fec,
        tier=spec.tier if spec else None,
        our_modem=spec is not None,
        our_framing=our_framing,
        grsat_catalogued=bool(catalogued),
        grsat_synthesizable=grsat_synth.can_synthesize(modulation, baud, framing),
    )

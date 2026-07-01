"""Framing registry — framing name → deframe (and, later, frame).

Phase 0 of docs/08 (universal modem + framing). The deframe dispatch that used to live in
``gnuradio_satellites._bits_deframe`` lives here now, so new framings (gr-satellites' ~50
deframers, CCSDS TM/TC/AOS/USLP, Argos, per docs/08) plug in as isolated entries. Deframers
encapsulate their own FEC (like gr-satellites): AX.25 does G3RUH descramble + NRZI internally.
This module is numpy-only (no GNU Radio) so it stays fully unit-testable.
"""
from __future__ import annotations

import numpy as np

# Link layers our own engine deframes. gr-satellites does its own for catalogued birds.
_FRAMINGS = ("ax25", "endurosat")


def known_framings() -> tuple[str, ...]:
    return _FRAMINGS


def deframe(bits, framing_name: str | None = None) -> tuple[list[bytes], str | None]:
    """Hard bits → ``(frames, matched_framing)``. ``framing_name`` runs ONLY that link layer
    (the backend told us); ``None`` tries every known framing and reports which matched (so the
    caller can lock onto it). Every framing is CRC/FCS-gated, so trying several is safe (a wrong
    one has a ~1/65536-per-flag spurious chance — hence the caller's lock-once-matched)."""
    from gfsk_ax25 import endurosat_link  # noqa: PLC0415
    from gfsk_ax25 import framing as ax25_framing

    arr = np.asarray(bits, dtype=np.uint8)
    if not len(arr):
        return [], None
    order = [framing_name.strip().lower()] if framing_name else list(_FRAMINGS)
    for name in order:
        if name == "endurosat":
            frames = endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)
        elif name == "ax25":  # G3RUH-descrambled and plain — same framing, different descrambling
            frames = []
            for scramble in (True, False):
                frames.extend(ax25_framing.decode(arr, scramble=scramble, nrzi=True))
        else:
            continue
        if frames:
            return frames, name
    return [], None

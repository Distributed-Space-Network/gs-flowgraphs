"""Engine-independent link layer: AX.25 body bytes <-> serial bit stream.

This is the piece shared by both flowgraph engines (numpy ``dsp`` and bench
``gnuradio``): once the front-end has recovered hard bits, deframing is pure
protocol and identical for both, so it is implemented and tested once here.

Transmit:  body -> HDLC frame -> NRZI -> G3RUH scramble -> bits
Receive:   bits -> G3RUH descramble -> NRZI decode -> HDLC deframe -> bodies

``scramble`` and ``nrzi`` are toggles because the spec we were given names
GFSK + AX.25 but not the bit-coding details; G3RUH + NRZI is the de-facto 9k6
default and our working assumption. Flip them in one place if telemetry shows
otherwise.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import numpy as np

from . import g3ruh, hdlc


def encode(
    body: bytes,
    *,
    preamble_flags: int = 16,
    postamble_flags: int = 2,
    scramble: bool = True,
    nrzi: bool = True,
) -> np.ndarray:
    """Body bytes -> transmit bit stream (pre-modulation)."""
    bits = hdlc.frame(body, preamble_flags=preamble_flags, postamble_flags=postamble_flags)
    if nrzi:
        bits = g3ruh.nrzi_encode(bits)
    if scramble:
        bits = g3ruh.scramble(bits)
    return bits


def decode(
    bits: np.ndarray,
    *,
    scramble: bool = True,
    nrzi: bool = True,
) -> list[bytes]:
    """Received bit stream -> list of valid AX.25 frame bodies (FCS-checked)."""
    out = np.asarray(bits, dtype=np.uint8)
    if scramble:
        out = g3ruh.descramble(out)
    if nrzi:
        out = g3ruh.nrzi_decode(out)
    return hdlc.deframe(out)


__all__ = ["decode", "encode"]

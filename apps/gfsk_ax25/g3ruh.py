"""NRZI line coding + G3RUH self-synchronizing scrambler.

The classic 9k6 cubesat physical layer (and the EnduroSat UHF configuration we
infer from the spec) applies, on transmit: HDLC bits -> NRZI -> G3RUH scramble
-> GFSK. Receive reverses it: GFSK demod -> descramble -> NRZI decode -> HDLC.

* **NRZI** (as used by AX.25): a ``0`` bit toggles the output level, a ``1``
  leaves it unchanged. This guarantees transitions for clock recovery as long
  as the data is not all-ones (which HDLC bit-stuffing already prevents).
* **G3RUH** is a multiplicative scrambler with polynomial ``1 + x^12 + x^17``:
  ``y[n] = x[n] XOR y[n-12] XOR y[n-17]``. It is self-synchronizing — the
  descrambler locks within 17 bits with no shared state — which is why no
  preamble handshake is needed.

Bits are 1-D ``numpy.uint8`` arrays of 0/1.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import numpy as np

_TAP_A = 12
_TAP_B = 17
_MASK = (1 << _TAP_B) - 1


def nrzi_encode(bits: np.ndarray, *, initial: int = 1) -> np.ndarray:
    out = np.empty(len(bits), dtype=np.uint8)
    level = initial & 1
    for i, bit in enumerate(bits.tolist()):
        if bit == 0:
            level ^= 1  # 0 -> transition
        out[i] = level
    return out


def nrzi_decode(bits: np.ndarray, *, initial: int = 1) -> np.ndarray:
    src = np.asarray(bits, dtype=np.uint8)
    out = np.empty(len(src), dtype=np.uint8)
    if not len(src):
        return out
    out[0] = int(src[0] == (initial & 1))
    out[1:] = (src[1:] == src[:-1]).astype(np.uint8)  # no change -> 1, transition -> 0
    return out


def scramble(bits: np.ndarray, *, state: int = 0) -> np.ndarray:
    """G3RUH multiplicative scramble. ``state`` seeds the 17-bit register."""
    out = np.empty(len(bits), dtype=np.uint8)
    sr = state & _MASK
    for i, bit in enumerate(bits.tolist()):
        fb = ((sr >> (_TAP_A - 1)) ^ (sr >> (_TAP_B - 1))) & 1
        y = bit ^ fb
        out[i] = y
        sr = ((sr << 1) | y) & _MASK
    return out


def descramble(bits: np.ndarray, *, state: int = 0) -> np.ndarray:
    """Inverse of :func:`scramble`; self-synchronizes within 17 bits."""
    src = np.asarray(bits, dtype=np.uint8)
    if state:
        # A non-zero seed is uncommon and only affects the first 17 bits; keep
        # the direct recurrence for that compatibility path.
        out = np.empty(len(src), dtype=np.uint8)
        sr = state & _MASK
        for i, y in enumerate(src.tolist()):
            fb = ((sr >> (_TAP_A - 1)) ^ (sr >> (_TAP_B - 1))) & 1
            out[i] = y ^ fb
            sr = ((sr << 1) | y) & _MASK
        return out
    # With the normal zero seed the descrambler is feed-forward: x[n] is the
    # received y[n] XOR the received bits 12 and 17 places back. NumPy performs
    # the same recurrence over a whole pass without a Python loop.
    out = src.copy()
    out[_TAP_A:] ^= src[:-_TAP_A]
    out[_TAP_B:] ^= src[:-_TAP_B]
    return out


__all__ = ["descramble", "nrzi_decode", "nrzi_encode", "scramble"]

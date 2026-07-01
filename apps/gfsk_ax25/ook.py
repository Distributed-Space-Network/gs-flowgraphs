"""OOK / ASK modulation — envelope keying (docs/08 Tier 3).

On-Off Keying and M-ary Amplitude-Shift Keying at complex baseband: the symbol amplitude is the
envelope, so modulation is "repeat each symbol amplitude for sps samples" and demodulation is
"integrate |IQ| per symbol → slice against an adaptive threshold". numpy-only (operates on a
captured complex IQ array), so it is fully unit-testable and can run post-pass on a .cf32.

For OOK (2-level) the slicer uses the midpoint of the observed min/max envelope (robust to gain);
M-ASK uses evenly spaced decision levels. Symbol timing is taken as a known ``sps`` with optional
integer sample offset (a real capture recovers it from a preamble / energy peak — bench).
"""
from __future__ import annotations

import numpy as np


def modulate(bits, sps: int, *, amp: float = 1.0, levels: int = 2) -> np.ndarray:
    """Symbols → complex baseband envelope. For OOK (``levels=2``) ``bits`` are 0/1; for M-ASK
    they are 0..levels-1, mapped to evenly spaced amplitudes in ``[0, amp]``. Each symbol is held
    for ``sps`` samples."""
    sym = np.asarray(bits, dtype=np.float64)
    env = sym * (amp / (levels - 1)) if levels > 2 else sym * amp
    return np.repeat(env, sps).astype(np.complex128)


def demodulate(iq, sps: int, *, offset: int = 0, levels: int = 2) -> np.ndarray:
    """Complex IQ → symbols. Integrates the magnitude over each ``sps``-sample symbol and slices
    against an adaptive threshold (OOK: min/max midpoint; M-ASK: nearest of ``levels`` levels)."""
    env = np.abs(np.asarray(iq, dtype=np.complex128))[offset:]
    n = len(env) // sps
    if n == 0:
        return np.empty(0, dtype=np.uint8)
    sym = env[:n * sps].reshape(n, sps).mean(axis=1)
    lo, hi = float(sym.min()), float(sym.max())
    if hi - lo < 1e-12:  # flat — no modulation present
        return np.zeros(n, dtype=np.uint8)
    if levels == 2:
        return (sym > (lo + hi) / 2.0).astype(np.uint8)
    step = (hi - lo) / (levels - 1)
    return np.clip(np.round((sym - lo) / step), 0, levels - 1).astype(np.uint8)

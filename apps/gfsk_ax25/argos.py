"""Argos PTT / PMT-A3 deframing (docs/08 Tier 1 — a framing NOT in gr-satellites).

Argos (ARGOS-2/3/4) platform-terminal transmissions in the 401 MHz band can be received
directly by the ground station. Their defining, mathematically-verifiable component is the
**BCH(31,21) code** that protects the platform-ID field — implemented here rigorously
(systematic encode + syndrome ≤2-error correction) and unit-tested by error injection.

The surrounding wire format has documented variants (Argos-2 PTT-A2 split-phase vs Argos-3/4
PMT-A3 high-data-rate). :func:`deframe` is therefore **parametric** in the frame-sync word and
field widths: its structure (sync search → BCH-protected ID recovery → payload) is what's tested;
the default constants are the documented Argos-2 PTT-A2 values and are flagged
BENCH/SPEC-CONFIRM — validate the sync word + field layout against a real 401 MHz PTT capture
before trusting the decoded IDs of a specific bird. numpy/stdlib-only (fully unit-testable).
"""
from __future__ import annotations

import numpy as np

# ── BCH(31,21,t=2) over GF(2^5), primitive poly x^5+x^2+1 ────────────────────────────────────
# g(x) = m_1(x)·m_3(x) = (x^5+x^2+1)(x^5+x^4+x^3+x^2+1), degree 10  ⇒  (31,21).
_BCH_G = 0b11101101001  # generator polynomial, deg 10 (verified: corrects all ≤2-bit errors)
_BCH_N = 31
_BCH_K = 21
_BCH_PARITY = _BCH_N - _BCH_K  # 10


def _polymod(a: int, g: int) -> int:
    dg = g.bit_length() - 1
    while a.bit_length() - 1 >= dg and a:
        a ^= g << ((a.bit_length() - 1) - dg)
    return a


def bch3121_encode(msg21: int) -> int:
    """Systematically encode a 21-bit message (MSB = x^20) into a 31-bit BCH codeword: the
    message occupies the high 21 bits, the 10 parity bits the low positions."""
    if not 0 <= msg21 < (1 << _BCH_K):
        raise ValueError("BCH(31,21) message must be 21 bits")
    shifted = msg21 << _BCH_PARITY
    return shifted ^ _polymod(shifted, _BCH_G)


def bch3121_decode(code31: int) -> int | None:
    """Correct up to 2 bit errors in a 31-bit codeword and return the 21-bit message, or ``None``
    if it is more than 2 errors from any codeword. Brute-forces weight-≤2 error patterns against
    the syndrome — correct and cheap for a t=2 code (≤ 31 + C(31,2) = 496 trials)."""
    if not 0 <= code31 < (1 << _BCH_N):
        raise ValueError("BCH(31,21) codeword must be 31 bits")
    if _polymod(code31, _BCH_G) == 0:
        return code31 >> _BCH_PARITY
    for i in range(_BCH_N):
        if _polymod(code31 ^ (1 << i), _BCH_G) == 0:
            return (code31 ^ (1 << i)) >> _BCH_PARITY
    for i in range(_BCH_N):
        for j in range(i + 1, _BCH_N):
            if _polymod(code31 ^ (1 << i) ^ (1 << j), _BCH_G) == 0:
                return (code31 ^ (1 << i) ^ (1 << j)) >> _BCH_PARITY
    return None


# ── PTT deframer (parametric; default = documented Argos-2 PTT-A2) ────────────────────────────
# BENCH/SPEC-CONFIRM: the frame-sync word and field widths below are the documented Argos-2
# PTT-A2 values; confirm against a real 401 MHz PTT capture before trusting a bird's decode.
ARGOS_PTT_A2_SYNC = 0xAC          # documented bit-sync/frame-sync (placeholder length; override)
ARGOS_PTT_A2_SYNC_BITS = 8


def _bits_of(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _int_of(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def deframe(
    bits,
    *,
    sync: int = ARGOS_PTT_A2_SYNC,
    sync_bits: int = ARGOS_PTT_A2_SYNC_BITS,
    payload_bits: int = 0,
) -> list[bytes]:
    """Locate PTT frames in a hard-bit stream and return each decoded message as bytes.

    For every occurrence of the ``sync`` word, the following 31 bits are read as a BCH(31,21)
    codeword protecting the 21-bit platform-ID field; if it is within 2 bit errors of a valid
    codeword the ID is corrected and emitted (as 3 big-endian bytes), optionally followed by
    ``payload_bits`` of trailing data.

    NOTE — the **frame-sync word length** is the primary false-alarm gate here: a rate-21/31
    BCH with t=2 correction accepts ~48% of random 31-bit inputs (2^21·497 ≈ 2^30), so BCH is
    NOT a standalone FCS. Real Argos framing pairs a long sync with a message-length field and
    per-block coding; pass the documented full-length ``sync`` (not the 8-bit placeholder) so
    accidental syncs are negligible."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    sync_pat = _bits_of(sync, sync_bits)
    need = sync_bits + _BCH_N + payload_bits
    if arr.size < need:
        return []
    out: list[bytes] = []
    win = np.lib.stride_tricks.sliding_window_view(arr, sync_bits)
    starts = np.nonzero((win == sync_pat).all(axis=1))[0]
    for s in starts:
        p = int(s) + sync_bits
        if p + _BCH_N + payload_bits > arr.size:
            continue
        code = _int_of(arr[p:p + _BCH_N])
        msg = bch3121_decode(code)
        if msg is None:
            continue  # >2 errors → not a real frame (BCH acts as the FCS)
        rec = bytearray(int(msg).to_bytes(3, "big"))  # 21-bit ID, left-justified in 3 bytes
        if payload_bits:
            pl = arr[p + _BCH_N:p + _BCH_N + payload_bits]
            rec += np.packbits(pl).tobytes()
        out.append(bytes(rec))
    return out

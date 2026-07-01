"""FEC registry — forward-error-correction codecs the framing layer composes (docs/08 Tier 1).

Each gr-satellites deframer already encapsulates its own coding (AX.25 G3RUH scramble, EnduroSat
CRC, AX.100 Golay/RS), so there is no mandatory ``demod → fec → deframe`` middle stage for those.
This registry provides the reusable coding primitives the NEW framings (CCSDS TM/TC/AOS, Argos)
need, split by where they run:

  * **numpy primitives (here, unit-tested):** the CCSDS pseudo-randomizer (CCSDS 131.0), CRC-16-
    CCITT / CRC-32, and the CCSDS Attached Sync Marker (ASM) correlator. Pure + verifiable against
    published reference vectors, so they are testable with no GNU Radio.
  * **GNU Radio / gr-satellites codecs (bench):** convolutional r=1/2 k=7 Viterbi
    (``fec.decode_ccsds_27``), Reed-Solomon (255,223) and Golay (gr-satellites), LDPC / Turbo /
    Polar (``fec.*``). Declared in the catalog so the composer knows they exist; construction is
    confirmed on the bench (no GNU Radio in CI).
"""
from __future__ import annotations

import numpy as np

from gfsk_ax25.crc import crc16_ccitt_false, crc32_ieee

# ── CCSDS pseudo-randomizer (CCSDS 131.0-B) ──────────────────────────────────────────────────
# h(x) = x^8 + x^7 + x^5 + x^3 + 1, all-ones seed. Empirically locked to the published PN
# sequence (FF 48 0E C0 9A 0D 70 BC …): 8-bit LFSR, feedback taps at bits {7,4,2,0}, MSB out,
# left shift. XOR-based ⇒ randomize and derandomize are the SAME operation (involutive).
_PN_TAPS = (7, 4, 2, 0)
_pn_cache: bytes = b""


def _pn_sequence(nbytes: int) -> bytes:
    """The first ``nbytes`` of the CCSDS randomizer PN sequence (cached/grown)."""
    global _pn_cache
    if len(_pn_cache) >= nbytes:
        return _pn_cache[:nbytes]
    state = 0xFF
    out = bytearray()
    for _ in range(nbytes):
        byte = 0
        for _ in range(8):
            bit = (state >> 7) & 1
            byte = (byte << 1) | bit
            fb = 0
            for t in _PN_TAPS:
                fb ^= (state >> t) & 1
            state = ((state << 1) | fb) & 0xFF
        out.append(byte)
    _pn_cache = bytes(out)
    return _pn_cache


def ccsds_randomize(data: bytes) -> bytes:
    """XOR ``data`` with the CCSDS PN sequence (CCSDS 131.0 pseudo-randomization). This is its
    own inverse — the same call de-randomizes a received frame. The randomizer restarts at the
    start of each transfer frame (after the ASM), so pass one frame's bytes at a time."""
    pn = _pn_sequence(len(data))
    return bytes(b ^ p for b, p in zip(data, pn, strict=True))


# De-randomization is the identical XOR; the alias documents intent at call sites.
ccsds_derandomize = ccsds_randomize


# ── CRCs (re-exported so the FEC layer is the single source) ─────────────────────────────────
def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Check value 0x29B1 for ``b"123456789"``."""
    return crc16_ccitt_false(data)


def crc32(data: bytes) -> int:
    """CRC-32/IEEE (Ethernet/zlib). Check value 0xCBF43926 for ``b"123456789"``."""
    return crc32_ieee(data)


# ── CCSDS Attached Sync Marker (ASM) ─────────────────────────────────────────────────────────
ASM_CCSDS = 0x1ACFFC1D          # standard 32-bit CCSDS TM/AOS frame sync marker
ASM_CCSDS_BYTES = ASM_CCSDS.to_bytes(4, "big")


def _u32_bits(value: int, width: int = 32) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def find_asm(bits, marker: int = ASM_CCSDS, width: int = 32) -> int:
    """Return the bit index just AFTER the first occurrence of ``marker`` (default the CCSDS ASM)
    in the hard-bit stream, i.e. where the randomized transfer frame begins — or ``-1`` if the
    marker isn't present. A plain sliding-window correlation (exact match); the demod is assumed
    bit-synchronous by this stage. Used by the CCSDS deframers (Tier 1/2)."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    m = _u32_bits(marker, width)
    if arr.size < width:
        return -1
    win = np.lib.stride_tricks.sliding_window_view(arr, width)
    hits = np.nonzero((win == m).all(axis=1))[0]
    return int(hits[0] + width) if hits.size else -1


# ── Catalog ──────────────────────────────────────────────────────────────────────────────────
# numpy-implemented here (verifiable in CI).
_NUMPY_CODES = ("ccsds_randomizer", "crc16", "crc32", "asm")
# GNU Radio fec.* — construction confirmed on the bench.
_GNURADIO_CODES = ("ccsds_conv_k7", "viterbi", "ldpc", "turbo", "polar")
# via gr-satellites deframers (reused whole, not called standalone).
_GRSAT_CODES = ("reed_solomon", "golay")


def known_codes() -> tuple[str, ...]:
    """Every FEC code the registry accounts for — numpy-implemented + GNU-Radio + gr-satellites."""
    return tuple(sorted({*_NUMPY_CODES, *_GNURADIO_CODES, *_GRSAT_CODES}))


def implemented_codes() -> tuple[str, ...]:
    """The subset implemented in-process (numpy) and exercised in CI."""
    return _NUMPY_CODES

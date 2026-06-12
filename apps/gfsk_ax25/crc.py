"""Generic CRC helpers used by the link layers (chip-packet + AX.25).

Kept separate from any protocol module so the public framing code does not
depend on protocol-specific modules. CRC algorithms are standard / public
domain; anchored to their published check values in the tests.

License: GPLv3 (see ../../COPYING).
"""

from __future__ import annotations

import zlib


def crc16_ccitt_false(data: bytes, *, init: int = 0xFFFF, poly: int = 0x1021) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, xorout 0).

    The EnduroSat chip-packet CRC. Check value: ``crc16_ccitt_false(b"123456789")
    == 0x29B1``.
    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def crc32_ieee(data: bytes) -> int:
    """CRC-32/ISO-HDLC (IEEE 802.3, zlib). Check: ``crc32_ieee(b"123456789") ==
    0xCBF43926``."""
    return zlib.crc32(data) & 0xFFFFFFFF


__all__ = ["crc16_ccitt_false", "crc32_ieee"]

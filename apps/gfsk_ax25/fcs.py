"""AX.25 / HDLC frame check sequence (CRC-16 CCITT, X.25 variant).

The FCS is the reflected CRC-16 with polynomial 0x1021 (reflected 0x8408),
initial value 0xFFFF, and a final one's-complement (xorout 0xFFFF). It is
transmitted least-significant-byte first. This is the same CRC ITU-T calls
CRC-16/X-25; the AX.25 v2.2 spec section 4.4 defines its use.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

_POLY_REFLECTED = 0x8408
_INIT = 0xFFFF
_XOROUT = 0xFFFF

# Precomputed byte table for the reflected CRC.
_TABLE: list[int] = []
for _b in range(256):
    _crc = _b
    for _ in range(8):
        _crc = (_crc >> 1) ^ _POLY_REFLECTED if _crc & 1 else _crc >> 1
    _TABLE.append(_crc & 0xFFFF)


def fcs(data: bytes) -> int:
    """Return the 16-bit FCS of ``data`` (already one's-complemented).

    The returned integer is the value to append little-endian after the frame
    body (address + control + PID + info). On receive, recompute over the body
    and compare to the received two FCS octets.
    """
    crc = _INIT
    for byte in data:
        crc = (crc >> 8) ^ _TABLE[(crc ^ byte) & 0xFF]
    return crc ^ _XOROUT


def fcs_bytes(data: bytes) -> bytes:
    """The FCS as the two octets to append, least-significant byte first."""
    value = fcs(data)
    return bytes((value & 0xFF, (value >> 8) & 0xFF))


def check(frame_with_fcs: bytes) -> bool:
    """True if ``frame_with_fcs`` (body + 2 FCS octets, LSB first) is intact."""
    if len(frame_with_fcs) < 2:
        return False
    body, recv = frame_with_fcs[:-2], frame_with_fcs[-2:]
    return fcs_bytes(body) == recv


__all__ = ["check", "fcs", "fcs_bytes"]

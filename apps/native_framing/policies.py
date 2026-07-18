"""Station policy checks kept separate from protocol integrity checks."""

from __future__ import annotations


def valid_ax25_address(body: bytes) -> bool:
    """Return whether the mandatory destination/source callsigns look like AX.25.

    This is a false-positive suppression policy, not part of the HDLC FCS.  It
    is therefore emitted as metadata by native decoders rather than changing
    their protocol integrity result.
    """

    if len(body) < 15:
        return False
    for base in (0, 7):
        for index in range(6):
            char = body[base + index] >> 1
            if not (0x41 <= char <= 0x5A or 0x30 <= char <= 0x39 or char == 0x20):
                return False
    return True


__all__ = ["valid_ax25_address"]

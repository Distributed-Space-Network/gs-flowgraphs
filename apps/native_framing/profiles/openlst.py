"""Bounded native OpenLST receive profile.

Parameters and stage order follow ``openlst_deframer.py`` in gr-satellites at
commit ``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.codes.openlst import decode_openlst_fec
from native_framing.crc import CRC16_CC11XX
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import pn9_bytes

SYNCWORD = "11010011100100011101001110010001"
CAPTURE_SIZE = 520
DEFAULT_SYNC_THRESHOLD = 4


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    if len(wire) != CAPTURE_SIZE:
        return None
    try:
        decoded = pn9_bytes(decode_openlst_fec(wire))
    except ValueError:
        return None
    if not decoded:
        return None
    declared_length = decoded[0] + 1
    if declared_length < 3 or declared_length > 256 or declared_length > len(decoded):
        return None
    frame = decoded[:declared_length]
    without_crc = CRC16_CC11XX.strip_if_valid(frame, byteorder="little")
    if without_crc is None or not without_crc:
        return None
    return DecodedFixedFrame(
        payload=without_crc[1:],
        metadata={
            "declared_length": declared_length,
            "fec": "CC1110 DN504 rate-1/2, constraint length 4",
            "interleaver": "4x4 dibit transpose",
            "whitening": "PN9 x^9+x^5+1 seed=0x1ff",
            "crc": CRC16_CC11XX.name,
            "crc_byteorder": "little",
        },
    )


def build_openlst(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="openlst",
        syncword=SYNCWORD,
        frame_size=CAPTURE_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = ["CAPTURE_SIZE", "DEFAULT_SYNC_THRESHOLD", "SYNCWORD", "build_openlst"]

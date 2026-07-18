"""Native GEOSCAN hard-bit receive profile.

Receive-chain parameters are adapted from gr-satellites
``python/components/deframers/geoscan_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2022 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.crc import CRC16_CC11XX
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import pn9_bytes

SYNCWORD = "10010011000010110101000111011110"
DEFAULT_FRAME_SIZE = 66
DEFAULT_SYNC_THRESHOLD = 4


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    decoded = pn9_bytes(wire)
    payload = CRC16_CC11XX.strip_if_valid(decoded, byteorder="big")
    if payload is None:
        return None
    return DecodedFixedFrame(
        payload=payload,
        metadata={
            "crc": CRC16_CC11XX.name,
            "frame_size_bytes": len(wire),
            "pn9": "x^9+x^5+1 seed=0x1ff lsb-first",
        },
    )


def build_geoscan(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="geoscan",
        syncword=SYNCWORD,
        frame_size=int(parameters.get("frame_size", DEFAULT_FRAME_SIZE)),
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = ["DEFAULT_FRAME_SIZE", "DEFAULT_SYNC_THRESHOLD", "SYNCWORD", "build_geoscan"]

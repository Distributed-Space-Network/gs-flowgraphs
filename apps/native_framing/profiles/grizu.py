"""Native Grizu-263A hard-bit receive profile.

The receive-chain order and constants are adapted from Daniel Estévez's GPLv3
gr-satellites ``grizu263a_deframer.py`` at pinned commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2021-2022 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.crc import CRC16_CC11XX
from native_framing.crop import cc11xx_packet, head_tail
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import pn9_bytes, reflect_bytes

SYNCWORD = "0000000100100011010001010110011110001001101010111100110111101111"
FRAME_SIZE = 258
DEFAULT_SYNC_THRESHOLD = 8
SCRAMBLER_SEED = 0x100


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    reflected = reflect_bytes(wire)
    whitened = pn9_bytes(reflected, seed=SCRAMBLER_SEED)
    packet_bytes = reflect_bytes(whitened)
    cropped = cc11xx_packet(packet_bytes, crc_bytes=2, maximum=FRAME_SIZE)
    if cropped is None:
        return None
    without_crc = CRC16_CC11XX.strip_if_valid(cropped, byteorder="big")
    if without_crc is None:
        return None
    payload = head_tail(without_crc, head=3, tail=1)
    if payload is None:
        return None
    return DecodedFixedFrame(
        payload=payload,
        metadata={
            "crc": CRC16_CC11XX.name,
            "packet_length": len(cropped),
            "pn9_seed": SCRAMBLER_SEED,
            "reflection_stages": 2,
        },
    )


def build_grizu(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="grizu263a",
        syncword=SYNCWORD,
        frame_size=FRAME_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = [
    "DEFAULT_SYNC_THRESHOLD",
    "FRAME_SIZE",
    "SCRAMBLER_SEED",
    "SYNCWORD",
    "build_grizu",
]

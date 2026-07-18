"""PN9/CC11xx-derived Reaktor Hello World and AALTO-1 profiles.

Receive-chain parameters are adapted from gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``:
``reaktor_hello_world_deframer.py`` and ``aalto1_deframer.py``.

Copyright 2019-2022 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.crc import CRC16_CC11XX, CRC16_X25, CrcSpec
from native_framing.crop import cc11xx_packet, head_tail
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import pn9_bytes

SYNCWORD = "00110101001011100011010100101110"
FRAME_SIZE = 258
DEFAULT_SYNC_THRESHOLD = 4


def _wire_decoder(spec: CrcSpec, byteorder: str):
    def decode(wire: bytes) -> DecodedFixedFrame | None:
        randomized = pn9_bytes(wire)
        cropped = cc11xx_packet(randomized, crc_bytes=2, maximum=FRAME_SIZE)
        if cropped is None:
            return None
        without_crc = spec.strip_if_valid(cropped, byteorder=byteorder)
        if without_crc is None:
            return None
        payload = head_tail(without_crc, head=3, tail=1)
        if payload is None:
            return None
        return DecodedFixedFrame(
            payload=payload,
            metadata={
                "crc": spec.name,
                "packet_length": len(cropped),
                "pn9": "x^9+x^5+1 seed=0x1ff lsb-first",
            },
        )

    return decode


def _build(canonical: str, spec: CrcSpec, byteorder: str, parameters: Mapping[str, object]):
    return FixedSyncFrameDecoder(
        canonical=canonical,
        syncword=SYNCWORD,
        frame_size=FRAME_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_wire_decoder(spec, byteorder),
    )


def build_reaktor(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return _build("reaktor_hello_world", CRC16_CC11XX, "big", parameters)


def build_aalto1(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return _build("aalto1", CRC16_X25, "little", parameters)


__all__ = [
    "DEFAULT_SYNC_THRESHOLD",
    "FRAME_SIZE",
    "SYNCWORD",
    "build_aalto1",
    "build_reaktor",
]

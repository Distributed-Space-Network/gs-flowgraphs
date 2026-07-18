"""AO-40 profile family components.

The uncoded receive-chain parameters are adapted from gr-satellites
``ao40_uncoded_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2019-2022 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.crc import CRC16_CCITT_FALSE
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder

UNCODED_SYNCWORD = "00111001000101011110110100110000"
UNCODED_FRAME_SIZE = 514
UNCODED_DEFAULT_SYNC_THRESHOLD = 3


def _decode_uncoded(wire: bytes) -> DecodedFixedFrame | None:
    payload = CRC16_CCITT_FALSE.strip_if_valid(wire, byteorder="big")
    if payload is None:
        return None
    return DecodedFixedFrame(
        payload=payload,
        metadata={"crc": CRC16_CCITT_FALSE.name, "frame_size_bytes": len(wire)},
    )


def build_ao40_uncoded(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="ao40_uncoded",
        syncword=UNCODED_SYNCWORD,
        frame_size=UNCODED_FRAME_SIZE,
        sync_threshold=int(
            parameters.get("sync_threshold", UNCODED_DEFAULT_SYNC_THRESHOLD)
        ),
        decode_wire=_decode_uncoded,
    )


__all__ = [
    "UNCODED_DEFAULT_SYNC_THRESHOLD",
    "UNCODED_FRAME_SIZE",
    "UNCODED_SYNCWORD",
    "build_ao40_uncoded",
]

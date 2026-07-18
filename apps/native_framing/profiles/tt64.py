"""Native TT-64 fixed frame, shortened RS, and CRC decoder.

The receive-chain parameters are adapted from gr-satellites
``tt64_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2019 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from gfsk_ax25.reedsolomon import RSCodec
from native_framing.crc import CRC16_ARC
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder

SYNCWORD = "0010110111010100"
FRAME_SIZE = 64
DATA_SIZE = 48
PARITY_SIZE = 16
DEFAULT_SYNC_THRESHOLD = 1

# gr-satellites decode_rs(8, 0x11d, 1, 1, 16, 1): GF(2^8), first root 1,
# primitive-element step 1, 16 parity symbols, one interleave path.
_CODEC = RSCodec(PARITY_SIZE, prim=0x11D, fcr=1, generator=2)


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    if len(wire) != FRAME_SIZE:
        return None
    decoded = _CODEC.decode_with_count(wire)
    if decoded is None:
        return None
    data, corrected = decoded
    if len(data) != DATA_SIZE:
        return None
    payload = CRC16_ARC.strip_if_valid(data, byteorder="little")
    if payload is None:
        return None
    return DecodedFixedFrame(
        payload=payload,
        corrected_symbols=corrected,
        metadata={
            "rs_field_polynomial": "0x11d",
            "rs_first_root": 1,
            "rs_parity_symbols": PARITY_SIZE,
            "crc": "CRC-16/ARC",
            "crc_byteorder": "little",
        },
    )


def build_tt64(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="tt64",
        syncword=SYNCWORD,
        frame_size=FRAME_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = [
    "DATA_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "FRAME_SIZE",
    "PARITY_SIZE",
    "SYNCWORD",
    "build_tt64",
]

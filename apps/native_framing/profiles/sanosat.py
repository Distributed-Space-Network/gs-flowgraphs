"""Native SanoSat-1 hard-bit receive profile.

Receive-chain construction is adapted from gr-satellites
``sanosat_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``. The wire contract is
qualified against ORION Space's mission protocol, packet generator, and
hardware recording at commit ``dfa5d131e2b41a02721cad0d4856b8ed2049f38f``.

Copyright 2022 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

from native_framing.crc import CRC16_CCITT_FALSE
from native_framing.crop import head_tail
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder

# ORION Space specifies 0x2DD4 transmitted LSB-first, observed as B4 2B on an
# MSB-first receiver. This resolves the stale 0xB22B comment in gr-satellites;
# its executable bit string was already the correct 0xB42B value.
SYNCWORD = "1011010000101011"
FRAME_SIZE = 135
DEFAULT_SYNC_THRESHOLD = 0


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    if not wire:
        return None
    packet_length = wire[0] + 5
    if packet_length > len(wire):
        return None
    packet = bytearray(wire[:packet_length])
    if len(packet) < 10:
        return None
    crc1_payload = CRC16_CCITT_FALSE.strip_if_valid(packet[:3], byteorder="little")
    if crc1_payload != packet[:1]:
        return None
    del packet[1:3]  # CRC1 is excluded from the CRC2 calculation upstream.
    without_crc = CRC16_CCITT_FALSE.strip_if_valid(packet, byteorder="little")
    if without_crc is None:
        return None
    if without_crc[1:5] != b"\xff\xff\x00\x00":
        return None
    # gr-satellites pdu_head_tail(3, 5) means mode 3 (drop five leading
    # bytes), not a three-byte head plus five-byte tail crop.
    payload = head_tail(without_crc, head=5)
    if not payload:
        return None
    return DecodedFixedFrame(
        payload=payload,
        metadata={
            "crc": CRC16_CCITT_FALSE.name,
            "crc_byteorder": "little",
            "crc1": "passed",
            "crc2": "passed",
            "packet_length_before_crc1_removal": packet_length,
            "syncword_source": "ORION Space 0x2dd4 LSB-first / received 0xb42b",
            "length_convention": "crc1 + message + crc2; total wire bytes = length + 5",
        },
    )


def build_sanosat(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="sanosat",
        syncword=SYNCWORD,
        frame_size=FRAME_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = ["DEFAULT_SYNC_THRESHOLD", "FRAME_SIZE", "SYNCWORD", "build_sanosat"]

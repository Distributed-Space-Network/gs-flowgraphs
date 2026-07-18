"""Bounded FEES type-254 LoRa telemetry preview.

Adapted on 2026-07-18 from ``ksy/fees.ksy`` and generated ``fees.py`` in
tinygs-decoders commit ``6b82a7f610349c2e46bcd97a0df38f9bdca1daf6``
under the project's explicit MIT treatment and attribution authorization.

The source schema assigns no engineering scales or integrity algorithm, so
this preview exposes the little-endian wire integers without inventing units.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import struct

from native_telemetry.types import FrameContext, ParserPreview

HEADER_BYTES = 3
TMI254_TYPE = 254
TMI254_BODY_BYTES = 44
TMI254_PACKET_BYTES = HEADER_BYTES + TMI254_BODY_BYTES
MAX_PACKET_BYTES = 255

_TMI254 = struct.Struct("<HIIIBHHHHHHHHHHHHHBH")
assert _TMI254.size == TMI254_BODY_BYTES

_FIELD_NAMES = (
    "frame_id",
    "timestamp",
    "unknown2",
    "unknown3",
    "unknown4",
    "unknown5",
    "vacd_2v5",
    "vbat",
    "vsol",
    "unknown6",
    "unknown7",
    "unknown8",
    "unknown9",
    "iload",
    "icharger",
    "unknown10",
    "unknown11",
    "unknown12",
    "unknown13",
    "unknown14",
)


def parse_fees(context: FrameContext) -> ParserPreview:
    """Parse FEES type 254 exactly and retain other types as bounded raw data."""

    data = context.payload
    if len(data) < HEADER_BYTES:
        raise ValueError("FEES packet is shorter than its three-byte header")
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"FEES packet exceeds {MAX_PACKET_BYTES} bytes")
    if data[0] != 0:
        raise ValueError("FEES msg_type_id0 must be 0x00")

    msg_type_id1 = data[1]
    msg_type_id2 = data[2]
    header = {
        "msg_type_id0": data[0],
        "msg_type_id1": msg_type_id1,
        "msg_type_id2": msg_type_id2,
    }
    if msg_type_id1 != TMI254_TYPE:
        return ParserPreview(
            status="ok",
            values={
                "kind": "unknown",
                "header": header,
                "payload_length": len(data) - HEADER_BYTES,
                "payload_hex": data[HEADER_BYTES:].hex(),
            },
        )

    if len(data) != TMI254_PACKET_BYTES:
        raise ValueError(
            f"FEES type 254 must be exactly {TMI254_PACKET_BYTES} bytes"
        )
    parsed = dict(zip(_FIELD_NAMES, _TMI254.unpack_from(data, HEADER_BYTES), strict=True))
    known = {
        name: int(parsed[name])
        for name in (
            "frame_id",
            "timestamp",
            "vacd_2v5",
            "vbat",
            "vsol",
            "iload",
            "icharger",
        )
    }
    unknown = {
        name: int(value) for name, value in parsed.items() if name.startswith("unknown")
    }
    return ParserPreview(
        status="ok",
        values={
            "kind": "tmi254",
            "header": header,
            "wire_values": known,
            "unknown_wire_values": unknown,
            "integrity": "not_specified",
        },
    )


__all__ = [
    "HEADER_BYTES",
    "MAX_PACKET_BYTES",
    "TMI254_BODY_BYTES",
    "TMI254_PACKET_BYTES",
    "TMI254_TYPE",
    "parse_fees",
]

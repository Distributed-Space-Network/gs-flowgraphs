"""Bounded Grizu-263A telemetry-header preview.

Adapted from ``ksy/grizu.ksy`` and its generated ``grizu.py`` in
tinygs-decoders commit ``6b82a7f610349c2e46bcd97a0df38f9bdca1daf6`` under the project's
explicit MIT treatment and attribution authorization dated 2026-07-18.

Modified into a declarative Construct schema with exact size, ASCII, and
calendar validation on 2026-07-18.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import datetime as dt

from construct import Bytes, Int8ub, Int16ul, Struct

from native_telemetry.types import FrameContext, ParserPreview

NORAD_ID = 51025
FRAMING = "grizu263a"
PACKET_BYTES = 60

GRIZU = Struct(
    "team_id" / Bytes(6),
    "year" / Int8ub,
    "month" / Int8ub,
    "date" / Int8ub,
    "hour" / Int8ub,
    "minute" / Int8ub,
    "second" / Int8ub,
    "temp" / Int16ul,
    "epstoobcina1_current" / Int16ul,
    "epstoobcina1_busvoltage" / Int16ul,
    "epsina2_current" / Int16ul,
    "epsina2_busvoltage" / Int16ul,
    "baseina3_current" / Int16ul,
    "baseina3_busvoltage" / Int16ul,
    "topina4_current" / Int16ul,
    "topina4_busvoltage" / Int16ul,
    "behindantenina5_current" / Int16ul,
    "behindantenina5_busvoltage" / Int16ul,
    "rightsideina6_current" / Int16ul,
    "rightsideina6_busvoltage" / Int16ul,
    "leftsideina7_current" / Int16ul,
    "leftsideina7_busvoltage" / Int16ul,
    "imumx" / Int16ul,
    "imumy" / Int16ul,
    "imumz" / Int16ul,
    "imuax" / Int16ul,
    "imuay" / Int16ul,
    "imuaz" / Int16ul,
    "imugx" / Int16ul,
    "imugy" / Int16ul,
    "imugz" / Int16ul,
)


def parse_grizu(context: FrameContext) -> ParserPreview:
    """Parse the exact post-NF-FRM-027 60-byte Grizu telemetry payload."""

    data = context.payload
    if len(data) != PACKET_BYTES:
        raise ValueError(f"Grizu telemetry must be exactly {PACKET_BYTES} bytes")
    parsed = GRIZU.parse(data)

    try:
        raw_team_id = bytes(parsed.team_id).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Grizu team ID must be ASCII") from exc
    if any(not 0x20 <= ord(char) <= 0x7E for char in raw_team_id):
        raise ValueError("Grizu team ID must contain printable ASCII only")
    team_id = raw_team_id.rstrip()
    if not team_id:
        raise ValueError("Grizu team ID must not be empty")
    if int(parsed.year) > 99:
        raise ValueError("Grizu year must be in the range 0..99")
    try:
        timestamp = dt.datetime(
            2000 + int(parsed.year),
            int(parsed.month),
            int(parsed.date),
            int(parsed.hour),
            int(parsed.minute),
            int(parsed.second),
            tzinfo=dt.timezone.utc,
        )
    except ValueError as exc:
        raise ValueError(f"Grizu timestamp fields are invalid: {exc}") from exc

    telemetry: dict[str, int] = {}
    for subcon in GRIZU.subcons[7:]:
        telemetry[subcon.name] = int(parsed[subcon.name])
    return ParserPreview(
        status="ok",
        values={
            "kind": "telemetry",
            "team_id": team_id,
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "timestamp_fields": {
                "year": int(parsed.year),
                "month": int(parsed.month),
                "date": int(parsed.date),
                "hour": int(parsed.hour),
                "minute": int(parsed.minute),
                "second": int(parsed.second),
            },
            "telemetry": telemetry,
        },
    )


__all__ = ["FRAMING", "GRIZU", "NORAD_ID", "PACKET_BYTES", "parse_grizu"]

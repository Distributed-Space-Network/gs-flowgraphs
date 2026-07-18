"""Bounded V-R3x LoRa beacon telemetry preview.

Adapted from ``vr3xDecoder.py`` in tinygs-decoders commit
``6b82a7f610349c2e46bcd97a0df38f9bdca1daf6`` under the project's
explicit MIT treatment and attribution authorization dated 2026-07-18.

The original decoder uses shared mutable storage, indexes unchecked packets,
and contains ``decodeStr == ""`` where assignment was intended.  This port is
stateless, requires the documented 45-byte packet, and preserves checksum
failure as an operator-visible integrity result rather than hiding telemetry.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import reduce
from operator import xor

from native_telemetry.types import FrameContext, ParserPreview

FRAMING = "LoRa"
PACKET_BYTES = 45
HEADER_BYTES = 4
BODY_BYTES = 41

SATELLITES = {
    0x3A: ("littlefoot", 47463),
    0x3B: ("petrie", 47467),
    0x3C: ("cera", 47524),
}
NORAD_IDS = frozenset(norad_id for _, norad_id in SATELLITES.values())

MISSION_PHASES = (
    "init",
    "cruise",
    "late",
    "experimental",
    "na",
    "na",
    "shutdown",
)


def _uint(data: bytes, start: int, end: int) -> int:
    return int.from_bytes(data[start:end], "big")


def _summary(body: bytes) -> dict[str, object]:
    selector = body[27]
    raw = body[28:37]
    common: dict[str, object] = {
        "selector": selector,
        "raw_hex": raw.hex(),
    }

    if selector == 0x52:
        raw_r1 = _uint(body, 28, 31)
        raw_temperature = _uint(body, 31, 34)
        raw_r2 = _uint(body, 34, 37)
        temperature_mv = raw_temperature * 1.49012e-07 * 1000
        return common | {
            "kind": "radiation",
            "r1_v": 2.5 - raw_r1 * 1.49012e-07,
            "temperature_c": -((129.0 - temperature_mv) * 0.403) + 25,
            "r2_v": 2.5 - raw_r2 * 1.49012e-07,
        }

    if selector in SATELLITES:
        peer_name, peer_norad = SATELLITES[selector]
        return common | {
            "kind": "range",
            "peer": {
                "id": selector,
                "name": peer_name,
                "norad_id": peer_norad,
            },
            "good_count": _uint(body, 28, 30),
            "bad_count": body[30],
            "last_range": _uint(body, 31, 34),
            "efe": _uint(body, 34, 37),
        }

    if selector in (0xFA, 0xFB, 0xFC):
        peer_id = (selector & 0x0F) | 0x30
        peer_name, peer_norad = SATELLITES[peer_id]
        return common | {
            "kind": "xlink",
            "peer": {
                "id": peer_id,
                "name": peer_name,
                "norad_id": peer_norad,
            },
            "good_uhf_count": body[28],
            "good_sband_count": body[29],
            "last_xlink_time_s": _uint(body, 30, 35) / 1000,
            "uhf_rssi_dbm": body[35] - 137,
            "sband_rssi_dbm": -body[36] / 2,
        }

    if selector == 0xA8:
        ecef_scalar = 8000 / (2**15)
        return common | {
            "kind": "gps",
            "time_of_week_s": _uint(body, 28, 31),
            "ecef_km": {
                "x": (_uint(body, 31, 33) - 32768) * ecef_scalar,
                "y": (_uint(body, 33, 35) - 32768) * ecef_scalar,
                "z": (_uint(body, 35, 37) - 32768) * ecef_scalar,
            },
        }

    return common | {"kind": "unknown"}


def parse_vr3x(context: FrameContext) -> ParserPreview:
    """Parse one exact V-R3x beacon while retaining XOR pass/fail evidence."""

    data = context.payload
    if len(data) != PACKET_BYTES:
        raise ValueError(f"V-R3x beacon must be exactly {PACKET_BYTES} bytes")

    satellite_id = data[1]
    if satellite_id not in SATELLITES:
        raise ValueError(f"V-R3x satellite ID 0x{satellite_id:02x} is unknown")
    satellite_name, satellite_norad = SATELLITES[satellite_id]
    if context.norad_id != satellite_norad:
        raise ValueError(
            f"V-R3x satellite ID 0x{satellite_id:02x} does not match NORAD "
            f"{context.norad_id}"
        )

    body = data[HEADER_BYTES:]
    assert len(body) == BODY_BYTES
    phase_code = body[16] >> 5
    if phase_code >= len(MISSION_PHASES):
        raise ValueError(f"V-R3x mission phase {phase_code} is outside 0..6")

    flags = body[16] & 0x1F
    secondary_flags = body[15] & 0x0F
    computed_xor = reduce(xor, body[:-1], 0)
    received_xor = body[-1]

    return ParserPreview(
        status="ok",
        values={
            "kind": "beacon",
            "header_hex": data[:HEADER_BYTES].hex(),
            "satellite": {
                "id": satellite_id,
                "name": satellite_name,
                "norad_id": satellite_norad,
            },
            "integrity": {
                "algorithm": "xor8",
                "status": "passed" if computed_xor == received_xor else "failed",
                "computed": computed_xor,
                "received": received_xor,
            },
            "spacecraft_time_s": _uint(body, 1, 6) / 1000,
            "battery_percent": body[20],
            "boot_count": body[0],
            "vbus_resets": body[6],
            "state_error_count": body[7],
            "time_rollovers": body[8],
            "timeouts": body[9],
            "ground_station_messages": body[10],
            "last_ground_station_rssi_dbm": body[37] - 137,
            "solar_charging_ma": body[11] * 2 if flags & 0x02 else 0,
            "uhf_crc_errors": body[39],
            "downlink_count": body[38],
            "gyro_deg_s": {
                "x": (body[17] - 128) * 0.234375,
                "y": (body[18] - 128) * 0.234375,
                "z": (body[19] - 128) * 0.234375,
            },
            "mission_phase": {
                "code": phase_code,
                "name": MISSION_PHASES[phase_code],
            },
            "state_errors": {
                "discovery": body[12] >> 4,
                "xlink": body[12] & 0x0F,
                "idle": body[13] >> 4,
                "beacon": body[13] & 0x0F,
                "radiation": body[14] >> 4,
                "downlink": body[14] & 0x0F,
                "range": body[15] >> 4,
            },
            "flags": {
                "low_battery": bool(flags & 0x01),
                "solar_charging": bool(flags & 0x02),
                "gps_on": bool(flags & 0x04),
                "mesh_leader": bool(flags & 0x08),
                "low_battery_timeout": bool(flags & 0x10),
                "gps_fix": bool(secondary_flags & 0x01),
                "secondary_reserved": secondary_flags >> 1,
            },
            "data_files": {
                "radiation": _uint(body, 21, 23),
                "range": _uint(body, 23, 25),
                "gps": _uint(body, 25, 27),
            },
            "summary": _summary(body),
        },
    )


__all__ = [
    "BODY_BYTES",
    "FRAMING",
    "HEADER_BYTES",
    "MISSION_PHASES",
    "NORAD_IDS",
    "PACKET_BYTES",
    "SATELLITES",
    "parse_vr3x",
]

"""Bounded NORBI/Norby telemetry preview attached to AX.25 UI information.

Adapted from ``ksy/norbi.ksy`` and ``norbyDecoder.py`` in tinygs-decoders
commit ``6b82a7f610349c2e46bcd97a0df38f9bdca1daf6`` under the project's explicit
MIT treatment and attribution authorization dated 2026-07-18.

Modified into a declarative Construct schema with exact native-frame handoff,
length, marker, and JSON bounds on 2026-07-18.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from construct import (
    Bytes,
    Int8sb,
    Int8ub,
    Int16sb,
    Int16sl,
    Int16ul,
    Int32sl,
    Int32ul,
    Struct,
)

from gfsk_ax25.ax25 import decode_ui
from native_telemetry.types import FrameContext, ParserPreview

NORAD_ID = 46494
FRAMING = "ax25_g3ruh"
HEADER_BYTES = 15
LENGTH_BASE = 14
TMI0_TYPE = 0
TMI0_PAYLOAD_BYTES = 128
TMI0_DECLARED_LENGTH = LENGTH_BASE + TMI0_PAYLOAD_BYTES
MAX_PACKET_BYTES = 256
MAX_RAW_PREVIEW_BYTES = 128

NORBY_HEADER = Struct(
    "length" / Int8ub,
    "receiver_address" / Int32ul,
    "transmitter_address" / Int32ul,
    "transaction_number" / Int16ul,
    "reserved" / Bytes(2),
    "message_type" / Int16sb,
)

TMI0 = Struct(
    "frame_start_mark" / Bytes(2),
    "frame_definition" / Int16ul,
    "frame_number" / Int16ul,
    "frame_generation_time" / Int32ul,
    "brk_title" / Bytes(24),
    "brk_number_active" / Int8ub,
    "brk_restarts_count_active" / Int32sl,
    "brk_current_mode_id" / Int8ub,
    "brk_transmitter_power_active" / Int8sb,
    "brk_temp_active" / Int8sb,
    "brk_module_state_active" / Bytes(2),
    "brk_voltage_offset_amplifier_active" / Int16ul,
    "brk_last_received_packet_rssi_active" / Int8sb,
    "brk_last_received_packet_snr_active" / Int8sb,
    "brk_archive_record_pointer" / Int16ul,
    "brk_last_received_packet_snr_inactive" / Int8sb,
    "ms_module_state" / Bytes(2),
    "ms_payload_state" / Bytes(2),
    "ms_temp" / Int8sb,
    "ms_pn_supply_state" / Int8ub,
    "sop_altitude_glonass" / Int32sl,
    "sop_latitude_glonass" / Int32sl,
    "sop_longitude_glonass" / Int32sl,
    "sop_date_time_glonass" / Int32ul,
    "sop_magnetic_induction_module" / Int16ul,
    "sop_angular_velocity_vector" / Bytes(6),
    "sop_angle_priority1" / Int16ul,
    "sop_angle_priority2" / Int16ul,
    "sop_mk_temp_dsg1" / Int8sb,
    "sop_mk_temp_dsg6" / Int8sb,
    "sop_board_temp" / Int8sb,
    "sop_state" / Bytes(2),
    "sop_state_dsg" / Bytes(6),
    "sop_orientation_number" / Int8ub,
    "ses_median_panel_x_temp_positive" / Int8sb,
    "ses_median_panel_x_temp_negative" / Int8sb,
    "ses_solar_panels_state" / Bytes(5),
    "ses_charge_level_m_ah" / Int16ul,
    "ses_battery_state" / Bytes(3),
    "ses_charging_keys_state" / Bytes(2),
    "ses_power_line_state" / Int8ub,
    "ses_total_charging_power" / Int16sl,
    "ses_total_generated_power" / Int16ul,
    "ses_total_power_load" / Int16ul,
    "ses_median_pmm_temp" / Int8sb,
    "ses_median_pam_temp" / Int8sb,
    "ses_median_pdm_temp" / Int8sb,
    "ses_module_state" / Bytes(3),
    "ses_voltage" / Int16ul,
    "crc16" / Int16ul,
)

_OPAQUE_FIELDS = frozenset(
    {
        "brk_module_state_active",
        "ms_module_state",
        "ms_payload_state",
        "sop_angular_velocity_vector",
        "sop_state",
        "sop_state_dsg",
        "ses_solar_panels_state",
        "ses_battery_state",
        "ses_charging_keys_state",
        "ses_module_state",
    }
)


def _header_values(header: object) -> dict[str, object]:
    return {
        "declared_length": int(header.length),
        "receiver_address": int(header.receiver_address),
        "transmitter_address": int(header.transmitter_address),
        "transaction_number": int(header.transaction_number),
        "reserved_hex": bytes(header.reserved).hex(),
        "message_type": int(header.message_type),
    }


def _tmi0_values(parsed: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in TMI0.subcons:
        if not name.name or name.name.startswith("_"):
            continue
        value = parsed[name.name]
        if name.name == "frame_start_mark":
            values["frame_start_mark_hex"] = bytes(value).hex()
        elif name.name == "brk_title":
            values[name.name] = bytes(value).decode("ascii").rstrip("\x00 ")
        elif name.name in _OPAQUE_FIELDS:
            values[f"{name.name}_hex"] = bytes(value).hex()
        else:
            values[name.name] = int(value)
    return values


def parse_norby(context: FrameContext) -> ParserPreview:
    """Parse a Norby application packet from a native AX.25 G3RUH frame body."""

    ui = decode_ui(context.payload)
    if ui is None:
        raise ValueError("Norby preview requires an AX.25 UI frame body")
    packet = ui.info
    if len(packet) < HEADER_BYTES:
        raise ValueError(f"Norby packet is shorter than {HEADER_BYTES} bytes")
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError(f"Norby packet exceeds {MAX_PACKET_BYTES} bytes")

    header = NORBY_HEADER.parse(packet[:HEADER_BYTES])
    declared_length = int(header.length)
    if declared_length < LENGTH_BASE:
        raise ValueError(f"Norby declared length is below {LENGTH_BASE}")
    expected_size = declared_length + 1
    if len(packet) != expected_size:
        raise ValueError(
            f"Norby packet size {len(packet)} does not match declared size {expected_size}"
        )

    payload = packet[HEADER_BYTES:]
    values: dict[str, object] = {
        "ax25": {"destination": ui.dest, "source": ui.src},
        "header": _header_values(header),
    }
    if int(header.message_type) != TMI0_TYPE:
        preview = payload[:MAX_RAW_PREVIEW_BYTES]
        values.update(
            {
                "kind": "raw",
                "data_hex": preview.hex(),
                "data_length": len(payload),
                "data_truncated": len(payload) > len(preview),
            }
        )
        return ParserPreview(status="ok", values=values)

    if declared_length != TMI0_DECLARED_LENGTH or len(payload) != TMI0_PAYLOAD_BYTES:
        raise ValueError(
            f"Norby TMI0 requires declared length {TMI0_DECLARED_LENGTH} "
            f"and {TMI0_PAYLOAD_BYTES} payload bytes"
        )
    if payload[:2] != b"\xf1\x0f":
        raise ValueError("Norby TMI0 frame marker must be f10f")
    parsed = TMI0.parse(payload)
    values.update(
        {
            "kind": "tmi0",
            "telemetry": _tmi0_values(parsed),
            "crc16_checked": False,
        }
    )
    return ParserPreview(status="ok", values=values)


__all__ = [
    "FRAMING",
    "HEADER_BYTES",
    "LENGTH_BASE",
    "MAX_PACKET_BYTES",
    "NORAD_ID",
    "NORBY_HEADER",
    "TMI0",
    "TMI0_DECLARED_LENGTH",
    "TMI0_PAYLOAD_BYTES",
    "parse_norby",
]

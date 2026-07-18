"""NF-TLM-003 exact-gated Norby telemetry-preview contracts."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from native_telemetry.output import derive_preview
from native_telemetry.profiles.norby import (
    TMI0_DECLARED_LENGTH,
    TMI0_PAYLOAD_BYTES,
    parse_norby,
)
from native_telemetry.registry import DEFAULT_REGISTRY
from native_telemetry.types import FrameContext

from gfsk_ax25.ax25 import encode_ui


def _tmi0_payload(*, marker: bytes = b"\xf1\x0f") -> bytes:
    parts = [
        marker,
        struct.pack("<HHI", 0x1234, 0x5678, 0x01020304),
        b"NORBY TMI0".ljust(24, b"\x00"),
        struct.pack("<BiBbb", 7, -123456, 9, -7, -8),
        bytes.fromhex("a1b2"),
        struct.pack("<HbbHb", 0x3456, -90, -11, 0x4567, -12),
        bytes.fromhex("c1c2d1d2"),
        struct.pack("<bB", -13, 1),
        struct.pack("<iiiIH", -1000, -2000, 3000, 0x11223344, 0x7788),
        bytes.fromhex("010203040506"),
        struct.pack("<HHbbb", 0x1111, 0x2222, -14, -15, -16),
        bytes.fromhex("3132414243444546"),
        struct.pack("<Bbb", 17, -18, -19),
        bytes.fromhex("5152535455"),
        struct.pack("<H", 0x6677),
        bytes.fromhex("6162637172"),
        struct.pack("<BhHHbbb", 1, -400, 0x8899, 0xAABB, -1, -2, -3),
        bytes.fromhex("818283"),
        struct.pack("<HH", 0xCCDD, 0xBEEF),
    ]
    payload = b"".join(parts)
    assert len(payload) == TMI0_PAYLOAD_BYTES
    return payload


def _application_packet(
    payload: bytes,
    *,
    message_type: int = 0,
    declared_length: int | None = None,
) -> bytes:
    length = 14 + len(payload) if declared_length is None else declared_length
    return b"".join(
        [
            bytes([length]),
            struct.pack("<IIH", 0x01020304, 0xA0B0C0D0, 0x1234),
            bytes.fromhex("aabb"),
            struct.pack(">h", message_type),
            payload,
        ]
    )


def _context(
    packet: bytes,
    *,
    norad_id: int | None = 46494,
    framing: str = "AX.25 G3RUH",
    as_ax25: bool = True,
) -> FrameContext:
    payload = encode_ui(dest="NORBI", src="GROUND", info=packet) if as_ax25 else packet
    return FrameContext(
        source_frame_id="c" * 64,
        source_line=3,
        norad_id=norad_id,
        framing=framing,
        payload=payload,
    )


def _norby_result(packet: bytes, **context: object):
    results = DEFAULT_REGISTRY.parse(_context(packet, **context))
    return next(result for result in results if result.parser == "norby")


def test_norby_tmi0_golden_packet_covers_endian_signed_marker_and_ax25_handoff() -> None:
    packet = _application_packet(_tmi0_payload())
    assert len(packet) == TMI0_DECLARED_LENGTH + 1

    result = _norby_result(packet)

    assert result.status == "ok"
    assert result.values["kind"] == "tmi0"
    assert result.values["ax25"] == {"destination": "NORBI", "source": "GROUND"}
    assert result.values["header"] == {
        "declared_length": 142,
        "message_type": 0,
        "receiver_address": 0x01020304,
        "reserved_hex": "aabb",
        "transaction_number": 0x1234,
        "transmitter_address": 0xA0B0C0D0,
    }
    telemetry = result.values["telemetry"]
    assert telemetry["frame_start_mark_hex"] == "f10f"
    assert telemetry["frame_definition"] == 0x1234
    assert telemetry["frame_number"] == 0x5678
    assert telemetry["frame_generation_time"] == 0x01020304
    assert telemetry["brk_title"] == "NORBY TMI0"
    assert telemetry["brk_restarts_count_active"] == -123456
    assert telemetry["brk_transmitter_power_active"] == -7
    assert telemetry["sop_altitude_glonass"] == -1000
    assert telemetry["sop_longitude_glonass"] == 3000
    assert telemetry["ses_total_charging_power"] == -400
    assert telemetry["ses_median_pdm_temp"] == -3
    assert telemetry["crc16"] == 0xBEEF
    assert result.values["crc16_checked"] is False


def test_norby_unknown_type_and_length_14_boundary_are_bounded_raw() -> None:
    raw = _norby_result(_application_packet(b"RAW", message_type=-2))
    assert raw.status == "ok"
    assert raw.values["kind"] == "raw"
    assert raw.values["header"]["message_type"] == -2
    assert raw.values["data_hex"] == b"RAW".hex()

    minimum = _norby_result(_application_packet(b"", message_type=1))
    assert minimum.status == "ok"
    assert minimum.values["header"]["declared_length"] == 14
    assert minimum.values["data_length"] == 0


@pytest.mark.parametrize(
    ("packet", "message"),
    [
        (_application_packet(b"", message_type=1, declared_length=13), "below 14"),
        (_application_packet(b"abc", message_type=1, declared_length=18), "declared size 19"),
        (_application_packet(_tmi0_payload()[:-1]), "TMI0 requires"),
        (_application_packet(_tmi0_payload(marker=b"\x00\x00")), "marker must be f10f"),
    ],
    ids=["length-underflow", "declared-mismatch", "tmi0-truncated", "marker"],
)
def test_norby_rejects_bad_lengths_and_marker(packet: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_norby(_context(packet))
    result = _norby_result(packet)
    assert result.status == "error"
    assert message in result.diagnostic


def test_norby_rejects_application_bytes_without_native_ax25_ui_envelope() -> None:
    packet = _application_packet(_tmi0_payload())
    with pytest.raises(ValueError, match="AX.25 UI frame body"):
        parse_norby(_context(packet, as_ax25=False))


def test_norby_rejects_packet_larger_than_declared_byte_can_represent() -> None:
    packet = bytes([255]) + bytes(256)
    with pytest.raises(ValueError, match="exceeds 256"):
        parse_norby(_context(packet))
    assert _norby_result(packet).status == "error"


def test_norby_sidecar_preserves_native_ax25_source_linkage(tmp_path: Path) -> None:
    packet = _application_packet(_tmi0_payload())
    body = encode_ui(dest="NORBI", src="GROUND", info=packet)
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        json.dumps(
            {
                "framing": "AX.25 G3RUH",
                "integrity": "passed",
                "payload_hex": body.hex(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    derive_preview(frames, sidecar, norad_id=46494, pass_framing="AX.25 G3RUH")

    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    record = next(item for item in records if item["parser"] == "norby")
    assert record["source_line"] == 1
    assert len(record["source_frame_id"]) == 64
    assert record["payload_sha256"] == hashlib.sha256(body).hexdigest()
    assert record["values"]["kind"] == "tmi0"


def test_norby_requires_exact_norad_and_ax25_g3ruh_framing() -> None:
    packet = _application_packet(_tmi0_payload())
    assert all(
        result.parser != "norby"
        for result in DEFAULT_REGISTRY.parse(_context(packet, norad_id=46493))
    )
    assert all(
        result.parser != "norby"
        for result in DEFAULT_REGISTRY.parse(_context(packet, framing="AX.25"))
    )
    assert _norby_result(packet, framing="ax25_g3ruh").status == "ok"

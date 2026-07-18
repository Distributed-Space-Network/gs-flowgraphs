"""NF-TLM-005 exact-gated Grizu-263A telemetry-preview contracts."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_CC11XX
from native_framing.linecode import pn9_bytes, reflect_bytes
from native_framing.profiles.grizu import FRAME_SIZE, SCRAMBLER_SEED, SYNCWORD
from native_telemetry.profiles.grizu import PACKET_BYTES, parse_grizu
from native_telemetry.registry import DEFAULT_REGISTRY
from native_telemetry.types import FrameContext

_FIELDS = (
    "temp",
    "epstoobcina1_current",
    "epstoobcina1_busvoltage",
    "epsina2_current",
    "epsina2_busvoltage",
    "baseina3_current",
    "baseina3_busvoltage",
    "topina4_current",
    "topina4_busvoltage",
    "behindantenina5_current",
    "behindantenina5_busvoltage",
    "rightsideina6_current",
    "rightsideina6_busvoltage",
    "leftsideina7_current",
    "leftsideina7_busvoltage",
    "imumx",
    "imumy",
    "imumz",
    "imuax",
    "imuay",
    "imuaz",
    "imugx",
    "imugy",
    "imugz",
)


def _payload(
    *,
    team_id: bytes = b"GRIZU ",
    timestamp: tuple[int, int, int, int, int, int] = (24, 2, 29, 23, 58, 59),
) -> bytes:
    values = tuple(0x1000 + index * 0x101 for index in range(len(_FIELDS)))
    payload = struct.pack("<6s6B24H", team_id, *timestamp, *values)
    assert len(payload) == PACKET_BYTES
    return payload


def _context(
    payload: bytes,
    *,
    norad_id: int | None = 51025,
    framing: str = "Grizu-263A",
) -> FrameContext:
    return FrameContext(
        source_frame_id="d" * 64,
        source_line=4,
        norad_id=norad_id,
        framing=framing,
        payload=payload,
    )


def _grizu_result(payload: bytes, **context: object):
    results = DEFAULT_REGISTRY.parse(_context(payload, **context))
    return next(result for result in results if result.parser == "grizu263a")


def _framing_wire(payload: bytes) -> np.ndarray:
    packet_without_crc = bytes([len(payload) + 3, 0xAA, 0x55]) + payload + b"\x7e"
    packet = CRC16_CC11XX.append(packet_without_crc, byteorder="big")
    decoded_capture = packet + bytes(FRAME_SIZE - len(packet))
    wire = reflect_bytes(
        pn9_bytes(reflect_bytes(decoded_capture), seed=SCRAMBLER_SEED)
    )
    sync = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
    return np.concatenate((sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


def test_grizu_golden_payload_is_exact_little_endian_and_calendar_checked() -> None:
    payload = _payload()
    result = _grizu_result(payload)

    assert result.status == "ok"
    assert result.values["kind"] == "telemetry"
    assert result.values["team_id"] == "GRIZU"
    assert result.values["timestamp_utc"] == "2024-02-29T23:58:59Z"
    assert result.values["timestamp_fields"] == {
        "date": 29,
        "hour": 23,
        "minute": 58,
        "month": 2,
        "second": 59,
        "year": 24,
    }
    expected = {
        name: 0x1000 + index * 0x101 for index, name in enumerate(_FIELDS)
    }
    assert result.values["telemetry"] == expected
    assert result.values["telemetry"]["temp"] == 0x1000
    assert result.values["telemetry"]["imugz"] == 0x2717


def test_grizu_nf_frm_027_output_hands_off_byte_exactly_to_telemetry() -> None:
    payload = _payload()
    frames = build_decoder("Grizu-263A").push(_framing_wire(payload))
    assert [frame.payload for frame in frames] == [payload]

    result = parse_grizu(_context(frames[0].payload))
    assert result.status == "ok"
    assert result.values["team_id"] == "GRIZU"
    assert result.values["telemetry"]["imugz"] == 0x2717


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload()[:-1], "exactly 60"),
        (_payload() + b"\x00", "exactly 60"),
        (_payload(team_id=b"GR\x00ZU "), "printable ASCII"),
        (_payload(team_id=b"      "), "must not be empty"),
        (_payload(timestamp=(100, 1, 1, 0, 0, 0)), "year must be"),
        (_payload(timestamp=(24, 2, 30, 0, 0, 0)), "timestamp fields are invalid"),
        (_payload(timestamp=(24, 1, 1, 24, 0, 0)), "timestamp fields are invalid"),
    ],
    ids=[
        "truncated",
        "extended",
        "team-control",
        "team-empty",
        "year",
        "date",
        "time",
    ],
)
def test_grizu_rejects_size_ascii_and_timestamp_errors(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_grizu(_context(payload))
    result = _grizu_result(payload)
    assert result.status == "error"
    assert message in result.diagnostic


def test_grizu_requires_exact_norad_and_canonical_framing() -> None:
    payload = _payload()
    assert all(
        result.parser != "grizu263a"
        for result in DEFAULT_REGISTRY.parse(_context(payload, norad_id=51024))
    )
    assert all(
        result.parser != "grizu263a"
        for result in DEFAULT_REGISTRY.parse(_context(payload, framing="AX.25"))
    )
    assert _grizu_result(payload, framing="grizu263a").status == "ok"

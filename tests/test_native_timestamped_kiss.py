"""Deterministic source-time and timestamped-KISS compatibility tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from native_framing.output import (
    TimestampedFrame,
    decode_timestamped_kiss,
    encode_timestamped_kiss,
    json_record,
    unix_milliseconds,
    utc_from_sample_offset,
)

from gfsk_ax25.kiss import FEND, FESC, kiss_decode_commands, kiss_encode


def test_source_sample_offset_derives_replay_stable_utc():
    start = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    timestamp = utc_from_sample_offset(start, 48_012, 48_000.0)
    assert timestamp == datetime(2026, 7, 18, 12, 0, 1, 250, tzinfo=timezone.utc)
    assert unix_milliseconds(timestamp) == 1_784_376_001_000
    assert utc_from_sample_offset(start, 48_012, 48_000.0) == timestamp


def test_timestamped_kiss_is_byte_exact_and_reserved_payload_round_trips():
    timestamp = datetime(2026, 7, 18, 12, 34, 56, 789000, tzinfo=timezone.utc)
    payload = bytes([0x01, FEND, 0x02, FESC, 0x03])
    encoded = encode_timestamped_kiss(TimestampedFrame(timestamp, payload))
    commands = kiss_decode_commands(encoded)
    assert [(port, command) for port, command, _ in commands] == [(0, 9), (0, 0)]
    assert commands[0][2] == unix_milliseconds(timestamp).to_bytes(8, "big")
    assert commands[1][2] == payload
    assert decode_timestamped_kiss(encoded) == [TimestampedFrame(timestamp, payload)]


def test_multiple_timestamp_data_pairs_keep_their_own_time():
    first = TimestampedFrame(datetime(2026, 1, 1, tzinfo=timezone.utc), b"first")
    second = TimestampedFrame(datetime(2026, 1, 2, tzinfo=timezone.utc), b"second")
    encoded = encode_timestamped_kiss(first) + encode_timestamped_kiss(second)
    assert decode_timestamped_kiss(encoded) == [first, second]


def test_data_without_valid_timestamp_has_no_wall_clock_fallback():
    data_only = kiss_encode(b"payload", command=0)
    bad_timestamp = kiss_encode(b"short", command=9) + data_only
    assert decode_timestamped_kiss(data_only) == []
    assert decode_timestamped_kiss(bad_timestamp) == []


def test_malformed_escape_is_rejected_without_damaging_later_records():
    malformed = bytes([FEND, 0, FESC, 0x01, FEND])
    valid = TimestampedFrame(datetime(2026, 1, 1, tzinfo=timezone.utc), b"valid")
    assert decode_timestamped_kiss(malformed + encode_timestamped_kiss(valid)) == [valid]


def test_json_record_preserves_payload_and_millisecond_timestamp():
    frame = TimestampedFrame(
        datetime(2026, 1, 1, 0, 0, 0, 123000, tzinfo=timezone.utc), b"\x00\xc0\xdb"
    )
    record = json_record(frame, framing="geoscan")
    assert record == {
        "timestamp": "2026-01-01T00:00:00.123000Z",
        "unix_ms": 1_767_225_600_123,
        "framing": "geoscan",
        "payload_hex": "00c0db",
    }


def test_timestamp_contract_validation_is_fail_closed():
    naive = datetime(2026, 1, 1)
    aware = naive.replace(tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        TimestampedFrame(naive, b"x")
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_from_sample_offset(naive, 0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        utc_from_sample_offset(aware, -1, 1)
    with pytest.raises(ValueError, match="positive"):
        utc_from_sample_offset(aware, 0, 0)
    with pytest.raises(ValueError, match="framing"):
        json_record(TimestampedFrame(aware, b"x"), framing="")

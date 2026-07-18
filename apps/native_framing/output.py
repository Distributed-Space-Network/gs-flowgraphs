"""Deterministic frame timestamps and SatNOGS-compatible timestamped KISS.

The compatibility contract is derived from the pinned SatNOGS client
``satnogsclient/radio/grsat.py`` at commit
``60d9902933d86a6133935586a0da4952a5803f9e``. That AGPL source is used only
as a behavioral reference; this module is a clean repository-owned
implementation over the existing GPLv3 KISS codec.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from gfsk_ax25.kiss import kiss_decode_commands, kiss_encode

TIMESTAMP_COMMAND = 0x09
DATA_COMMAND = 0x00


@dataclass(frozen=True)
class TimestampedFrame:
    timestamp: datetime
    payload: bytes

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("frame timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "payload", bytes(self.payload))


def utc_from_sample_offset(
    pass_start: datetime, source_sample_offset: int, sample_rate_hz: float
) -> datetime:
    """Derive frame UTC from capture start and a source-domain sample offset."""

    if pass_start.tzinfo is None or pass_start.utcoffset() is None:
        raise ValueError("pass_start must be timezone-aware")
    if source_sample_offset < 0:
        raise ValueError("source_sample_offset must be non-negative")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    start_utc = pass_start.astimezone(timezone.utc)
    return start_utc + timedelta(seconds=source_sample_offset / sample_rate_hz)


def unix_milliseconds(timestamp: datetime) -> int:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    milliseconds = round(timestamp.timestamp() * 1000)
    if not 0 <= milliseconds < (1 << 64):
        raise ValueError("timestamp is outside the unsigned 64-bit millisecond range")
    return milliseconds


def encode_timestamped_kiss(frame: TimestampedFrame) -> bytes:
    """Encode command-9 Unix milliseconds followed by command-0 frame data."""

    stamp = unix_milliseconds(frame.timestamp).to_bytes(8, "big")
    return kiss_encode(stamp, command=TIMESTAMP_COMMAND, port=0) + kiss_encode(
        frame.payload, command=DATA_COMMAND, port=0
    )


def decode_timestamped_kiss(stream: bytes) -> list[TimestampedFrame]:
    """Decode timestamp/data pairs without any process-wall-clock fallback."""

    timestamp: datetime | None = None
    output: list[TimestampedFrame] = []
    for port, command, payload in kiss_decode_commands(stream, strict=True):
        if port != 0:
            continue
        if command == TIMESTAMP_COMMAND:
            if len(payload) != 8:
                continue
            milliseconds = int.from_bytes(payload, "big")
            timestamp = datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
        elif command == DATA_COMMAND and timestamp is not None:
            output.append(TimestampedFrame(timestamp, payload))
    return output


def json_record(frame: TimestampedFrame, *, framing: str) -> dict[str, object]:
    """Return a deterministic JSON-ready record without changing payload bytes."""

    if not framing.strip():
        raise ValueError("framing must not be empty")
    return {
        "timestamp": frame.timestamp.isoformat().replace("+00:00", "Z"),
        "unix_ms": unix_milliseconds(frame.timestamp),
        "framing": framing,
        "payload_hex": frame.payload.hex(),
    }


__all__ = [
    "DATA_COMMAND",
    "TIMESTAMP_COMMAND",
    "TimestampedFrame",
    "decode_timestamped_kiss",
    "encode_timestamped_kiss",
    "json_record",
    "unix_milliseconds",
    "utc_from_sample_offset",
]

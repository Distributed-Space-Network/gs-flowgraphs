"""NF-TLM-004/006 bounded schema previews without guessed mission activation."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from native_framing.provenance import load_manifest
from native_telemetry.profiles.fees import (
    MAX_PACKET_BYTES as FEES_MAX_PACKET_BYTES,
)
from native_telemetry.profiles.fees import TMI254_PACKET_BYTES, parse_fees
from native_telemetry.profiles.sdsat import MAGIC, MAX_RELAY_BYTES, parse_sdsat
from native_telemetry.registry import DEFAULT_REGISTRY
from native_telemetry.types import FrameContext

_TEST_FILE = Path(__file__).resolve()
_MANIFEST = _TEST_FILE.parent / "fixtures" / "native_telemetry" / "MANIFEST.csv"
_TINYGS = _TEST_FILE.parents[2] / "related-projects" / "tinygs-decoders"


def _context(payload: bytes) -> FrameContext:
    return FrameContext(
        source_frame_id="e" * 64,
        source_line=6,
        norad_id=None,
        framing="LoRa",
        payload=payload,
    )


def _fees_tmi254() -> bytes:
    body = struct.pack(
        "<HIIIBHHHHHHHHHHHHHBH",
        0x1234,
        0x01020304,
        0x11121314,
        0x21222324,
        0x31,
        0x3233,
        0x3435,
        0x3637,
        0x3839,
        0x3A3B,
        0x3C3D,
        0x3E3F,
        0x4041,
        0x4243,
        0x4445,
        0x4647,
        0x4849,
        0x4A4B,
        0x4C,
        0x4D4E,
    )
    packet = bytes((0, 254, 7)) + body
    assert len(packet) == TMI254_PACKET_BYTES
    return packet


def test_fees_type254_golden_packet_is_exact_little_endian_without_guessed_units():
    preview = parse_fees(_context(_fees_tmi254()))
    assert preview.status == "ok"
    assert preview.values["kind"] == "tmi254"
    assert preview.values["header"] == {
        "msg_type_id0": 0,
        "msg_type_id1": 254,
        "msg_type_id2": 7,
    }
    assert preview.values["wire_values"] == {
        "frame_id": 0x1234,
        "timestamp": 0x01020304,
        "vacd_2v5": 0x3435,
        "vbat": 0x3637,
        "vsol": 0x3839,
        "iload": 0x4243,
        "icharger": 0x4445,
    }
    assert preview.values["unknown_wire_values"]["unknown14"] == 0x4D4E
    assert preview.values["integrity"] == "not_specified"


def test_fees_unknown_type_is_bounded_raw_and_tmi254_size_is_exact():
    unknown = parse_fees(_context(bytes((0, 99, 2)) + b"\x01\x02\x03"))
    assert unknown.values == {
        "kind": "unknown",
        "header": {"msg_type_id0": 0, "msg_type_id1": 99, "msg_type_id2": 2},
        "payload_length": 3,
        "payload_hex": "010203",
    }

    packet = _fees_tmi254()
    for malformed in (b"", b"\x00\xfe", packet[:-1], packet + b"\x00"):
        with pytest.raises(ValueError):
            parse_fees(_context(malformed))
    with pytest.raises(ValueError, match="msg_type_id0"):
        parse_fees(_context(b"\x01\x63\x00"))
    with pytest.raises(ValueError, match="exceeds"):
        parse_fees(_context(bytes(FEES_MAX_PACKET_BYTES + 1)))


def test_sdsat_magic_utf8_empty_maximum_and_display_safety():
    empty = parse_sdsat(_context(MAGIC))
    assert empty.values["relay_text"] == ""
    assert empty.values["relay_utf8_bytes"] == 0

    relay = "Hello, ground!\nRTL\u202e".encode()
    preview = parse_sdsat(_context(MAGIC + relay))
    assert preview.values["kind"] == "relay_text"
    assert preview.values["relay_text"] == "Hello, ground!\\u000aRTL\\u202e"
    assert preview.values["relay_hex"] == relay.hex()

    maximum = parse_sdsat(_context(MAGIC + b"A" * MAX_RELAY_BYTES))
    assert maximum.values["relay_utf8_bytes"] == MAX_RELAY_BYTES
    assert maximum.values["relay_text"] == "A" * MAX_RELAY_BYTES


def test_sdsat_rejects_identity_utf8_and_size_errors():
    with pytest.raises(ValueError, match="magic"):
        parse_sdsat(_context(b"SDSAT,LORA BROKEN:"))
    with pytest.raises(ValueError, match="UTF-8"):
        parse_sdsat(_context(MAGIC + b"\xff"))
    with pytest.raises(ValueError, match="exceeds"):
        parse_sdsat(_context(MAGIC + b"A" * (MAX_RELAY_BYTES + 1)))


def test_unattributed_mission_profiles_remain_absent_from_production_registry():
    names = {spec.name for spec in DEFAULT_REGISTRY.specs}
    assert "fees" not in names
    assert "sdsat" not in names


def test_fees_and_sdsat_sources_are_hash_pinned():
    artifacts = {
        item.artifact_id: item
        for item in load_manifest(_MANIFEST)
        if item.artifact_id.startswith(("tinygs-fees-", "tinygs-sdsat-"))
    }
    assert set(artifacts) == {
        "tinygs-fees-generated",
        "tinygs-fees-schema",
        "tinygs-sdsat-generated",
        "tinygs-sdsat-schema",
    }
    for artifact in artifacts.values():
        assert artifact.source_commit == "6b82a7f610349c2e46bcd97a0df38f9bdca1daf6"
        source = _TINYGS / artifact.source_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact.sha256

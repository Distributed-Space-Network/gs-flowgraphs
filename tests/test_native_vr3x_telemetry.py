"""NF-TLM-008 exact-gated V-R3x telemetry-preview contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from native_framing.provenance import load_manifest
from native_telemetry.output import derive_preview
from native_telemetry.profiles.vr3x import PACKET_BYTES, SATELLITES, parse_vr3x
from native_telemetry.registry import DEFAULT_REGISTRY
from native_telemetry.types import FrameContext

_TEST_FILE = Path(__file__).resolve()
_GS_ROOT = _TEST_FILE.parents[2]
_MANIFEST = _TEST_FILE.parent / "fixtures" / "native_telemetry" / "MANIFEST.csv"
_TINYGS = _GS_ROOT / "related-projects" / "tinygs-decoders"


def _context(
    payload: bytes,
    *,
    norad_id: int | None = 47463,
    framing: str = "LoRa",
) -> FrameContext:
    return FrameContext(
        source_frame_id="e" * 64,
        source_line=8,
        norad_id=norad_id,
        framing=framing,
        payload=payload,
    )


def _packet(
    *,
    satellite_id: int = 0x3A,
    phase: int = 2,
    summary_selector: int = 0x52,
    summary_data: bytes = bytes.fromhex("010203040506070809"),
    valid_xor: bool = True,
) -> bytes:
    assert len(summary_data) == 9
    body = bytearray(41)
    body[0] = 7
    body[1:6] = (12_345_678).to_bytes(5, "big")
    body[6:12] = bytes((1, 2, 3, 4, 5, 25))
    body[12:16] = bytes.fromhex("a3b4c5b1")
    body[16] = (phase << 5) | 0x1F
    body[17:20] = bytes((129, 127, 132))
    body[20] = 88
    body[21:27] = bytes.fromhex("123456789abc")
    body[27] = summary_selector
    body[28:37] = summary_data
    body[37:40] = bytes((100, 9, 10))
    checksum = 0
    for value in body[:-1]:
        checksum ^= value
    body[40] = checksum ^ (not valid_xor)
    packet = bytes((0xD3, satellite_id, 0x01, 0x02)) + bytes(body)
    assert len(packet) == PACKET_BYTES
    return packet


def _vr3x_result(payload: bytes, **context: object):
    results = DEFAULT_REGISTRY.parse(_context(payload, **context))
    return next(result for result in results if result.parser == "vr3x")


def test_vr3x_golden_packet_covers_units_flags_fields_and_xor() -> None:
    result = _vr3x_result(_packet())

    assert result.status == "ok"
    assert result.values["kind"] == "beacon"
    assert result.values["header_hex"] == "d33a0102"
    assert result.values["satellite"] == {
        "id": 0x3A,
        "name": "littlefoot",
        "norad_id": 47463,
    }
    assert result.values["integrity"]["status"] == "passed"
    assert result.values["integrity"]["computed"] == result.values["integrity"]["received"]
    assert result.values["spacecraft_time_s"] == pytest.approx(12_345.678)
    assert result.values["battery_percent"] == 88
    assert result.values["boot_count"] == 7
    assert result.values["vbus_resets"] == 1
    assert result.values["state_error_count"] == 2
    assert result.values["time_rollovers"] == 3
    assert result.values["timeouts"] == 4
    assert result.values["ground_station_messages"] == 5
    assert result.values["last_ground_station_rssi_dbm"] == -37
    assert result.values["solar_charging_ma"] == 50
    assert result.values["uhf_crc_errors"] == 10
    assert result.values["downlink_count"] == 9
    assert result.values["gyro_deg_s"] == {
        "x": 0.234375,
        "y": -0.234375,
        "z": 0.9375,
    }
    assert result.values["mission_phase"] == {"code": 2, "name": "late"}
    assert result.values["state_errors"] == {
        "beacon": 4,
        "discovery": 10,
        "downlink": 5,
        "idle": 11,
        "radiation": 12,
        "range": 11,
        "xlink": 3,
    }
    assert result.values["flags"] == {
        "gps_fix": True,
        "gps_on": True,
        "low_battery": True,
        "low_battery_timeout": True,
        "mesh_leader": True,
        "secondary_reserved": 0,
        "solar_charging": True,
    }
    assert result.values["data_files"] == {
        "gps": 0x9ABC,
        "radiation": 0x1234,
        "range": 0x5678,
    }
    summary = result.values["summary"]
    assert summary["kind"] == "radiation"
    assert summary["raw_hex"] == "010203040506070809"
    assert summary["r1_v"] == pytest.approx(2.5 - 0x010203 * 1.49012e-07)
    assert summary["temperature_c"] == pytest.approx(
        -((129 - 0x040506 * 1.49012e-07 * 1000) * 0.403) + 25
    )
    assert summary["r2_v"] == pytest.approx(2.5 - 0x070809 * 1.49012e-07)


@pytest.mark.parametrize(
    ("satellite_id", "norad_id", "name"),
    [
        (0x3A, 47463, "littlefoot"),
        (0x3B, 47467, "petrie"),
        (0x3C, 47524, "cera"),
    ],
)
def test_vr3x_all_satellite_ids_require_their_exact_norad(
    satellite_id: int, norad_id: int, name: str
) -> None:
    result = _vr3x_result(_packet(satellite_id=satellite_id), norad_id=norad_id)
    assert result.status == "ok"
    assert result.values["satellite"] == {
        "id": satellite_id,
        "name": name,
        "norad_id": norad_id,
    }

    other_norad = next(value[1] for key, value in SATELLITES.items() if key != satellite_id)
    mismatch = _vr3x_result(
        _packet(satellite_id=satellite_id), norad_id=other_norad
    )
    assert mismatch.status == "error"
    assert "does not match NORAD" in mismatch.diagnostic


@pytest.mark.parametrize(
    ("selector", "data", "kind", "expected"),
    [
        (
            0x3B,
            bytes.fromhex("123456789abcdeffff"),
            "range",
            {
                "good_count": 0x1234,
                "bad_count": 0x56,
                "last_range": 0x789ABC,
                "efe": 0xDEFFFF,
            },
        ),
        (
            0xFC,
            bytes.fromhex("0708010203040564fe"),
            "xlink",
            {
                "good_uhf_count": 7,
                "good_sband_count": 8,
                "last_xlink_time_s": 4_328_719.365,
                "uhf_rssi_dbm": -37,
                "sband_rssi_dbm": -127.0,
            },
        ),
        (
            0xA8,
            bytes.fromhex("01020380007fff9000"),
            "gps",
            {"time_of_week_s": 0x010203},
        ),
    ],
)
def test_vr3x_summary_variants_are_big_endian_and_bounded(
    selector: int, data: bytes, kind: str, expected: dict[str, object]
) -> None:
    summary = _vr3x_result(
        _packet(summary_selector=selector, summary_data=data)
    ).values["summary"]
    assert summary["kind"] == kind
    for key, value in expected.items():
        if isinstance(value, float):
            assert summary[key] == pytest.approx(value)
        else:
            assert summary[key] == value

    if selector == 0x3B:
        assert summary["peer"] == {"id": 0x3B, "name": "petrie", "norad_id": 47467}
    if selector == 0xFC:
        assert summary["peer"] == {"id": 0x3C, "name": "cera", "norad_id": 47524}
    if selector == 0xA8:
        assert summary["ecef_km"] == {"x": 0.0, "y": -0.244140625, "z": 1000.0}


def test_vr3x_unknown_summary_and_failed_xor_remain_explicit() -> None:
    payload = _packet(
        summary_selector=0x00,
        summary_data=b"RAWBYTES!",
        valid_xor=False,
    )
    result = _vr3x_result(payload)
    assert result.status == "ok"
    assert result.values["integrity"]["status"] == "failed"
    assert result.values["integrity"]["computed"] != result.values["integrity"]["received"]
    assert result.values["summary"] == {
        "kind": "unknown",
        "raw_hex": b"RAWBYTES!".hex(),
        "selector": 0,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_packet()[:-1], "exactly 45"),
        (_packet() + b"\x00", "exactly 45"),
        (_packet(satellite_id=0x39), "satellite ID 0x39 is unknown"),
        (_packet(phase=7), "mission phase 7 is outside 0..6"),
    ],
    ids=["short", "long", "satellite-id", "phase"],
)
def test_vr3x_rejects_length_identity_and_enum_errors(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_vr3x(_context(payload))
    result = _vr3x_result(payload)
    assert result.status == "error"
    assert message in result.diagnostic


def test_vr3x_parser_is_stateless_and_registry_gates_framing_and_norad() -> None:
    first = parse_vr3x(_context(_packet(summary_selector=0x00))).values
    second = parse_vr3x(
        _context(
            _packet(satellite_id=0x3C, summary_selector=0xA8),
            norad_id=47524,
        )
    ).values
    assert first["satellite"]["name"] == "littlefoot"
    assert first["summary"]["kind"] == "unknown"
    assert second["satellite"]["name"] == "cera"
    assert second["summary"]["kind"] == "gps"
    assert all(
        result.parser != "vr3x"
        for result in DEFAULT_REGISTRY.parse(_context(_packet(), norad_id=51085))
    )
    assert all(
        result.parser != "vr3x"
        for result in DEFAULT_REGISTRY.parse(_context(_packet(), framing="AX.25"))
    )


def test_vr3x_sidecar_keeps_source_linkage(tmp_path: Path) -> None:
    payload = _packet()
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        json.dumps(
            {
                "framing": "LoRa",
                "integrity": "passed",
                "payload_hex": payload.hex(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    derive_preview(frames, sidecar, norad_id=47463, pass_framing="LoRa")

    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    record = next(item for item in records if item["parser"] == "vr3x")
    assert record["status"] == "ok"
    assert record["source_line"] == 1
    assert len(record["source_frame_id"]) == 64
    assert record["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["values"]["satellite"]["name"] == "littlefoot"


def test_vr3x_source_decoder_is_hash_pinned() -> None:
    artifact = next(
        item
        for item in load_manifest(_MANIFEST)
        if item.artifact_id == "tinygs-vr3x-decoder"
    )
    source = _TINYGS / artifact.source_path
    assert artifact.source_commit == "6b82a7f610349c2e46bcd97a0df38f9bdca1daf6"
    assert artifact.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()

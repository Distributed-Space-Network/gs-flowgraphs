"""NF-TLM-007 exact-gated VZLUSAT-2 telemetry-preview contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
from pathlib import Path

import pytest
from native_framing.provenance import load_manifest
from native_telemetry.output import derive_preview
from native_telemetry.profiles.vzlusat2 import (
    BEACON_PACKET_BYTES,
    MAX_PACKET_BYTES,
    parse_vzlusat2,
)
from native_telemetry.registry import DEFAULT_REGISTRY
from native_telemetry.types import FrameContext

_TEST_FILE = Path(__file__).resolve()
_GS_ROOT = _TEST_FILE.parents[2]
_MANIFEST = _TEST_FILE.parent / "fixtures" / "native_telemetry" / "MANIFEST.csv"
_GRSAT = _GS_ROOT / "related-projects" / "gr-satellites"
_TINYGS_DECODERS = _GS_ROOT / "related-projects" / "tinygs-decoders"

# Literal packets are intentionally not built through the production Construct schema.
# Both use CSP source=1, destination=26, destination_port=18, source_port=18.
_CSP_GATE = bytes.fromhex("83a49200")
_BEACON = bytes.fromhex(
    "83a49200"
    "56"
    "5a4c555341542d32"
    "01020304"
    "11121314"
    "21222324"
    "0ce4"
    "0102"
    "0304"
    "ffe7"
    "ff85"
    "31323334"
    "41424344"
)
_DROP = bytes.fromhex("83a49200037f01020304a0b0c0d0deadbeef")


def _context(
    payload: bytes,
    *,
    norad_id: int | None = 51085,
    framing: str = "AX100 ASM+Golay",
) -> FrameContext:
    return FrameContext(
        source_frame_id="b" * 64,
        source_line=7,
        norad_id=norad_id,
        framing=framing,
        payload=payload,
    )


def _vz_result(payload: bytes, **context: object):
    results = DEFAULT_REGISTRY.parse(_context(payload, **context))
    return next(result for result in results if result.parser == "vzlusat2")


def _tinygs_ksy_beacon_oracle(packet: bytes) -> dict[str, object]:
    """Independent literal translation of the pinned TinyGS KSY comparison schema."""

    csp = int.from_bytes(packet[:4], "big")
    assert ((csp >> 25) & 0x1F, (csp >> 20) & 0x1F) == (1, 26)
    assert ((csp >> 8) & 0x3F, (csp >> 14) & 0x3F) == (18, 18)
    assert packet[4] == 0x56
    assert packet[5:13] == b"ZLUSAT-2"
    values = struct.unpack(">IIIHHHhhII", packet[13:])
    return {
        "obc_timestamp": values[0],
        "obc_boot_count": values[1],
        "obc_reset_cause": values[2],
        "eps_vbatt": values[3],
        "eps_cursun": values[4],
        "eps_cursys": values[5],
        "eps_temp_bat": values[6],
        "radio_temp_pa": values[7] * 0.1,
        "radio_tot_tx_count": values[8],
        "radio_tot_rx_count": values[9],
    }


def test_vzlusat2_beacon_matches_literal_and_tinygs_comparison_schema() -> None:
    assert len(_BEACON) == BEACON_PACKET_BYTES
    result = _vz_result(_BEACON)

    assert result.status == "ok"
    assert result.values["kind"] == "beacon"
    assert result.values["command_hex"] == "0x56"
    assert result.values["mission_csp_gate"] is True
    assert result.values["csp"] == {
        "crc": False,
        "destination": 26,
        "destination_port": 18,
        "fragmentation": False,
        "hmac": False,
        "priority": 2,
        "rdp": False,
        "reserved": 0,
        "source": 1,
        "source_port": 18,
        "xtea": False,
    }
    assert result.values["telemetry"] == _tinygs_ksy_beacon_oracle(_BEACON)
    assert result.values["telemetry"]["eps_temp_bat"] == -25
    assert result.values["telemetry"]["radio_temp_pa"] == -12.3


def test_vzlusat2_drop_preserves_header_and_bounds_raw_data() -> None:
    result = _vz_result(_DROP)

    assert result.status == "ok"
    assert result.values["kind"] == "drop"
    assert result.values["telemetry"] == {
        "chunk": 0x01020304,
        "data_hex": "deadbeef",
        "data_length": 4,
        "data_truncated": False,
        "flag": 0x7F,
        "time": 0xA0B0C0D0,
    }

    large = _CSP_GATE + bytes.fromhex("03000000000100000002") + b"x" * 600
    bounded = _vz_result(large)
    assert bounded.values["telemetry"]["data_length"] == 600
    assert len(bounded.values["telemetry"]["data_hex"]) == 1_024
    assert bounded.values["telemetry"]["data_truncated"] is True


def test_vzlusat2_csp_and_unknown_command_default_to_bounded_raw_preview() -> None:
    unknown = _CSP_GATE + b"\x99" + b"RAW"
    result = _vz_result(unknown)
    assert result.status == "ok"
    assert result.values["kind"] == "raw"
    assert result.values["mission_csp_gate"] is True
    assert result.values["data_hex"] == b"RAW".hex()

    wrong_csp = bytearray(_BEACON)
    wrong_csp[0] ^= 0x06  # source 1 -> 2 without changing packet length or command.
    result = _vz_result(bytes(wrong_csp))
    assert result.status == "ok"
    assert result.values["kind"] == "raw"
    assert result.values["mission_csp_gate"] is False
    assert result.values["data_length"] == len(_BEACON) - 5


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\x00" * 4, "shorter than 5"),
        (_BEACON[:-1], "exactly 43"),
        (_BEACON + b"\x00", "exactly 43"),
        (_CSP_GATE + b"\x03" + b"\x00" * 8, "shorter than 14"),
        (b"\x00" * (MAX_PACKET_BYTES + 1), "exceeds 65536"),
    ],
    ids=[
        "packet-short",
        "beacon-truncated",
        "beacon-extended",
        "drop-truncated",
        "packet-oversized",
    ],
)
def test_vzlusat2_rejects_truncation_extension_and_oversize(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_vzlusat2(_context(payload))
    result = _vz_result(payload)
    assert result.status == "error"
    assert message in result.diagnostic


def test_vzlusat2_requires_exact_norad_and_canonical_framing() -> None:
    assert all(
        result.parser != "vzlusat2"
        for result in DEFAULT_REGISTRY.parse(_context(_BEACON, norad_id=51084))
    )
    assert all(
        result.parser != "vzlusat2"
        for result in DEFAULT_REGISTRY.parse(_context(_BEACON, framing="AX100 Mode 5"))
    )
    assert _vz_result(_BEACON, framing="ax100_asm_golay").status == "ok"


def test_vzlusat2_sidecar_keeps_source_frame_linkage(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        json.dumps(
            {
                "framing": "AX100 ASM+Golay",
                "integrity": "passed",
                "payload_hex": _BEACON.hex(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    derive_preview(
        frames,
        sidecar,
        norad_id=51085,
        pass_framing="AX100 ASM+Golay",
    )

    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    record = next(item for item in records if item["parser"] == "vzlusat2")
    assert record["status"] == "ok"
    assert record["source_line"] == 1
    assert len(record["source_frame_id"]) == 64
    assert record["payload_sha256"] == hashlib.sha256(_BEACON).hexdigest()
    assert record["values"]["kind"] == "beacon"


def test_telemetry_manifest_and_construct_pin_are_exact() -> None:
    artifacts = load_manifest(_MANIFEST)
    assert {
        "construct-wheel",
        "grsat-vzlusat2-schema",
        "tinygs-norby-schema",
        "tinygs-vzlusat2-schema",
    } <= {artifact.artifact_id for artifact in artifacts}
    assert importlib.metadata.version("construct") == "2.10.70"
    assert importlib.metadata.metadata("construct")["License"] == "MIT"

    roots = {
        "https://github.com/daniestevez/gr-satellites": _GRSAT,
        "https://github.com/4m1g0/tinygs-decoders": _TINYGS_DECODERS,
    }
    for artifact in artifacts:
        if artifact.artifact_id == "construct-wheel":
            assert artifact.source_commit == "c25a47172d4bde392b7ad188175b07b319d3dea4"
            assert artifact.sha256 == (
                "c80be81ef595a1a821ec69dc16099550ed22197615f4320b57cc9ce2a672cb30"
            )
            continue
        source = roots[artifact.source_repo] / artifact.source_path
        assert source.is_file(), artifact.artifact_id
        assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact.sha256

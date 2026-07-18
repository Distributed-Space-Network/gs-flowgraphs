from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from native_framing.provenance import load_manifest
from native_lora.profiles import (
    MAX_CATALOG_BYTES,
    LoRaCatalogSnapshot,
    LoRaProfileError,
    catalog_sha256,
    load_catalog_snapshot,
    normalize_sync_word,
    resolve_lora_profile,
)

_ROOT = Path(__file__).resolve().parents[2]
_TINYGS_COMMIT = "6dcbf47c45ac35bd3c2307113d12bdad42f415bd"


def _configuration(**over: object) -> dict[str, object]:
    config: dict[str, object] = {
        "mode": "LoRa",
        "freq": 437.7,
        "bw": 125.0,
        "sf": 7,
        "cr": 8,
        "sw": 0x12,
        "pl": 8,
        "crc": True,
        "fldro": 0,
        "iIQ": False,
        "len": 0,
    }
    config.update(over)
    return config


def _snapshot(
    satellites: list[object], *, retrieved_at: str = "2026-07-18T09:00:00Z"
) -> LoRaCatalogSnapshot:
    return LoRaCatalogSnapshot.from_mapping(
        {
            "schema_version": 1,
            "source_url": "https://api.tinygs.com/v3/satellites/?status=Supported",
            "source_version": "TinyGS-Webapp-index-BIJZkZ8e.js",
            "retrieved_at": retrieved_at,
            "catalog_sha256": catalog_sha256(satellites),
            "satellites": satellites,
        }
    )


def test_tinygs_profile_oracles_are_hash_pinned_and_exact() -> None:
    manifest_path = Path(__file__).parent / "fixtures" / "native_lora" / "MANIFEST.csv"
    artifacts = load_manifest(manifest_path)
    tinygs = [item for item in artifacts if item.source_commit == _TINYGS_COMMIT]
    assert len(artifacts) == 20
    assert {item.artifact_id for item in tinygs} == {
        "tinygs-modem-schema",
        "tinygs-lora-application",
        "tinygs-profile-parser",
    }
    assert {item.license for item in tinygs} == {"GPL-3.0-only"}
    checkout = _ROOT / "related-projects" / "tiny-gs" / "tinyGS"
    for artifact in tinygs:
        source = checkout / artifact.source_path
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact.sha256


def test_catalog_resolves_explicit_and_implicit_tinygs_profiles() -> None:
    satellites = [
        {
            "name": "EXAMPLE-LORA",
            "displayName": "Example LoRa",
            "norad": 99901,
            "configurations": [
                _configuration(),
                _configuration(
                    freq=436.5,
                    bw=62.5,
                    sf=12,
                    cr=7,
                    sw=0x1424,
                    pl=12,
                    crc=1,
                    fldro=2,
                    iIQ=1,
                    len=47,
                ),
            ],
        }
    ]
    snapshot = _snapshot(satellites)

    explicit = resolve_lora_profile(snapshot, 99901, configuration_index=0)
    assert explicit.satellite_name == "Example LoRa"
    assert (explicit.frequency_hz, explicit.bandwidth_hz) == (437_700_000, 125_000)
    assert (explicit.sf, explicit.cr, explicit.sync_word) == (7, 4, 0x12)
    assert explicit.explicit_header and explicit.payload_length is None
    assert explicit.has_crc and explicit.ldro is False and not explicit.inverted_iq
    assert explicit.phy_config().explicit_header
    modem = explicit.modem_config(sample_rate=250_000)
    assert (modem.sf, modem.bandwidth, modem.sync_word) == (7, 125_000, 0x12)

    implicit = resolve_lora_profile(snapshot, 99901, configuration_index=1)
    assert (implicit.frequency_hz, implicit.bandwidth_hz) == (436_500_000, 62_500)
    assert (implicit.sf, implicit.cr, implicit.sync_word) == (12, 3, 0x12)
    assert not implicit.explicit_header and implicit.payload_length == 47
    assert implicit.has_crc and implicit.ldro is None and implicit.inverted_iq
    phy = implicit.phy_config()
    assert not phy.explicit_header and phy.cr == 3 and phy.payload_length == 47
    assert phy.has_crc and phy.resolved_ldro


def test_backend_values_override_catalog_without_silent_retune() -> None:
    snapshot = _snapshot(
        [
            {
                "name": "OVERRIDE-ME",
                "NORAD": 99902,
                "configurations": [_configuration(len=23, fldro=1)],
            }
        ]
    )
    profile = resolve_lora_profile(
        snapshot,
        99902,
        overrides={
            "frequency_hz": 438_125_000,
            "bandwidth_hz": 250_000,
            "sf": 9,
            "cr": 5,
            "sync_word": "0x3444",
            "preamble_length": 16,
            "explicit_header": True,
            "has_crc": False,
            "ldro": None,
            "inverted_iq": True,
        },
    )
    assert (profile.frequency_hz, profile.bandwidth_hz) == (438_125_000, 250_000)
    assert (profile.sf, profile.cr, profile.sync_word) == (9, 1, 0x34)
    assert profile.preamble_length == 16
    assert profile.explicit_header and profile.payload_length is None
    assert not profile.has_crc and profile.ldro is None and profile.inverted_iq
    params = profile.waveform_parameters()
    assert params["catalog_sha256"] == snapshot.digest
    assert params["catalog_configuration_index"] == 0

    with pytest.raises(LoRaProfileError, match="unknown LoRa profile overrides"):
        resolve_lora_profile(snapshot, 99902, overrides={"frequency": 438.1})


def test_missing_ambiguous_stale_and_conflicting_profiles_fail_explicitly() -> None:
    satellites = [
        {
            "name": "AMBIGUOUS",
            "noradId": 99903,
            "configurations": [_configuration(), _configuration(freq=438.0)],
        }
    ]
    snapshot = _snapshot(satellites, retrieved_at="2026-01-01T00:00:00Z")
    with pytest.raises(LoRaProfileError, match="ambiguous"):
        resolve_lora_profile(snapshot, 99903)
    with pytest.raises(LoRaProfileError, match="no LoRa catalog profile"):
        resolve_lora_profile(snapshot, 12345)
    with pytest.raises(LoRaProfileError, match="stale"):
        resolve_lora_profile(
            snapshot,
            99903,
            configuration_index=0,
            now=datetime(2026, 7, 18, tzinfo=UTC),
            max_age=timedelta(days=30),
        )
    future = _snapshot(satellites, retrieved_at="2026-07-19T00:00:00Z")
    with pytest.raises(LoRaProfileError, match="future"):
        resolve_lora_profile(
            future,
            99903,
            configuration_index=0,
            now=datetime(2026, 7, 18, tzinfo=UTC),
            max_age=timedelta(days=30),
        )

    conflict = _snapshot(
        [
            {
                "name": "CONFLICT",
                "norad": 99904,
                "configurations": [_configuration(NORAD=99905)],
            }
        ]
    )
    with pytest.raises(LoRaProfileError, match="NORAD identities conflict"):
        resolve_lora_profile(conflict, 99904)


def test_snapshot_hash_schema_duplicate_keys_and_sync_words_are_fail_closed(
    tmp_path: Path,
) -> None:
    satellites = [
        {"name": "HASHED", "norad": 99906, "configurations": [_configuration()]}
    ]
    raw = {
        "schema_version": 1,
        "source_url": "https://api.tinygs.com/v3/satellites/?status=Supported",
        "source_version": "test-source",
        "retrieved_at": "2026-07-18T09:00:00Z",
        "catalog_sha256": "0" * 64,
        "satellites": satellites,
    }
    with pytest.raises(LoRaProfileError, match="catalog_sha256 mismatch"):
        LoRaCatalogSnapshot.from_mapping(raw)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"source_url":"https://example.test",'
        '"source_version":"x","retrieved_at":"2026-07-18T09:00:00Z",'
        '"catalog_sha256":"' + catalog_sha256(satellites) + '","satellites":[]}',
        encoding="utf-8",
    )
    with pytest.raises(LoRaProfileError, match="duplicate key"):
        load_catalog_snapshot(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_CATALOG_BYTES + 1))
    with pytest.raises(LoRaProfileError, match="must contain"):
        load_catalog_snapshot(oversized)

    assert normalize_sync_word(0x12) == 0x12
    assert normalize_sync_word(0x1424) == 0x12
    assert normalize_sync_word(0x3444) == 0x34
    with pytest.raises(LoRaProfileError, match="SX127x octet or SX126x"):
        normalize_sync_word(0x1234)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("freq", 0, "frequency_hz"),
        ("bw", 0, "bandwidth_hz"),
        ("sf", 13, "sf"),
        ("cr", 9, "coding rate"),
        ("sw", 0x1234, "sync word"),
        ("pl", 4, "preamble_length"),
        ("fldro", 3, "fldro"),
        ("len", 256, "len"),
    ],
)
def test_invalid_catalog_radio_parameters_fail_before_use(
    field: str, value: object, message: str
) -> None:
    snapshot = _snapshot(
        [
            {
                "name": "INVALID",
                "norad": 99907,
                "configurations": [_configuration(**{field: value})],
            }
        ]
    )
    with pytest.raises(LoRaProfileError, match=message):
        resolve_lora_profile(snapshot, 99907)

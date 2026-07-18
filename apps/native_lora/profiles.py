"""Fail-closed per-mission LoRa profile resolution for NF-FRM-030.

SPDX-License-Identifier: GPL-3.0-only

The field semantics are adapted from TinyGS firmware at commit
6dcbf47c45ac35bd3c2307113d12bdad42f415bd (GPL-3.0-only).  TinyGS applies
remote catalog configurations; it does not ship a static per-bird catalog.
Consequently this module contains no mission defaults.  It accepts only a
versioned, hash-verified snapshot supplied by the operator, rejects ambiguous
NORAD/configuration matches, and lets explicit backend fields override the
selected snapshot record.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from .framing import MAX_SF, MIN_SF, LoRaPhyConfig
from .modem import LoRaModemConfig

CATALOG_SCHEMA_VERSION: Final = 1
MAX_CATALOG_BYTES: Final = 2 * 1024 * 1024
MAX_SATELLITES: Final = 4096
MAX_CONFIGURATIONS_PER_SATELLITE: Final = 32
MIN_FREQUENCY_HZ: Final = 100_000_000
MAX_FREQUENCY_HZ: Final = 2_500_000_000
MIN_BANDWIDTH_HZ: Final = 1_000
MAX_BANDWIDTH_HZ: Final = 1_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MISSING = object()


class LoRaProfileError(ValueError):
    """A catalog, selector, or override violates the NF-FRM-030 contract."""


def _canonical_catalog_json(satellites: Sequence[object]) -> str:
    return json.dumps(
        satellites,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def catalog_sha256(satellites: Sequence[object]) -> str:
    """Hash the canonical JSON representation embedded in a snapshot."""

    return hashlib.sha256(_canonical_catalog_json(satellites).encode("utf-8")).hexdigest()


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise LoRaProfileError("retrieved_at must be a non-empty ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LoRaProfileError("retrieved_at must be valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LoRaProfileError("retrieved_at must include a timezone")
    return parsed.astimezone(UTC)


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LoRaProfileError(f"catalog JSON contains duplicate key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class LoRaCatalogSnapshot:
    """Immutable, attributable wrapper around one retrieved TinyGS response."""

    source_url: str
    source_version: str
    retrieved_at: datetime
    digest: str
    _catalog_json: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> LoRaCatalogSnapshot:
        if raw.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise LoRaProfileError(
                f"schema_version must be exactly {CATALOG_SCHEMA_VERSION}"
            )
        source_url = raw.get("source_url")
        if not isinstance(source_url, str) or not source_url.startswith("https://"):
            raise LoRaProfileError("source_url must be an HTTPS URL")
        source_version = raw.get("source_version")
        if (
            not isinstance(source_version, str)
            or not source_version.strip()
            or len(source_version) > 160
        ):
            raise LoRaProfileError("source_version must be a bounded non-empty string")
        digest = raw.get("catalog_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise LoRaProfileError("catalog_sha256 must be 64 lowercase hex characters")
        satellites = raw.get("satellites")
        if not isinstance(satellites, list):
            raise LoRaProfileError("satellites must be a JSON array")
        if not 1 <= len(satellites) <= MAX_SATELLITES:
            raise LoRaProfileError(f"satellites must contain 1..{MAX_SATELLITES} entries")
        for satellite in satellites:
            if not isinstance(satellite, dict):
                raise LoRaProfileError("each satellite must be a JSON object")
            configurations = satellite.get("configurations")
            if not isinstance(configurations, list):
                raise LoRaProfileError("each satellite must contain configurations")
            if len(configurations) > MAX_CONFIGURATIONS_PER_SATELLITE:
                raise LoRaProfileError(
                    "satellite configuration count exceeds the bounded maximum"
                )
            if not all(isinstance(item, dict) for item in configurations):
                raise LoRaProfileError("each configuration must be a JSON object")
        catalog_json = _canonical_catalog_json(satellites)
        actual = hashlib.sha256(catalog_json.encode("utf-8")).hexdigest()
        if actual != digest:
            raise LoRaProfileError(
                f"catalog_sha256 mismatch: expected {digest}, computed {actual}"
            )
        return cls(
            source_url=source_url,
            source_version=source_version.strip(),
            retrieved_at=_parse_timestamp(raw.get("retrieved_at")),
            digest=digest,
            _catalog_json=catalog_json,
        )

    @property
    def satellites(self) -> tuple[Mapping[str, object], ...]:
        # Decode a fresh copy so callers cannot mutate the frozen snapshot.
        decoded = json.loads(self._catalog_json)
        return tuple(decoded)

    def require_fresh(
        self,
        *,
        now: datetime,
        max_age: timedelta,
        future_tolerance: timedelta = timedelta(minutes=5),
    ) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise LoRaProfileError("freshness reference time must include a timezone")
        if max_age <= timedelta(0):
            raise LoRaProfileError("maximum catalog age must be positive")
        reference = now.astimezone(UTC)
        if self.retrieved_at > reference + future_tolerance:
            raise LoRaProfileError("catalog retrieval time is in the future")
        if reference - self.retrieved_at > max_age:
            raise LoRaProfileError("catalog snapshot is stale")


def load_catalog_snapshot(path: str | Path) -> LoRaCatalogSnapshot:
    """Load a bounded JSON snapshot without accepting duplicate object keys."""

    snapshot_path = Path(path)
    size = snapshot_path.stat().st_size
    if not 1 <= size <= MAX_CATALOG_BYTES:
        raise LoRaProfileError(
            f"catalog snapshot must contain 1..{MAX_CATALOG_BYTES} bytes"
        )
    try:
        raw = json.loads(
            snapshot_path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LoRaProfileError(f"catalog JSON constant {value!r} is not permitted")
            ),
        )
    except UnicodeDecodeError as exc:
        raise LoRaProfileError("catalog snapshot must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LoRaProfileError("catalog snapshot must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise LoRaProfileError("catalog snapshot root must be an object")
    return LoRaCatalogSnapshot.from_mapping(raw)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise LoRaProfileError(f"{label} must be an integer, not boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 0)
        except ValueError as exc:
            raise LoRaProfileError(f"{label} must be an integer") from exc
    raise LoRaProfileError(f"{label} must be an integer")


def _scaled_integer(value: object, scale: int, label: str) -> int:
    if isinstance(value, bool):
        raise LoRaProfileError(f"{label} must be numeric, not boolean")
    try:
        scaled = Decimal(str(value)) * scale
    except (InvalidOperation, ValueError) as exc:
        raise LoRaProfileError(f"{label} must be finite numeric data") from exc
    if not scaled.is_finite() or scaled != scaled.to_integral_value():
        raise LoRaProfileError(f"{label} does not resolve to an integral Hz value")
    return int(scaled)


def _boolean(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise LoRaProfileError(f"{label} must be boolean or 0/1")


def _alias(
    mapping: Mapping[str, object],
    names: tuple[str, ...],
    parser,
    label: str,
    *,
    default: object = _MISSING,
):
    found = [
        (name, parser(mapping[name], label))
        for name in names
        if mapping.get(name) is not None
    ]
    if not found:
        if default is _MISSING:
            raise LoRaProfileError(f"missing {label}")
        return default
    first = found[0][1]
    if any(value != first for _, value in found[1:]):
        aliases = ", ".join(name for name, _ in found)
        raise LoRaProfileError(f"conflicting {label} aliases: {aliases}")
    return first


def _norad(satellite: Mapping[str, object], config: Mapping[str, object]) -> int:
    values: list[int] = []
    for mapping in (satellite, config):
        for key in ("norad", "NORAD", "noradId"):
            if mapping.get(key) is not None:
                values.append(_integer(mapping[key], "NORAD"))
    if not values:
        raise LoRaProfileError("satellite configuration has no NORAD identity")
    if any(value != values[0] for value in values[1:]):
        raise LoRaProfileError("satellite/configuration NORAD identities conflict")
    if not 1 <= values[0] <= 999_999:
        raise LoRaProfileError("NORAD must be in 1..999999")
    return values[0]


def normalize_sync_word(value: object) -> int:
    """Normalize SX127x octets and SX126x ``0x?4?4`` register words."""

    word = _integer(value, "sync word")
    if 0 <= word <= 0xFF:
        return word
    if 0 <= word <= 0xFFFF and (word & 0x0F0F) == 0x0404:
        return ((word >> 8) & 0xF0) | ((word >> 4) & 0x0F)
    raise LoRaProfileError(
        "sync word must be an SX127x octet or SX126x 0x?4?4 register word"
    )


def _native_cr(value: object, label: str = "coding rate") -> int:
    cr = _integer(value, label)
    if 1 <= cr <= 4:
        return cr
    if 5 <= cr <= 8:
        return cr - 4
    raise LoRaProfileError("coding rate must be native 1..4 or TinyGS/RadioLib 5..8")


def _ldro(value: object, label: str = "LDRO") -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    mode = _integer(value, label)
    if mode == 2:
        return None
    if mode in (0, 1):
        return bool(mode)
    raise LoRaProfileError("TinyGS fldro must be 0 (off), 1 (on), or 2 (auto)")


@dataclass(frozen=True)
class LoRaMissionProfile:
    norad_id: int
    satellite_name: str
    configuration_index: int
    frequency_hz: int
    bandwidth_hz: int
    sf: int
    cr: int
    sync_word: int
    preamble_length: int
    explicit_header: bool
    payload_length: int | None
    has_crc: bool
    ldro: bool | None
    inverted_iq: bool
    catalog_sha256: str
    source_version: str

    def __post_init__(self) -> None:
        if not 1 <= self.norad_id <= 999_999:
            raise LoRaProfileError("NORAD must be in 1..999999")
        if not self.satellite_name or len(self.satellite_name) > 160:
            raise LoRaProfileError("satellite name must be bounded and non-empty")
        if not MIN_FREQUENCY_HZ <= self.frequency_hz <= MAX_FREQUENCY_HZ:
            raise LoRaProfileError(
                f"frequency_hz must be in {MIN_FREQUENCY_HZ}..{MAX_FREQUENCY_HZ}"
            )
        if not MIN_BANDWIDTH_HZ <= self.bandwidth_hz <= MAX_BANDWIDTH_HZ:
            raise LoRaProfileError(
                f"bandwidth_hz must be in {MIN_BANDWIDTH_HZ}..{MAX_BANDWIDTH_HZ}"
            )
        if not MIN_SF <= self.sf <= MAX_SF:
            raise LoRaProfileError(f"sf must be in {MIN_SF}..{MAX_SF}")
        if self.cr not in range(1, 5):
            raise LoRaProfileError("cr must be in 1..4")
        if not 0 <= self.sync_word <= 0xFF:
            raise LoRaProfileError("sync_word must be an octet")
        if not 5 <= self.preamble_length <= 65_535:
            raise LoRaProfileError("preamble_length must be in 5..65535")
        if self.explicit_header:
            if self.payload_length is not None:
                raise LoRaProfileError("explicit-header profile cannot fix payload_length")
            if self.sf < 7:
                raise LoRaProfileError("explicit-header LoRa requires sf >= 7")
        elif self.payload_length is None or not 1 <= self.payload_length <= 255:
            raise LoRaProfileError("implicit-header payload_length must be in 1..255")
        if not isinstance(self.has_crc, bool):
            raise LoRaProfileError("has_crc must be boolean")
        if self.ldro is not None and not isinstance(self.ldro, bool):
            raise LoRaProfileError("ldro must be boolean or automatic")
        if not isinstance(self.inverted_iq, bool):
            raise LoRaProfileError("inverted_iq must be boolean")

    def modem_config(self, *, sample_rate: int, **quality: object) -> LoRaModemConfig:
        return LoRaModemConfig(
            sf=self.sf,
            bandwidth=self.bandwidth_hz,
            sample_rate=sample_rate,
            sync_word=self.sync_word,
            preamble_length=self.preamble_length,
            inverted_iq=self.inverted_iq,
            **quality,
        )

    def phy_config(self) -> LoRaPhyConfig:
        if self.explicit_header:
            return LoRaPhyConfig(
                sf=self.sf,
                bandwidth=self.bandwidth_hz,
                explicit_header=True,
                ldro=self.ldro,
            )
        return LoRaPhyConfig(
            sf=self.sf,
            bandwidth=self.bandwidth_hz,
            explicit_header=False,
            cr=self.cr,
            payload_length=self.payload_length,
            has_crc=self.has_crc,
            ldro=self.ldro,
        )

    def waveform_parameters(self) -> dict[str, object]:
        parameters: dict[str, object] = {
            "norad_id": self.norad_id,
            "frequency_hz": self.frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "sf": self.sf,
            "cr": self.cr,
            "sync_word": self.sync_word,
            "preamble_length": self.preamble_length,
            "explicit_header": self.explicit_header,
            "has_crc": self.has_crc,
            "inverted_iq": self.inverted_iq,
            "catalog_sha256": self.catalog_sha256,
            "catalog_source_version": self.source_version,
            "catalog_configuration_index": self.configuration_index,
        }
        if self.payload_length is not None:
            parameters["payload_length"] = self.payload_length
        if self.ldro is not None:
            parameters["ldro"] = self.ldro
        return parameters


_OVERRIDE_KEYS = frozenset(
    {
        "frequency_hz",
        "bandwidth_hz",
        "sf",
        "cr",
        "sync_word",
        "preamble_length",
        "explicit_header",
        "payload_length",
        "has_crc",
        "ldro",
        "inverted_iq",
    }
)


def _catalog_profile_values(config: Mapping[str, object]) -> dict[str, object]:
    frequency_hz = _scaled_integer(config.get("freq"), 1_000_000, "TinyGS freq MHz")
    bandwidth_hz = _scaled_integer(config.get("bw"), 1_000, "TinyGS bw kHz")
    length = _alias(config, ("len",), _integer, "TinyGS payload length")
    if not 0 <= length <= 255:
        raise LoRaProfileError("TinyGS len must be in 0..255")
    return {
        "frequency_hz": frequency_hz,
        "bandwidth_hz": bandwidth_hz,
        "sf": _alias(config, ("sf",), _integer, "TinyGS sf"),
        "cr": _alias(config, ("cr",), _native_cr, "TinyGS coding rate"),
        "sync_word": _alias(
            config,
            ("sw",),
            lambda value, _label: normalize_sync_word(value),
            "TinyGS sync word",
        ),
        "preamble_length": _alias(
            config, ("pl",), _integer, "TinyGS preamble length"
        ),
        "explicit_header": length == 0,
        "payload_length": None if length == 0 else length,
        "has_crc": _alias(config, ("crc",), _boolean, "TinyGS payload CRC"),
        "ldro": _alias(config, ("fldro",), _ldro, "TinyGS LDRO"),
        "inverted_iq": _alias(config, ("iIQ",), _boolean, "TinyGS IQ inversion"),
    }


def _apply_overrides(values: dict[str, object], overrides: Mapping[str, object]) -> None:
    unknown = sorted(set(overrides) - _OVERRIDE_KEYS)
    if unknown:
        raise LoRaProfileError(f"unknown LoRa profile overrides: {', '.join(unknown)}")
    parsers = {
        "frequency_hz": lambda value, _label: _integer(value, "frequency_hz"),
        "bandwidth_hz": lambda value, _label: _integer(value, "bandwidth_hz"),
        "sf": lambda value, _label: _integer(value, "sf"),
        "cr": _native_cr,
        "sync_word": lambda value, _label: normalize_sync_word(value),
        "preamble_length": lambda value, _label: _integer(value, "preamble_length"),
        "explicit_header": _boolean,
        "payload_length": lambda value, _label: (
            None if value is None else _integer(value, "payload_length")
        ),
        "has_crc": _boolean,
        "ldro": _ldro,
        "inverted_iq": _boolean,
    }
    for key, raw in overrides.items():
        values[key] = parsers[key](raw, key)
    if overrides.get("explicit_header") is True and "payload_length" not in overrides:
        values["payload_length"] = None
    if (
        "payload_length" in overrides
        and overrides["payload_length"] is not None
        and "explicit_header" not in overrides
    ):
        values["explicit_header"] = False


def resolve_lora_profile(
    snapshot: LoRaCatalogSnapshot,
    norad_id: int,
    *,
    configuration_index: int | None = None,
    overrides: Mapping[str, object] | None = None,
    now: datetime | None = None,
    max_age: timedelta | None = None,
) -> LoRaMissionProfile:
    """Resolve exactly one LoRa configuration for ``norad_id``.

    Multiple LoRa configurations are deliberately ambiguous unless the caller
    supplies their original catalog index.  ``overrides`` are canonical backend
    values and take precedence only after a unique catalog record is selected.
    """

    target = _integer(norad_id, "NORAD")
    if not 1 <= target <= 999_999:
        raise LoRaProfileError("NORAD must be in 1..999999")
    if (now is None) != (max_age is None):
        raise LoRaProfileError("now and max_age must be provided together")
    if now is not None and max_age is not None:
        snapshot.require_fresh(now=now, max_age=max_age)
    if configuration_index is not None:
        configuration_index = _integer(configuration_index, "configuration_index")
        if configuration_index < 0:
            raise LoRaProfileError("configuration_index must be non-negative")

    matches: list[tuple[str, int, Mapping[str, object]]] = []
    for satellite in snapshot.satellites:
        name_value = satellite.get("displayName") or satellite.get("name")
        if not isinstance(name_value, str) or not name_value.strip():
            raise LoRaProfileError("satellite name/displayName must be non-empty")
        configurations = satellite["configurations"]
        assert isinstance(configurations, list)
        for index, config in enumerate(configurations):
            assert isinstance(config, dict)
            try:
                config_norad = _norad(satellite, config)
            except LoRaProfileError as exc:
                if "has no NORAD" in str(exc):
                    continue
                raise
            mode = config.get("mode")
            if config_norad != target or not isinstance(mode, str) or mode.casefold() != "lora":
                continue
            if configuration_index is None or configuration_index == index:
                matches.append((name_value.strip(), index, config))
    if not matches:
        suffix = (
            "" if configuration_index is None else f" at configuration index {configuration_index}"
        )
        raise LoRaProfileError(f"no LoRa catalog profile for NORAD {target}{suffix}")
    if len(matches) != 1:
        indexes = ", ".join(f"{name}[{index}]" for name, index, _ in matches)
        raise LoRaProfileError(
            f"ambiguous LoRa catalog profiles for NORAD {target}: {indexes}"
        )

    name, index, config = matches[0]
    values = _catalog_profile_values(config)
    _apply_overrides(values, overrides or {})
    return LoRaMissionProfile(
        norad_id=target,
        satellite_name=name,
        configuration_index=index,
        catalog_sha256=snapshot.digest,
        source_version=snapshot.source_version,
        **values,
    )


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "LoRaCatalogSnapshot",
    "LoRaMissionProfile",
    "LoRaProfileError",
    "catalog_sha256",
    "load_catalog_snapshot",
    "normalize_sync_word",
    "resolve_lora_profile",
]

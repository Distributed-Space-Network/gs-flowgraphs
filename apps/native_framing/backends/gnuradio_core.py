"""Fail-closed GNU Radio core capability probe for native framing adapters.

This module deliberately constructs no scheduler blocks.  It pins the minimum
Python API contract that a future adapter must satisfy and makes missing/old or
partial installations observable before a flowgraph is spawned.  Local codec
implementations remain authoritative until vector, chunk/reset, and scheduler
boundary parity has been proven for a selected adapter.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Final

MIN_GNURADIO_VERSION: Final = (3, 10, 0)
FEATURE_PATHS: Final = (
    ("hard_correlation", "gnuradio.digital", "correlate_access_code_tag_bb"),
    ("soft_correlation", "gnuradio.digital", "correlate_access_code_ff_ts"),
    ("differential_decode", "gnuradio.digital", "diff_decoder_bb"),
    ("multiplicative_descramble", "gnuradio.digital", "additive_scrambler_bb"),
    ("fractional_symbol_timing", "gnuradio.digital", "symbol_sync_ff"),
    ("convolutional_decode", "gnuradio.fec", "code.cc_decoder.make"),
)
_FEATURE_NAMES: Final = frozenset(name for name, _, _ in FEATURE_PATHS)


class GnuradioCoreUnavailable(RuntimeError):
    """The requested GNU Radio adapter contract is not executable."""


@dataclass(frozen=True)
class GnuradioCoreCapabilities:
    installed: bool
    version: str | None
    version_tuple: tuple[int, int, int] | None
    features: tuple[tuple[str, bool], ...]
    reason: str

    @property
    def feature_map(self) -> dict[str, bool]:
        return dict(self.features)

    @property
    def missing_features(self) -> tuple[str, ...]:
        return tuple(name for name, available in self.features if not available)

    @property
    def version_supported(self) -> bool:
        return self.version_tuple is not None and self.version_tuple >= MIN_GNURADIO_VERSION

    @property
    def available(self) -> bool:
        return self.installed and self.version_supported and not self.missing_features


def _version_tuple(value: object) -> tuple[int, int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", str(value))
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def _resolve_attribute(module: object, path: str) -> bool:
    current = module
    for component in path.split("."):
        if not hasattr(current, component):
            return False
        current = getattr(current, component)
    return callable(current)


def probe_gnuradio_core(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> GnuradioCoreCapabilities:
    """Inspect the pinned API surface without constructing GNU Radio blocks."""

    try:
        package = importer("gnuradio")
        gr = importer("gnuradio.gr")
    except (ImportError, OSError) as exc:
        features = tuple((name, False) for name, _, _ in FEATURE_PATHS)
        return GnuradioCoreCapabilities(
            installed=False,
            version=None,
            version_tuple=None,
            features=features,
            reason=f"GNU Radio import unavailable: {type(exc).__name__}",
        )

    version_value: object | None = None
    version_function = getattr(gr, "version", None)
    if callable(version_function):
        try:
            version_value = version_function()
        except Exception as exc:  # pragma: no cover - defensive external API boundary
            reason = f"GNU Radio version probe failed: {type(exc).__name__}"
            features = tuple((name, False) for name, _, _ in FEATURE_PATHS)
            return GnuradioCoreCapabilities(True, None, None, features, reason)
    if version_value is None:
        version_value = getattr(package, "__version__", None)
    version = str(version_value) if version_value is not None else None
    parsed_version = _version_tuple(version_value) if version_value is not None else None

    modules: dict[str, ModuleType | None] = {}
    feature_results: list[tuple[str, bool]] = []
    for name, module_name, attribute in FEATURE_PATHS:
        if module_name not in modules:
            try:
                modules[module_name] = importer(module_name)
            except (ImportError, OSError):
                modules[module_name] = None
        module = modules[module_name]
        feature_results.append(
            (name, module is not None and _resolve_attribute(module, attribute))
        )

    features = tuple(feature_results)
    missing = tuple(name for name, available in features if not available)
    if parsed_version is None:
        reason = "GNU Radio version is missing or unparsable"
    elif parsed_version < MIN_GNURADIO_VERSION:
        minimum = ".".join(str(part) for part in MIN_GNURADIO_VERSION)
        reason = f"GNU Radio {version} is older than required {minimum}"
    elif missing:
        reason = "GNU Radio API is missing: " + ", ".join(missing)
    else:
        reason = "GNU Radio core capability contract is available"
    return GnuradioCoreCapabilities(
        installed=True,
        version=version,
        version_tuple=parsed_version,
        features=features,
        reason=reason,
    )


def require_gnuradio_core(
    capabilities: GnuradioCoreCapabilities,
    required: Iterable[str] = _FEATURE_NAMES,
) -> None:
    """Reject unavailable or undeclared adapter features deterministically."""

    names = tuple(required)
    unknown = sorted(set(names) - _FEATURE_NAMES)
    if unknown:
        raise ValueError("unknown GNU Radio capabilities: " + ", ".join(unknown))
    if not capabilities.installed or not capabilities.version_supported:
        raise GnuradioCoreUnavailable(capabilities.reason)
    feature_map = capabilities.feature_map
    missing = sorted(name for name in names if not feature_map.get(name, False))
    if missing:
        raise GnuradioCoreUnavailable(
            "GNU Radio API is missing: " + ", ".join(missing)
        )

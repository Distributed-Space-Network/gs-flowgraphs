"""Typed contracts shared by every native framing profile.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np


class SymbolInput(str, Enum):
    HARD_BITS = "hard_bits"
    SOFT_SYMBOLS = "soft_symbols"


class Polarity(str, Enum):
    NORMAL = "normal"
    INVERTED = "inverted"
    AMBIGUOUS = "ambiguous"


class IntegrityStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_PRESENT = "not_present"
    NOT_CHECKED = "not_checked"


class DecodeDisposition(str, Enum):
    """Truthful status of an advertised profile in the native migration."""

    EXISTING_PENDING_PARITY = "existing_pending_parity"
    IN_PROGRESS = "in_progress"
    PLANNED = "planned"
    NATIVE = "native"


@dataclass(frozen=True)
class FrameResult:
    canonical_framing: str
    payload: bytes
    integrity: IntegrityStatus
    source_start: int
    source_end: int
    polarity: Polarity
    sync_distance: float | None = None
    corrected_symbols: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_framing.strip():
            raise ValueError("canonical_framing must not be empty")
        if self.source_start < 0 or self.source_end < self.source_start:
            raise ValueError("invalid source offset interval")
        if self.sync_distance is not None and self.sync_distance < 0:
            raise ValueError("sync_distance must be non-negative")
        if self.corrected_symbols is not None and self.corrected_symbols < 0:
            raise ValueError("corrected_symbols must be non-negative")
        object.__setattr__(self, "payload", bytes(self.payload))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class StreamingDecoder(Protocol):
    """Bounded decoder contract used identically by live and replay paths."""

    @property
    def retained_symbols(self) -> int: ...

    @property
    def max_retained_symbols(self) -> int: ...

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]: ...

    def flush(self) -> list[FrameResult]: ...


DecoderFactory = Callable[[Mapping[str, object]], StreamingDecoder]
EncoderFactory = Callable[[Mapping[str, object]], object]


@dataclass(frozen=True)
class ParameterSpec:
    value_type: type
    required: bool = False
    default: object | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[object, ...] = ()

    def validate(self, name: str, value: object) -> object:
        # bool is an int subclass; accepting it for numeric protocol parameters
        # makes configuration mistakes surprisingly hard to spot.
        if self.value_type in (int, float) and isinstance(value, bool):
            raise ValueError(f"{name} must be {self.value_type.__name__}, not bool")
        if not isinstance(value, self.value_type):
            raise ValueError(f"{name} must be {self.value_type.__name__}")
        if self.minimum is not None and float(value) < self.minimum:
            raise ValueError(f"{name} must be >= {self.minimum}")
        if self.maximum is not None and float(value) > self.maximum:
            raise ValueError(f"{name} must be <= {self.maximum}")
        if self.choices and value not in self.choices:
            raise ValueError(f"{name} must be one of {self.choices!r}")
        return value


@dataclass(frozen=True)
class FramingProfile:
    canonical: str
    advertised_label: str
    aliases: tuple[str, ...]
    disposition: DecodeDisposition
    symbol_input: SymbolInput
    accepted_polarities: tuple[Polarity, ...]
    max_retained_symbols: int
    sync_policy: str
    frame_length_policy: str
    transforms: tuple[str, ...]
    integrity_policy: str
    output_semantics: str
    live_supported: bool
    post_pass_supported: bool
    parameters: Mapping[str, ParameterSpec] = field(default_factory=dict)
    decoder_factory: DecoderFactory | None = None
    encoder_factory: EncoderFactory | None = None

    def __post_init__(self) -> None:
        canonical = self.canonical.strip().lower()
        if not canonical or canonical != self.canonical:
            raise ValueError("canonical token must be non-empty, stripped, and lowercase")
        if not self.advertised_label.strip():
            raise ValueError("advertised_label must not be empty")
        aliases = tuple(alias.strip() for alias in self.aliases)
        if any(not alias for alias in aliases):
            raise ValueError("profile aliases must not be empty")
        folded = [alias.casefold() for alias in aliases]
        if len(folded) != len(set(folded)):
            raise ValueError(f"duplicate aliases inside profile {canonical}")
        if self.max_retained_symbols <= 0:
            raise ValueError("max_retained_symbols must be positive")
        if not self.accepted_polarities:
            raise ValueError("at least one accepted polarity is required")
        if len(set(self.accepted_polarities)) != len(self.accepted_polarities):
            raise ValueError("accepted polarities must be unique")
        required_text = (
            self.sync_policy,
            self.frame_length_policy,
            self.integrity_policy,
            self.output_semantics,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("profile policies must not be empty")
        if self.disposition is DecodeDisposition.NATIVE and self.decoder_factory is None:
            raise ValueError("a native profile requires a decoder factory")
        if self.disposition is DecodeDisposition.PLANNED and self.decoder_factory is not None:
            raise ValueError("a planned profile cannot expose a decoder")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def decoder_available(self) -> bool:
        return self.decoder_factory is not None

    @property
    def encoder_available(self) -> bool:
        return self.encoder_factory is not None

    def validate_parameters(
        self, supplied: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        values = dict(supplied or {})
        unknown = sorted(set(values) - set(self.parameters))
        if unknown:
            raise ValueError(f"unknown parameters for {self.canonical}: {', '.join(unknown)}")
        validated: dict[str, object] = {}
        for name, spec in self.parameters.items():
            if name in values:
                validated[name] = spec.validate(name, values[name])
            elif spec.required and spec.default is None:
                raise ValueError(f"missing required parameter: {name}")
            elif spec.default is not None:
                validated[name] = spec.validate(name, spec.default)
        return validated

"""Exact-gated registry for untrusted payload-preview parsers."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from native_framing.registry import resolve_profile

from native_telemetry.profiles.grizu import FRAMING as GRIZU_FRAMING
from native_telemetry.profiles.grizu import NORAD_ID as GRIZU_NORAD_ID
from native_telemetry.profiles.grizu import parse_grizu
from native_telemetry.profiles.norby import FRAMING as NORBY_FRAMING
from native_telemetry.profiles.norby import NORAD_ID as NORBY_NORAD_ID
from native_telemetry.profiles.norby import parse_norby
from native_telemetry.profiles.strings import extract_strings
from native_telemetry.profiles.vr3x import FRAMING as VR3X_FRAMING
from native_telemetry.profiles.vr3x import NORAD_IDS as VR3X_NORAD_IDS
from native_telemetry.profiles.vr3x import parse_vr3x
from native_telemetry.profiles.vzlusat2 import FRAMING as VZLUSAT2_FRAMING
from native_telemetry.profiles.vzlusat2 import NORAD_ID as VZLUSAT2_NORAD_ID
from native_telemetry.profiles.vzlusat2 import parse_vzlusat2
from native_telemetry.types import FrameContext, ParserResult, ParserSpec

MAX_JSON_DEPTH = 12
MAX_JSON_NODES = 8_192
MAX_DIAGNOSTIC_CHARS = 256
_BIDI_CONTROLS = frozenset(
    {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI", "BN"}
)


def canonical_framing(label: object) -> str:
    text = str(label or "").strip()
    profile = resolve_profile(text)
    return profile.canonical if profile is not None else text


def safe_diagnostic(value: object) -> str:
    """Render untrusted exception text without terminal/control or bidi effects."""

    output: list[str] = []
    for char in str(value):
        if unicodedata.category(char).startswith("C") or (
            unicodedata.bidirectional(char) in _BIDI_CONTROLS
        ):
            codepoint = ord(char)
            output.append(f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}")
        else:
            output.append(char)
        if sum(len(part) for part in output) >= MAX_DIAGNOSTIC_CHARS:
            break
    return "".join(output)[:MAX_DIAGNOSTIC_CHARS]


def _json_safe(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Copy parser output into deterministic JSON primitives with hard complexity limits."""

    if budget is None:
        budget = [MAX_JSON_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_JSON_DEPTH:
        raise ValueError("parser output exceeds JSON complexity limits")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("parser output contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("parser output mapping keys must be strings")
        return {
            key: _json_safe(item, depth=depth + 1, budget=budget)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_json_safe(item, depth=depth + 1, budget=budget) for item in value]
    raise TypeError(f"parser output contains unsupported type {type(value).__name__}")


class ParserRegistry:
    def __init__(self, specs: Sequence[ParserSpec]) -> None:
        names: set[str] = set()
        checked: list[ParserSpec] = []
        for spec in specs:
            if not spec.name or spec.name in names:
                raise ValueError("parser names must be non-empty and unique")
            if (
                isinstance(spec.version, bool)
                or not isinstance(spec.version, int)
                or spec.version <= 0
            ):
                raise ValueError(f"parser {spec.name!r} has invalid version")
            if not spec.generic and (not spec.norad_ids or not spec.framings):
                raise ValueError(
                    f"mission parser {spec.name!r} requires exact NORAD and framing gates"
                )
            if spec.generic and (spec.norad_ids or spec.framings):
                raise ValueError(f"generic parser {spec.name!r} cannot declare mission gates")
            if any(norad <= 0 for norad in spec.norad_ids):
                raise ValueError(f"parser {spec.name!r} has invalid NORAD gate")
            if not spec.generic:
                normalized_framings = frozenset(
                    canonical_framing(framing) for framing in spec.framings
                )
                if "" in normalized_framings:
                    raise ValueError(f"parser {spec.name!r} has an empty framing gate")
                spec = replace(spec, framings=normalized_framings)
            names.add(spec.name)
            checked.append(spec)
        self._specs = tuple(sorted(checked, key=lambda item: item.name))

    @property
    def specs(self) -> tuple[ParserSpec, ...]:
        return self._specs

    def matching(self, context: FrameContext) -> tuple[ParserSpec, ...]:
        framing = canonical_framing(context.framing)
        return tuple(
            spec
            for spec in self._specs
            if spec.generic
            or (
                context.norad_id is not None
                and context.norad_id in spec.norad_ids
                and framing in spec.framings
            )
        )

    def parse(self, context: FrameContext) -> tuple[ParserResult, ...]:
        results: list[ParserResult] = []
        for spec in self.matching(context):
            try:
                preview = spec.parser(context)
                if preview.status not in {"ok", "no_preview"}:
                    raise ValueError(f"parser returned invalid status {preview.status!r}")
                values = _json_safe(preview.values)
                results.append(
                    ParserResult(
                        parser=spec.name,
                        parser_version=spec.version,
                        status=preview.status,
                        values=values,
                    )
                )
            except Exception as exc:
                detail = safe_diagnostic(f"{type(exc).__name__}: {exc}")
                results.append(
                    ParserResult(
                        parser=spec.name,
                        parser_version=spec.version,
                        status="error",
                        diagnostic=detail,
                    )
                )
        return tuple(results)


DEFAULT_REGISTRY = ParserRegistry(
    [
        ParserSpec(name="strings", version=1, parser=extract_strings, generic=True),
        ParserSpec(
            name="grizu263a",
            version=1,
            parser=parse_grizu,
            norad_ids=frozenset({GRIZU_NORAD_ID}),
            framings=frozenset({GRIZU_FRAMING}),
        ),
        ParserSpec(
            name="norby",
            version=1,
            parser=parse_norby,
            norad_ids=frozenset({NORBY_NORAD_ID}),
            framings=frozenset({NORBY_FRAMING}),
        ),
        ParserSpec(
            name="vzlusat2",
            version=1,
            parser=parse_vzlusat2,
            norad_ids=frozenset({VZLUSAT2_NORAD_ID}),
            framings=frozenset({VZLUSAT2_FRAMING}),
        ),
        ParserSpec(
            name="vr3x",
            version=1,
            parser=parse_vr3x,
            norad_ids=VR3X_NORAD_IDS,
            framings=frozenset({VR3X_FRAMING}),
        ),
    ]
)

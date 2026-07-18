"""Typed contracts shared by telemetry-preview parsers and output."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

PreviewStatus = Literal["ok", "no_preview"]


@dataclass(frozen=True)
class FrameContext:
    source_frame_id: str
    source_line: int
    norad_id: int | None
    framing: str
    payload: bytes


@dataclass(frozen=True)
class ParserPreview:
    status: PreviewStatus
    values: Mapping[str, Any]


Parser = Callable[[FrameContext], ParserPreview]


@dataclass(frozen=True)
class ParserSpec:
    name: str
    version: int
    parser: Parser
    generic: bool = False
    norad_ids: frozenset[int] = frozenset()
    framings: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ParserResult:
    parser: str
    parser_version: int
    status: Literal["ok", "no_preview", "error"]
    values: Mapping[str, Any] | None = None
    diagnostic: str = ""

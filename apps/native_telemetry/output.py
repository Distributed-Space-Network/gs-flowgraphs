"""Immutable frames.jsonl to atomic, bounded telemetry-preview sidecar."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from native_telemetry.profiles.strings import MAX_INPUT_BYTES
from native_telemetry.registry import (
    DEFAULT_REGISTRY,
    ParserRegistry,
    canonical_framing,
    safe_diagnostic,
)
from native_telemetry.types import FrameContext

SCHEMA = "gs.telemetry-preview"
SCHEMA_VERSION = 1
MAX_FRAMES_FILE_BYTES = 64 * 1024 * 1024
MAX_FRAME_LINES = 100_000
MAX_SIDECAR_BYTES = 16 * 1024 * 1024
MAX_DIAGNOSTIC_CHARS = 256
_PAYLOAD_FIELDS = ("payload_hex", "deframed_hex", "info_hex")
_HEX_RE = re.compile(r"[0-9a-fA-F]*\Z")


@dataclass(frozen=True)
class PreviewSummary:
    source_frames: int
    preview_records: int
    diagnostics: int
    output_bytes: int
    truncated: bool


def _source_id(line_number: int, raw_line: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(str(line_number).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw_line)
    return digest.hexdigest()


def _record_bytes(record: dict[str, Any]) -> bytes:
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _input_diagnostic(
    *, source_frame_id: str, source_line: int, diagnostic: str
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_frame_id": source_frame_id,
        "source_line": source_line,
        "parser": "input",
        "parser_version": 1,
        "status": "error",
        "diagnostic": safe_diagnostic(diagnostic),
    }


def _truncation_diagnostic(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_frame_id": None,
        "source_line": None,
        "parser": "output",
        "parser_version": 1,
        "status": "error",
        "diagnostic": safe_diagnostic(reason),
    }


def _payload_hex(record: dict[str, Any]) -> tuple[str, str]:
    for field in _PAYLOAD_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            return field, value
    raise ValueError("frame has no payload_hex/deframed_hex/info_hex string")


def _accepted(record: dict[str, Any]) -> bool:
    if record.get("crc_ok") is False:
        return False
    return str(record.get("integrity", "")).strip().lower() not in {
        "failed",
        "rejected",
        "uncorrectable",
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def derive_preview(
    frames_path: Path,
    output_path: Path,
    *,
    norad_id: int | None = None,
    pass_framing: str = "",
    registry: ParserRegistry = DEFAULT_REGISTRY,
) -> PreviewSummary:
    """Derive an operator-only sidecar without ever opening ``frames_path`` for write."""

    frames_path = Path(frames_path)
    output_path = Path(output_path)
    if frames_path.resolve() == output_path.resolve():
        raise ValueError("telemetry preview output must differ from frames.jsonl")
    if norad_id is not None and (
        isinstance(norad_id, bool) or not isinstance(norad_id, int) or norad_id <= 0
    ):
        raise ValueError("norad_id must be a positive integer or None")
    if frames_path.stat().st_size > MAX_FRAMES_FILE_BYTES:
        raise ValueError("frames.jsonl exceeds the telemetry-preview input limit")

    source = frames_path.read_bytes()
    if len(source) > MAX_FRAMES_FILE_BYTES:
        raise ValueError("frames.jsonl exceeds the telemetry-preview input limit")
    source_hash = hashlib.sha256(source).digest()
    lines = source.splitlines()
    output = bytearray()
    preview_records = 0
    diagnostics = 0
    source_frames = 0
    truncated = len(lines) > MAX_FRAME_LINES
    output_full = False
    truncation_reason = (
        f"input contains more than {MAX_FRAME_LINES} frame lines" if truncated else ""
    )
    truncation_reserve = len(
        _record_bytes(_truncation_diagnostic("x" * MAX_DIAGNOSTIC_CHARS))
    )

    def append(record: dict[str, Any]) -> bool:
        nonlocal diagnostics, output_full, preview_records, truncated, truncation_reason
        encoded = _record_bytes(record)
        if len(output) + len(encoded) > MAX_SIDECAR_BYTES - truncation_reserve:
            truncated = True
            output_full = True
            truncation_reason = f"sidecar exceeds {MAX_SIDECAR_BYTES} output bytes"
            return False
        output.extend(encoded)
        preview_records += 1
        if record.get("status") == "error":
            diagnostics += 1
        return True

    for line_number, raw_line in enumerate(lines[:MAX_FRAME_LINES], start=1):
        source_frame_id = _source_id(line_number, raw_line)
        try:
            loaded = json.loads(raw_line)
            if not isinstance(loaded, dict):
                raise TypeError("frame record must be a JSON object")
            if not _accepted(loaded):
                continue
            payload_field, encoded_payload = _payload_hex(loaded)
            if len(encoded_payload) > 2 * MAX_INPUT_BYTES:
                raise ValueError(f"{payload_field} exceeds {MAX_INPUT_BYTES} decoded bytes")
            if len(encoded_payload) % 2 or _HEX_RE.fullmatch(encoded_payload) is None:
                raise ValueError(f"{payload_field} must be canonical hexadecimal bytes")
            payload = bytes.fromhex(encoded_payload)
            framing = canonical_framing(loaded.get("framing") or pass_framing)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if not append(
                _input_diagnostic(
                    source_frame_id=source_frame_id,
                    source_line=line_number,
                    diagnostic=f"{type(exc).__name__}: {exc}",
                )
            ):
                break
            continue

        source_frames += 1
        context = FrameContext(
            source_frame_id=source_frame_id,
            source_line=line_number,
            norad_id=norad_id,
            framing=framing,
            payload=payload,
        )
        common = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_frame_id": source_frame_id,
            "source_line": line_number,
            "norad_id": norad_id,
            "framing": framing,
            "payload_field": payload_field,
            "payload_length": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "operator_preview": True,
            "authoritative": False,
        }
        for result in registry.parse(context):
            record = {**common, **asdict(result)}
            record = {key: value for key, value in record.items() if value not in (None, "")}
            if not append(record):
                break
        if output_full:
            break

    if truncated:
        output.extend(
            _record_bytes(_truncation_diagnostic(truncation_reason or "output truncated"))
        )
        preview_records += 1
        diagnostics += 1

    if hashlib.sha256(frames_path.read_bytes()).digest() != source_hash:
        raise RuntimeError("frames.jsonl changed while telemetry preview was being derived")
    _atomic_write(output_path, bytes(output))
    return PreviewSummary(
        source_frames=source_frames,
        preview_records=preview_records,
        diagnostics=diagnostics,
        output_bytes=len(output),
        truncated=truncated,
    )

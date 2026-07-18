"""Validation for the external framing evidence manifest."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")

EVIDENCE_CLASSES = frozenset(
    {
        "normative_vector",
        "upstream_oracle",
        "independent_oracle",
        "hardware_packet_engine",
        "real_capture",
        "payload_only_parser",
    }
)

REQUIRED_COLUMNS = (
    "artifact_id",
    "source_repo",
    "source_commit",
    "source_path",
    "sha256",
    "license",
    "evidence_class",
    "expected_output",
)


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    source_repo: str
    source_commit: str
    source_path: str
    sha256: str
    license: str
    evidence_class: str
    expected_output: str


def validate_manifest_rows(rows: Iterable[dict[str, str]]) -> tuple[EvidenceArtifact, ...]:
    artifacts: list[EvidenceArtifact] = []
    ids: set[str] = set()
    locations: set[tuple[str, str, str]] = set()
    for line_number, raw in enumerate(rows, start=2):
        missing = [name for name in REQUIRED_COLUMNS if not str(raw.get(name, "")).strip()]
        if missing:
            raise ValueError(f"manifest line {line_number} has empty fields: {', '.join(missing)}")
        values = {name: str(raw[name]).strip() for name in REQUIRED_COLUMNS}
        if not _HEX_40.fullmatch(values["source_commit"]):
            raise ValueError(f"manifest line {line_number} has an invalid source commit")
        if not _HEX_64.fullmatch(values["sha256"]):
            raise ValueError(f"manifest line {line_number} has an invalid SHA-256")
        if values["evidence_class"] not in EVIDENCE_CLASSES:
            raise ValueError(f"manifest line {line_number} has an unknown evidence class")
        if values["artifact_id"] in ids:
            raise ValueError(f"duplicate artifact_id: {values['artifact_id']}")
        location = (
            values["source_repo"],
            values["source_commit"],
            values["source_path"],
        )
        if location in locations:
            raise ValueError(f"duplicate source artifact: {values['source_path']}")
        ids.add(values["artifact_id"])
        locations.add(location)
        artifacts.append(EvidenceArtifact(**values))
    if not artifacts:
        raise ValueError("evidence manifest must not be empty")
    return tuple(artifacts)


def load_manifest(path: str | Path) -> tuple[EvidenceArtifact, ...]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(
                f"manifest columns must be exactly {', '.join(REQUIRED_COLUMNS)}"
            )
        return validate_manifest_rows(reader)


__all__ = [
    "EVIDENCE_CLASSES",
    "EvidenceArtifact",
    "REQUIRED_COLUMNS",
    "load_manifest",
    "validate_manifest_rows",
]

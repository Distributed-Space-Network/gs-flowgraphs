"""NF-TLM-001/002 telemetry-preview contract and bounded string extraction."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import native_telemetry.output as preview_output
import native_telemetry.profiles.strings as strings_profile
import pytest
from native_telemetry.output import derive_preview
from native_telemetry.registry import DEFAULT_REGISTRY, ParserRegistry
from native_telemetry.types import FrameContext, ParserPreview, ParserSpec
from telemetry_preview import main

from gfsk_ax25.ax25 import encode_ui


def _context(
    payload: bytes,
    *,
    norad_id: int | None = None,
    framing: str = "AX.25",
) -> FrameContext:
    return FrameContext(
        source_frame_id="a" * 64,
        source_line=1,
        norad_id=norad_id,
        framing=framing,
        payload=payload,
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_strings_have_exact_utf8_offsets_and_exclude_controls_and_bidi() -> None:
    payload = (
        b"\x00HELLO world\x1bBAD\x00"
        + "Привіт".encode()
        + "\u202eHIDDEN".encode()
        + b"\xffTAIL"
    )
    preview = strings_profile.extract_strings(_context(payload))

    assert preview.status == "ok"
    runs = preview.values["strings"]
    assert [run["text"] for run in runs] == ["HELLO world", "Привіт", "HIDDEN", "TAIL"]
    assert runs[0]["byte_start"] == 1
    assert payload[runs[1]["byte_start"] : runs[1]["byte_end"]].decode() == "Привіт"
    assert "\x1b" not in preview.values["rendered"]
    assert "\u202e" not in preview.values["rendered"]


def test_strings_enforce_count_per_string_and_rendered_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strings_profile, "MAX_STRINGS", 2)
    monkeypatch.setattr(strings_profile, "MAX_STRING_CHARS", 5)
    monkeypatch.setattr(strings_profile, "MAX_RENDERED_BYTES", 9)
    preview = strings_profile.extract_strings(_context(b"ABCDEFGHIJ\x00KLMNOP\x00QRSTUV"))

    runs = preview.values["strings"]
    assert len(runs) == 2
    assert runs[0] == {"byte_start": 0, "byte_end": 10, "text": "ABCDE", "truncated": True}
    assert runs[1]["text"] == "KLMN"
    assert len(preview.values["rendered"].encode("utf-8")) <= 9


def test_strings_bound_input_and_binary_noise_without_false_runs() -> None:
    oversized = strings_profile.extract_strings(
        _context(b"A" * (strings_profile.MAX_INPUT_BYTES + 10))
    )
    assert oversized.values["input_truncated"] is True
    assert oversized.values["strings"][0]["byte_end"] == strings_profile.MAX_INPUT_BYTES

    binary = strings_profile.extract_strings(_context(b"A\x00" * 10_000))
    assert binary.status == "no_preview"
    assert binary.values["strings"] == []


def test_registry_requires_exact_norad_and_canonical_framing_gates() -> None:
    def generic(_context: FrameContext) -> ParserPreview:
        return ParserPreview(status="ok", values={"kind": "generic"})

    def mission(_context: FrameContext) -> ParserPreview:
        return ParserPreview(status="ok", values={"kind": "mission"})

    registry = ParserRegistry(
        [
            ParserSpec(name="generic", version=1, parser=generic, generic=True),
            ParserSpec(
                name="mission",
                version=2,
                parser=mission,
                norad_ids=frozenset({25544}),
                framings=frozenset({"AX.25"}),
            ),
        ]
    )
    assert [r.parser for r in registry.parse(_context(b"DATA"))] == ["generic"]
    assert [r.parser for r in registry.parse(_context(b"DATA", norad_id=1))] == ["generic"]
    assert [
        r.parser for r in registry.parse(_context(b"DATA", norad_id=25544, framing="GEOSCAN"))
    ] == ["generic"]
    assert [r.parser for r in registry.parse(_context(b"DATA", norad_id=25544))] == [
        "generic",
        "mission",
    ]


def test_registry_contains_parser_exceptions_and_non_json_values() -> None:
    def raises(_context: FrameContext) -> ParserPreview:
        raise IndexError("short\npacket\u202e")

    def nonfinite(_context: FrameContext) -> ParserPreview:
        return ParserPreview(status="ok", values={"temperature": float("nan")})

    registry = ParserRegistry(
        [
            ParserSpec(name="nonfinite", version=1, parser=nonfinite, generic=True),
            ParserSpec(name="raises", version=1, parser=raises, generic=True),
        ]
    )
    results = registry.parse(_context(b"DATA"))
    assert [result.status for result in results] == ["error", "error"]
    assert results[0].diagnostic == "ValueError: parser output contains a non-finite number"
    assert results[1].diagnostic == r"IndexError: short\u000apacket\u202e"


def test_exact_gated_mission_parsers_contain_deterministic_malformed_fuzz() -> None:
    rng = random.Random(0x7E1E)
    gates = (
        (51085, "AX100 ASM+Golay", lambda data: data),
        (51025, "Grizu-263A", lambda data: data),
        (
            46494,
            "AX.25 G3RUH",
            lambda data: encode_ui(dest="NORBI", src="FUZZ", info=data),
        ),
    )
    for _ in range(300):
        raw = rng.randbytes(rng.randrange(0, 300))
        for norad_id, framing, envelope in gates:
            results = DEFAULT_REGISTRY.parse(
                _context(envelope(raw), norad_id=norad_id, framing=framing)
            )
            mission = [result for result in results if result.parser != "strings"]
            assert len(mission) == 1
            assert mission[0].status in {"ok", "no_preview", "error"}
            json.dumps(
                {
                    "status": mission[0].status,
                    "values": mission[0].values,
                    "diagnostic": mission[0].diagnostic,
                },
                allow_nan=False,
                sort_keys=True,
            )


@pytest.mark.parametrize(
    "spec",
    [
        ParserSpec(
            name="", version=1, parser=lambda context: ParserPreview("ok", {}), generic=True
        ),
        ParserSpec(
            name="bad", version=0, parser=lambda context: ParserPreview("ok", {}), generic=True
        ),
        ParserSpec(name="bad", version=1, parser=lambda context: ParserPreview("ok", {})),
        ParserSpec(
            name="bad",
            version=1,
            parser=lambda context: ParserPreview("ok", {}),
            generic=True,
            norad_ids=frozenset({1}),
        ),
    ],
)
def test_registry_configuration_fails_closed(spec: ParserSpec) -> None:
    with pytest.raises(ValueError):
        ParserRegistry([spec])


def test_sidecar_is_deterministic_source_linked_and_does_not_mutate_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    lines = [
        {"framing": "AX.25", "crc_ok": True, "payload_hex": b"CALL hello".hex()},
        {"framing": "AX.25", "crc_ok": True, "payload_hex": b"CALL hello".hex()},
        {"framing": "GEOSCAN", "integrity": "passed", "deframed_hex": b"GEO data".hex()},
        {"framing": "AX.25", "crc_ok": False, "payload_hex": b"REJECTED".hex()},
    ]
    frames.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    before = frames.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()

    first = derive_preview(frames, sidecar, norad_id=25544)
    first_output = sidecar.read_bytes()
    second = derive_preview(frames, sidecar, norad_id=25544)

    assert first == second
    assert sidecar.read_bytes() == first_output
    assert frames.read_bytes() == before
    assert hashlib.sha256(frames.read_bytes()).hexdigest() == before_hash
    records = _records(sidecar)
    assert len(records) == 3
    assert records[0]["source_frame_id"] != records[1]["source_frame_id"]
    assert records[0]["operator_preview"] is True
    assert records[0]["authoritative"] is False
    assert records[0]["norad_id"] == 25544
    assert records[2]["payload_field"] == "deframed_hex"
    assert all(record["parser"] == "strings" for record in records)


def test_missing_norad_runs_generic_only_and_pass_framing_is_canonicalized(tmp_path: Path) -> None:
    calls: list[str] = []

    def generic(_context: FrameContext) -> ParserPreview:
        calls.append("generic")
        return ParserPreview("ok", {"text": "generic"})

    def mission(_context: FrameContext) -> ParserPreview:
        calls.append("mission")
        return ParserPreview("ok", {"text": "mission"})

    registry = ParserRegistry(
        [
            ParserSpec("generic", 1, generic, generic=True),
            ParserSpec(
                "mission",
                1,
                mission,
                norad_ids=frozenset({25544}),
                framings=frozenset({"AX.25 G3RUH"}),
            ),
        ]
    )
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(json.dumps({"payload_hex": b"HELLO".hex()}) + "\n")

    derive_preview(frames, sidecar, pass_framing="AX.25 G3RUH", registry=registry)
    assert calls == ["generic"]
    assert _records(sidecar)[0]["framing"] == "ax25_g3ruh"


def test_bad_records_and_oversized_payloads_become_bounded_diagnostics(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        "not-json\n"
        + json.dumps({"payload_hex": "0x12"})
        + "\n"
        + json.dumps({"payload_hex": "41" * (strings_profile.MAX_INPUT_BYTES + 1)})
        + "\n",
        encoding="utf-8",
    )
    before = frames.read_bytes()

    summary = derive_preview(frames, sidecar)

    assert frames.read_bytes() == before
    assert summary.diagnostics == 3
    assert summary.output_bytes <= preview_output.MAX_SIDECAR_BYTES
    records = _records(sidecar)
    assert all(record["status"] == "error" for record in records)
    assert all(
        len(record["diagnostic"]) <= preview_output.MAX_DIAGNOSTIC_CHARS
        for record in records
    )


def test_parser_exception_is_a_source_linked_diagnostic_and_source_stays_immutable(
    tmp_path: Path,
) -> None:
    def broken(_context: FrameContext) -> ParserPreview:
        raise RuntimeError("vendor parser failed")

    registry = ParserRegistry(
        [ParserSpec(name="broken", version=7, parser=broken, generic=True)]
    )
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(json.dumps({"payload_hex": b"PAYLOAD".hex()}) + "\n")
    before = frames.read_bytes()

    derive_preview(frames, sidecar, registry=registry)

    assert frames.read_bytes() == before
    record = _records(sidecar)[0]
    assert record["parser"] == "broken"
    assert record["parser_version"] == 7
    assert record["status"] == "error"
    assert record["source_frame_id"]
    assert record["diagnostic"] == "RuntimeError: vendor parser failed"


def test_frame_line_limit_emits_versioned_terminal_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        "".join(json.dumps({"payload_hex": b"HELLO".hex()}) + "\n" for _ in range(3))
    )
    monkeypatch.setattr(preview_output, "MAX_FRAME_LINES", 2)

    summary = derive_preview(frames, sidecar)

    assert summary.truncated is True
    records = _records(sidecar)
    assert records[-1]["parser"] == "output"
    assert records[-1]["schema_version"] == preview_output.SCHEMA_VERSION
    assert "more than 2 frame lines" in records[-1]["diagnostic"]


def test_sidecar_output_limit_is_explicit_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames.jsonl"
    sidecar = tmp_path / "telemetry_preview.jsonl"
    frames.write_text(
        "".join(
            json.dumps({"payload_hex": (b"A" * 300).hex()}) + "\n" for _ in range(20)
        ),
        encoding="utf-8",
    )
    sidecar.write_bytes(b"old partial output")
    monkeypatch.setattr(preview_output, "MAX_SIDECAR_BYTES", 1_200)

    summary = derive_preview(frames, sidecar)

    assert summary.truncated is True
    assert summary.output_bytes <= 1_200
    records = _records(sidecar)
    assert records[-1]["parser"] == "output"
    assert records[-1]["status"] == "error"
    assert b"old partial output" not in sidecar.read_bytes()


def test_cli_writes_default_sidecar_and_rejects_invalid_norad(tmp_path: Path) -> None:
    frames = tmp_path / "frames.jsonl"
    frames.write_text(json.dumps({"payload_hex": b"HELLO".hex()}) + "\n")

    assert main(["--frames", str(frames), "--norad-id", "25544", "--framing", "AX.25"]) == 0
    assert (tmp_path / "telemetry_preview.jsonl").exists()
    with pytest.raises(SystemExit):
        main(["--frames", str(frames), "--norad-id", "0"])

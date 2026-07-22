"""Phase-0 executable contracts for the native framing migration."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import framings
import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.provenance import EVIDENCE_CLASSES, load_manifest, validate_manifest_rows
from native_framing.registry import REGISTRY, ProfileRegistry, advertised_profiles
from native_framing.types import DecodeDisposition, FrameResult, IntegrityStatus, Polarity

from gfsk_ax25 import ax25, gfsk
from gfsk_ax25 import fcs as ax25_fcs
from gfsk_ax25 import framing as ax25_framing

_TEST_FILE = Path(__file__).resolve()
_FIXTURES = _TEST_FILE.parent / "fixtures" / "native_framing"
_LEDGER = _TEST_FILE.parents[2] / "docs" / "native_grsat_framing_traceability_ledger.csv"
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 9_600.0
_AFSK_SYMBOL_RATE = 1_200.0

# TAPR AX.25 v2.2 section 3 address/control/PID rules and section 4.4 FCS,
# serialized LSB-first with HDLC stuffing and NRZI (0 -> transition). Keeping
# this as a literal prevents the production encoder from serving as its own
# receive oracle.
_TAPR_UI_BODY = bytes.fromhex("82a0a4a64040609c60868298986103f03e74657374")
_TAPR_UI_FCS = bytes.fromhex("0bed")
_TAPR_UI_NRZI = np.fromiter(
    (
        bit == "1"
        for bit in (
            "00000001001010110101001101101100111011001010100101010110101011101000010010101110"
            "11101011001010110100010010111011101011100010101010100000111111010100111100110111"
            "00010000101100001110010100111000011111110"
        )
    ),
    dtype=np.uint8,
)


def _body(payload: bytes = b"native-contract") -> bytes:
    return ax25.encode_ui(dest="DSN0", src="NATIVE", info=payload)


def _decode_in_steps(label: str, bits: np.ndarray, step: int) -> list[FrameResult]:
    decoder = build_decoder(label)
    output: list[FrameResult] = []
    for start in range(0, bits.size, step):
        output.extend(decoder.push(bits[start : start + step]))
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    output.extend(decoder.flush())
    assert decoder.retained_symbols == 0
    return output


def _rectangular_fsk_capture(bits: np.ndarray, *, mod_index: float = 0.8) -> np.ndarray:
    samples_per_symbol = int(_IQ_SAMPLE_RATE / _IQ_SYMBOL_RATE)
    symbols = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    frequency = (
        np.repeat(symbols, samples_per_symbol) * mod_index * _IQ_SYMBOL_RATE / 2.0
    )
    phase = 2.0 * np.pi * np.cumsum(frequency) / _IQ_SAMPLE_RATE
    burst = np.exp(1j * phase).astype(np.complex64)
    guard = np.zeros(2_000, dtype=np.complex64)
    return np.concatenate((guard, burst, guard))


def _fm_afsk_capture(
    bits: np.ndarray,
    *,
    one_hz: float = 1_200.0,
    zero_hz: float = 2_200.0,
    deviation_hz: float = 3_000.0,
) -> np.ndarray:
    samples_per_symbol = int(_IQ_SAMPLE_RATE / _AFSK_SYMBOL_RATE)
    audio_parts: list[np.ndarray] = []
    tone_phase = 0.31
    for bit in np.asarray(bits, dtype=np.uint8):
        frequency = one_hz if bit else zero_hz
        phase_step = 2.0 * np.pi * frequency / _IQ_SAMPLE_RATE
        phase = tone_phase + phase_step * np.arange(samples_per_symbol)
        audio_parts.append(np.sin(phase))
        tone_phase = float(phase[-1] + phase_step)
    audio = np.concatenate(audio_parts)
    rf_phase = 2.0 * np.pi * deviation_hz * np.cumsum(audio) / _IQ_SAMPLE_RATE
    burst = np.exp(1j * rf_phase).astype(np.complex64)
    guard = np.ones(2_400, dtype=np.complex64)
    return np.concatenate((guard, burst, np.full(2_400, burst[-1], dtype=np.complex64)))


def test_all_advertised_labels_have_one_registry_and_ledger_disposition():
    profiles = advertised_profiles()
    assert tuple(profiles) == framings.GRSATELLITES_FRAMINGS
    assert len(profiles) == len(set(profiles)) == 28
    assert all(profile.disposition in DecodeDisposition for profile in profiles.values())

    with _LEDGER.open(newline="", encoding="utf-8") as stream:
        framing_rows = [
            row
            for row in csv.DictReader(stream)
            if row["label"] in framings.GRSATELLITES_FRAMINGS
        ]
    labels = [row["label"] for row in framing_rows]
    assert labels == list(framings.GRSATELLITES_FRAMINGS)
    assert len(labels) == len(set(labels)) == 28


def test_registry_rejects_cross_profile_alias_collisions():
    original = REGISTRY.resolve("AX.25")
    assert original is not None
    conflicting = replace(
        original,
        canonical="conflicting",
        advertised_label="Conflicting",
        aliases=("AX25",),
    )
    with pytest.raises(ValueError, match="shared"):
        ProfileRegistry((original, conflicting))


def test_no_advertised_profile_remains_planned_without_a_decoder():
    assert all(
        profile.disposition is not DecodeDisposition.PLANNED
        and profile.decoder_available
        for profile in REGISTRY.profiles
    )


def test_profile_parameter_validation_is_fail_closed():
    with pytest.raises(ValueError, match="unknown parameters"):
        build_decoder("AX.25", {"mystery": 1})
    with pytest.raises(ValueError, match=">= 18"):
        build_decoder("AX.25", {"max_frame_bytes": 17})
    with pytest.raises(ValueError, match="not bool"):
        build_decoder("AX.25", {"max_frame_bytes": True})


@pytest.mark.parametrize("step", [1, 2, 7, 8, 17, 31, 64, 257])
@pytest.mark.parametrize("inverted", [False, True])
def test_ax25_g3ruh_streaming_is_chunk_and_polarity_invariant(step: int, inverted: bool):
    expected = _body()
    bits = ax25_framing.encode(expected, scramble=True, nrzi=True)
    if inverted:
        bits = 1 - bits
    frames = _decode_in_steps("AX.25 G3RUH", bits, step)
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.integrity is IntegrityStatus.PASSED
    # NRZI carries transitions, not absolute levels, so a whole-stream
    # inversion is intentionally reported as ambiguous after both hypotheses
    # validate rather than as a fabricated polarity decision.
    assert frame.polarity is Polarity.AMBIGUOUS
    assert set(frame.metadata["polarity_hypotheses"]) == {"normal", "inverted"}
    assert frame.metadata["address_policy_ok"] is True
    assert frame.metadata["g3ruh"] is True
    assert 0 <= frame.source_start < frame.source_end <= bits.size


def test_ax25_g3ruh_production_gfsk_file_iq_replay(tmp_path: Path) -> None:
    expected = _body(b"production-gfsk-postpass")
    bits = ax25_framing.encode(expected, scramble=True, nrzi=True)
    burst = gfsk.modulate(
        bits,
        gfsk.GfskParams(
            sample_rate_hz=_IQ_SAMPLE_RATE,
            symbol_rate_hz=_IQ_SYMBOL_RATE,
            mod_index=0.5,
            bt=0.5,
        ),
    )
    capture = np.concatenate(
        (
            np.zeros(2_000, dtype=np.complex64),
            np.asarray(burst, dtype=np.complex64),
            np.zeros(2_000, dtype=np.complex64),
        )
    )
    path = tmp_path / "ax25-g3ruh-production-gfsk.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("AX.25 G3RUH",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="gfsk",
        mod_index=0.5,
        bt=0.5,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "ax25_g3ruh"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


def test_ax25_plain_fsk_file_iq_replay_routes_production_profile(tmp_path: Path) -> None:
    expected = _body(b"plain-fsk-postpass")
    bits = ax25_framing.encode(expected, scramble=False, nrzi=True)
    path = tmp_path / "ax25-plain-fsk.cf32"
    _rectangular_fsk_capture(bits).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("AX.25",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="fsk",
        mod_index=0.8,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "ax25"
    assert records[0]["integrity"] == "passed"
    assert records[0]["metadata"]["g3ruh"] is False


def test_ax25_plain_afsk_file_iq_replay_routes_bell202_profile(tmp_path: Path) -> None:
    expected = _body(b"plain-bell202-postpass")
    bits = ax25_framing.encode(expected, scramble=False, nrzi=True)
    path = tmp_path / "ax25-plain-afsk.cf32"
    _fm_afsk_capture(bits).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_AFSK_SYMBOL_RATE,
        framings_to_try=("AX.25",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        framing_parameters={
            "modulation": "afsk",
            "tones_hz": [1_200.0, 2_200.0],
        },
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "ax25"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["metadata"]["g3ruh"] is False


@pytest.mark.parametrize("scramble", [False, True])
def test_ax25_family_profile_accepts_declared_line_coding(scramble: bool):
    expected = _body(b"plain-or-g3ruh")
    bits = ax25_framing.encode(expected, scramble=scramble, nrzi=True)
    frames = _decode_in_steps("AX.25", bits, 13)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["g3ruh"] is scramble


def test_tapr_ax25_ui_fcs_and_nrzi_literal_vector_decodes():
    assert ax25_fcs.fcs_bytes(_TAPR_UI_BODY) == _TAPR_UI_FCS
    parsed = ax25.decode_ui(_TAPR_UI_BODY)
    assert parsed == ax25.Ui(dest="APRS", src="N0CALL", info=b">test")
    frames = _decode_in_steps("AX.25", _TAPR_UI_NRZI, 11)
    assert [frame.payload for frame in frames] == [_TAPR_UI_BODY]
    assert frames[0].integrity is IntegrityStatus.PASSED


def test_ax25_literal_vector_fcs_mutation_is_rejected():
    damaged = _TAPR_UI_NRZI.copy()
    damaged[100] ^= 1
    assert _decode_in_steps("AX.25", damaged, 19) == []


def test_ax25_max_frame_bytes_is_enforced_for_single_chunk_valid_frames():
    exact = ax25.encode_ui(dest="APRS", src="N0CALL", info=b"")
    oversized = ax25.encode_ui(dest="APRS", src="N0CALL", info=b"x")
    exact_bits = ax25_framing.encode(exact, scramble=False, nrzi=True)
    oversized_bits = ax25_framing.encode(oversized, scramble=False, nrzi=True)
    decoder = build_decoder("AX.25", {"max_frame_bytes": 18})
    assert [frame.payload for frame in decoder.push(exact_bits)] == [exact]
    assert decoder.push(oversized_bits) == []


def test_ax25_stream_state_stays_bounded_on_unclosed_and_random_input():
    decoder = build_decoder("AX.25", {"max_frame_bytes": 18})
    rng = np.random.default_rng(0xA25)
    for _ in range(200):
        decoder.push(rng.integers(0, 2, size=4096, dtype=np.uint8))
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    decoder.flush()
    assert decoder.retained_symbols == 0


def test_stream_input_validation_rejects_soft_or_malformed_chunks():
    decoder = build_decoder("AX.25")
    with pytest.raises(ValueError, match="only 0 and 1"):
        decoder.push(np.array([0.0, 0.25, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="one-dimensional"):
        decoder.push(np.zeros((2, 2), dtype=np.uint8))


def test_frame_metadata_is_immutable_and_offsets_are_validated():
    frame = FrameResult(
        canonical_framing="test",
        payload=b"payload",
        integrity=IntegrityStatus.NOT_PRESENT,
        source_start=10,
        source_end=20,
        polarity=Polarity.NORMAL,
        metadata={"key": "value"},
    )
    with pytest.raises(TypeError):
        frame.metadata["key"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="offset"):
        replace(frame, source_end=9)


def test_encoder_is_absent_until_independently_declared():
    assert all(not profile.encoder_available for profile in REGISTRY.profiles)


def test_pinned_evidence_manifest_is_complete_and_classified():
    artifacts = load_manifest(_FIXTURES / "MANIFEST.csv")
    assert len(artifacts) == 108
    assert {artifact.source_commit for artifact in artifacts} == {
        "b8b227d456a6c7e65a590dfb8f00e80e89d86a3c",
        "60d9902933d86a6133935586a0da4952a5803f9e",
        "f06564b3c09c260e64b6d613d9d0424f1621779a",
        "6dcbf47c45ac35bd3c2307113d12bdad42f415bd",
        "6b82a7f610349c2e46bcd97a0df38f9bdca1daf6",
        "eda1383f5fa9d8ba3cb27f99db1d2c79494404c9",
        "29c4fd393049ac3483d9ffa034e867361d0f1764",
        "952ddfe53f62a150c53559249c83370630254cab",
        "b75e8c9d497fbbca5f5f518700f05ec6c897a2bd",
        "dfa5d131e2b41a02721cad0d4856b8ed2049f38f",
        "34a7d6adc46497d2431b0232b500a52958c6670b",
            "0d85612ba2498d248c9a599b73ebd5dbcb4c04eb",
            "ac12b77974a4478fb4f24ae8b41bf74b808fb03a",
        }
    assert {artifact.license for artifact in artifacts} == {
        "AGPL-3.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-2.1-or-later",
        "MIT",
        "GPL-2.0-or-later",
        "LGPL-3.0-only",
        "Unlicense",
        "LGPL-2.1-only",
    }
    assert {artifact.evidence_class for artifact in artifacts} <= EVIDENCE_CLASSES
    tinygs = next(
        artifact
        for artifact in artifacts
        if artifact.artifact_id == "tinygs-grizu-schema"
    )
    assert tinygs.license == "MIT"
    assert tinygs.evidence_class == "payload_only_parser"
    assert "Project-authorized MIT treatment" in tinygs.expected_output
    assert "cannot satisfy framing parity" in tinygs.expected_output


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_commit", "", "empty fields"),
        ("license", "", "empty fields"),
        ("sha256", "bad", "invalid SHA-256"),
        ("evidence_class", "guess", "unknown evidence class"),
        ("expected_output", "", "empty fields"),
    ],
)
def test_evidence_manifest_rejects_incomplete_rows(field: str, value: str, message: str):
    with (_FIXTURES / "MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_manifest_rows((row,))

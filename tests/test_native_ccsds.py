"""Construction and correction tests for native CCSDS hard-bit profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.linecode import ccsds_randomize, differential_encode
from native_framing.profiles.ccsds import SYNCWORD
from native_framing.provenance import load_manifest
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity
from native_framing.viterbi import ConvolutionalCode

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 9_600.0
_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).parent / "fixtures/native_framing/MANIFEST.csv"


def _fsk_capture(bits: np.ndarray, *, mod_index: float = 0.8) -> np.ndarray:
    samples_per_symbol = int(_IQ_SAMPLE_RATE / _IQ_SYMBOL_RATE)
    symbols = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    instantaneous_hz = (
        np.repeat(symbols, samples_per_symbol) * mod_index * _IQ_SYMBOL_RATE / 2.0
    )
    phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / _IQ_SAMPLE_RATE
    burst = np.exp(1j * phase).astype(np.complex64)
    return np.concatenate(
        (np.zeros(2_000, dtype=np.complex64), burst, np.zeros(2_000, dtype=np.complex64))
    )


def _stream(wire: bytes) -> np.ndarray:
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


def test_ccsds_differential_decoder_oracle_is_hash_pinned() -> None:
    artifacts = {artifact.artifact_id: artifact for artifact in load_manifest(_MANIFEST)}
    artifact = artifacts["gnuradio-differential-decoder"]
    assert artifact.source_commit == "f06564b3c09c260e64b6d613d9d0424f1621779a"
    assert artifact.sha256 == (
        "ef7837a3983ac740107cc1209c38c54994ec031807b093d60dd62884dbb0dd74"
    )
    assert artifact.license == "GPL-3.0-or-later"
    source = (
        _ROOT
        / "related-projects/gnu-radio/gnuradio/gr-digital/lib/diff_decoder_bb_impl.cc"
    )
    if source.is_file():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact.sha256


@pytest.mark.parametrize("basis", ["conventional", "dual"])
@pytest.mark.parametrize("interleaving", [1, 2, 4])
@pytest.mark.parametrize("scrambler", ["none", "CCSDS"])
def test_ccsds_rs_basis_interleave_randomizer_and_chunking(
    basis: str, interleaving: int, scrambler: str
):
    payload = bytes(range(32 * interleaving))
    wire = CcsdsReedSolomon(basis=basis, interleaving=interleaving).encode(payload)
    if scrambler == "CCSDS":
        wire = ccsds_randomize(wire)
    stream = 1 - _stream(wire)
    decoder = build_decoder(
        "CCSDS Reed-Solomon",
        {
            "frame_size": len(payload),
            "rs_basis": basis,
            "rs_interleaving": interleaving,
            "scrambler": scrambler,
            "sync_threshold": 0,
        },
    )
    frames = []
    for start in range(0, stream.size, 17):
        frames += decoder.push(stream[start : start + 17])
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].polarity is Polarity.INVERTED
    assert frames[0].corrected_symbols == 0


def test_ccsds_rs_correction_count_and_uncorrectable_rejection():
    payload = bytes(range(223))
    codec = CcsdsReedSolomon(basis="dual")
    corrected = bytearray(codec.encode(payload))
    for index in range(16):
        corrected[index * 11] ^= index + 1
    frames = build_decoder(
        "CCSDS RS", {"frame_size": 223, "scrambler": "none", "sync_threshold": 0}
    ).push(_stream(bytes(corrected)))
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].corrected_symbols == 16

    rejected = bytearray(codec.encode(payload))
    for index in range(17):
        rejected[index * 11] ^= index + 1
    assert build_decoder(
        "CCSDS RS", {"frame_size": 223, "scrambler": "none", "sync_threshold": 0}
    ).push(_stream(bytes(rejected))) == []


@pytest.mark.parametrize("precoding", ["none", "differential"])
def test_ccsds_rs_fsk_file_iq_replay_routes_parameterized_native_profile(
    tmp_path: Path, precoding: str
) -> None:
    payload = bytes(range(64))
    wire = ccsds_randomize(CcsdsReedSolomon(basis="dual").encode(payload))
    channel_bits = _stream(wire)
    if precoding == "differential":
        channel_bits = np.concatenate(
            (np.tile(np.asarray([0, 1], dtype=np.uint8), 8), channel_bits)
        )
        channel_bits = differential_encode(channel_bits)
    path = tmp_path / f"ccsds-rs-{precoding}-fsk.cf32"
    _fsk_capture(channel_bits).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("CCSDS Reed-Solomon",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        native_evaluation=True,
        framing_parameters={
            "modulation": "fsk",
            "mod_index": 0.8,
            "frame_size": 64.0,
            "rs_basis": "dual",
            "rs_interleaving": 1.0,
            "scrambler": "CCSDS",
            "precoding": precoding,
            "sync_threshold": 0.0,
        },
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [payload]
    assert records[0]["framing"] == "ccsds_reed_solomon"
    assert records[0]["metadata"]["precoding"] == precoding


def test_ccsds_uncoded_is_explicit_and_reports_no_integrity():
    payload = bytes(range(64))
    wire = ccsds_randomize(payload)
    frames = build_decoder(
        "CCSDS Uncoded", {"frame_size": 64, "scrambler": "CCSDS", "sync_threshold": 0}
    ).push(_stream(wire))
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].integrity is IntegrityStatus.NOT_PRESENT
    assert "no integrity gate" in frames[0].metadata["false_positive_policy"]


def test_ccsds_profiles_reject_invalid_precoding_and_bad_interleave():
    with pytest.raises(ValueError, match="one of"):
        build_decoder("CCSDS RS", {"precoding": "NRZI"})
    with pytest.raises(ValueError, match="divide"):
        build_decoder("CCSDS RS", {"frame_size": 223, "rs_interleaving": 2})


def test_ccsds_profiles_are_not_production_enabled_or_claimed_complete():
    for label in ("CCSDS Concatenated", "CCSDS Reed-Solomon", "CCSDS Uncoded"):
        profile = REGISTRY.resolve(label)
        assert profile is not None
        assert profile.disposition is DecodeDisposition.IN_PROGRESS
        assert profile.decoder_available
        assert not profile.live_supported
        assert not profile.post_pass_supported


@pytest.mark.parametrize("label", ["CCSDS Reed-Solomon", "CCSDS Uncoded"])
@pytest.mark.parametrize("invert", [False, True])
def test_ccsds_hard_profiles_differential_precoding_is_streaming_and_bounded(
    label: str, invert: bool
) -> None:
    payload = bytes(range(48))
    if label == "CCSDS Reed-Solomon":
        wire = CcsdsReedSolomon(basis="dual").encode(payload)
    else:
        wire = payload
    prefix = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
    plain = np.concatenate((prefix, _stream(wire)))
    channel = differential_encode(plain)
    if invert:
        channel = 1 - channel
    decoder = build_decoder(
        label,
        {
            "frame_size": len(payload),
            "scrambler": "none",
            "precoding": "differential",
            "sync_threshold": 0,
        },
    )
    frames = []
    for offset in range(channel.size):
        frames += decoder.push(channel[offset : offset + 1])
        assert decoder.retained_symbols <= decoder.max_retained_symbols

    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].source_start == prefix.size
    assert frames[0].polarity is Polarity.AMBIGUOUS
    assert frames[0].metadata["precoding"] == "differential"
    assert frames[0].metadata["line_polarity_unobservable"] is True


def test_ccsds_differential_and_plain_profiles_cross_reject_and_flush_resets() -> None:
    payload = bytes(range(48))
    plain = _stream(CcsdsReedSolomon(basis="dual").encode(payload))
    differential = differential_encode(plain)
    parameters = {
        "frame_size": len(payload),
        "scrambler": "none",
        "sync_threshold": 0,
    }
    assert build_decoder(
        "CCSDS RS", {**parameters, "precoding": "differential"}
    ).push(plain) == []
    assert build_decoder("CCSDS RS", parameters).push(differential) == []

    decoder = build_decoder(
        "CCSDS RS", {**parameters, "precoding": "differential"}
    )
    assert [frame.payload for frame in decoder.push(differential)] == [payload]
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(differential)] == [payload]


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("convention", ["CCSDS", "NASA-DSN"])
def test_ccsds_concatenated_dual_phase_chunked_decode(phase: int, convention: str):
    payload = bytes(range(48))
    wire = ccsds_randomize(CcsdsReedSolomon(basis="dual").encode(payload))
    decoded_bits = tuple(_stream(wire)) + tuple(
        np.random.default_rng(91).integers(0, 2, 120)
    )
    encoded = ConvolutionalCode(convention).encode(decoded_bits, mode="truncated")
    soft = np.asarray((0.1,) * phase + tuple(4.0 if bit else -4.0 for bit in encoded))
    decoder = build_decoder(
        "CCSDS Concatenated",
        {
            "frame_size": len(payload),
            "convolutional": convention,
            "sync_threshold": 0,
            "viterbi_traceback": 35,
        },
    )
    frames = []
    for offset in range(0, soft.size, 23):
        frames += decoder.push(soft[offset : offset + 23])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    frames += decoder.flush()
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].canonical_framing == "ccsds_concatenated"
    assert frames[0].metadata["convolutional_phase"] == phase
    assert frames[0].metadata["source_offset_domain"] == "input_soft_symbols"


@pytest.mark.parametrize("basis", ["conventional", "dual"])
@pytest.mark.parametrize("interleaving", [1, 2, 4])
@pytest.mark.parametrize("scrambler", ["none", "CCSDS"])
@pytest.mark.parametrize("precoding", ["none", "differential"])
def test_ccsds_concatenated_complete_rs_and_optional_stage_matrix(
    basis: str,
    interleaving: int,
    scrambler: str,
    precoding: str,
) -> None:
    payload = bytes((index * 37 + 11) & 0xFF for index in range(16 * interleaving))
    wire = CcsdsReedSolomon(basis=basis, interleaving=interleaving).encode(payload)
    if scrambler == "CCSDS":
        wire = ccsds_randomize(wire)
    decoded_bits = _stream(wire)
    if precoding == "differential":
        decoded_bits = differential_encode(decoded_bits)
    decoded_bits = np.concatenate(
        (decoded_bits, np.random.default_rng(93).integers(0, 2, 120, dtype=np.uint8))
    )
    encoded = ConvolutionalCode("CCSDS").encode(decoded_bits, mode="truncated")
    soft = np.asarray(encoded, dtype=np.float64) * 8.0 - 4.0
    decoder = build_decoder(
        "CCSDS Concatenated",
        {
            "frame_size": len(payload),
            "rs_basis": basis,
            "rs_interleaving": interleaving,
            "scrambler": scrambler,
            "precoding": precoding,
            "sync_threshold": 0,
            "viterbi_traceback": 35,
        },
    )
    frames = []
    for offset in range(0, soft.size, 31):
        frames += decoder.push(soft[offset : offset + 31])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    frames += decoder.flush()

    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].metadata["rs_basis"] == basis
    assert frames[0].metadata["rs_interleaving"] == interleaving
    assert frames[0].metadata["randomizer"] == scrambler
    assert frames[0].metadata["precoding"] == precoding


@pytest.mark.parametrize("precoding", ["none", "differential"])
def test_ccsds_concatenated_fsk_file_iq_replay_routes_soft_viterbi_rs_chain(
    tmp_path: Path, precoding: str
) -> None:
    payload = bytes(range(48))
    wire = ccsds_randomize(CcsdsReedSolomon(basis="dual").encode(payload))
    decoded_bits = np.concatenate(
        (_stream(wire), np.random.default_rng(91).integers(0, 2, 120))
    ).astype(np.uint8)
    if precoding == "differential":
        decoded_bits = np.concatenate(
            (np.tile(np.asarray([0, 1], dtype=np.uint8), 8), decoded_bits)
        )
        decoded_bits = differential_encode(decoded_bits)
    encoded = np.asarray(
        ConvolutionalCode("CCSDS").encode(decoded_bits, mode="truncated"),
        dtype=np.uint8,
    )
    path = tmp_path / f"ccsds-concatenated-{precoding}-fsk.cf32"
    _fsk_capture(encoded).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("CCSDS Concatenated",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        native_evaluation=True,
        framing_parameters={
            "modulation": "fsk",
            "mod_index": 0.8,
            "frame_size": 48.0,
            "rs_enabled": True,
            "rs_basis": "dual",
            "rs_interleaving": 1.0,
            "scrambler": "CCSDS",
            "precoding": precoding,
            "convolutional": "CCSDS",
            "viterbi_traceback": 35.0,
            "sync_threshold": 0.0,
        },
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [payload]
    assert records[0]["framing"] == "ccsds_concatenated"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["metadata"]["precoding"] == precoding


@pytest.mark.parametrize("phase", [0, 1])
def test_ccsds_concatenated_differential_precoding_dual_phase(phase: int) -> None:
    payload = bytes(range(48))
    wire = ccsds_randomize(CcsdsReedSolomon(basis="dual").encode(payload))
    plain = np.concatenate(
        (
            _stream(wire),
            np.random.default_rng(92).integers(0, 2, 120, dtype=np.uint8),
        )
    )
    precoded = differential_encode(plain)
    encoded = ConvolutionalCode("CCSDS").encode(precoded, mode="truncated")
    soft = np.asarray((0.1,) * phase + tuple(4.0 if bit else -4.0 for bit in encoded))
    decoder = build_decoder(
        "CCSDS Concatenated",
        {
            "frame_size": len(payload),
            "precoding": "differential",
            "sync_threshold": 0,
            "viterbi_traceback": 35,
        },
    )
    frames = []
    for offset in range(0, soft.size, 19):
        frames += decoder.push(soft[offset : offset + 19])
    frames += decoder.flush()

    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].metadata["convolutional_phase"] == phase
    assert frames[0].metadata["precoding"] == "differential"
    assert frames[0].polarity is Polarity.AMBIGUOUS


def test_ccsds_concatenated_optional_rs_is_explicit():
    payload = bytes(range(40))
    decoded_bits = tuple(_stream(payload)) + (0, 1) * 50
    encoded = ConvolutionalCode().encode(decoded_bits, mode="truncated")
    soft = np.asarray(encoded) * 2.0 - 1.0
    frames = build_decoder(
        "CCSDS Concatenated",
        {
            "frame_size": len(payload),
            "rs_enabled": False,
            "scrambler": "none",
            "sync_threshold": 0,
            "viterbi_traceback": 35,
        },
    ).push(soft)
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].integrity is IntegrityStatus.NOT_PRESENT

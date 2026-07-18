"""Pinned-algorithm construction tests for native SMOG-P RA framing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.codes.ra import encode_ra, ra_config, ra_wire_soft
from native_framing.profiles.smogp_ra import SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity

_SYNC = np.fromiter((character == "1" for character in SYNCWORD), dtype=np.uint8)
_PREFIX = np.asarray([-0.7, 0.8, -0.9, -0.6, 0.75], dtype=np.float64)
_ENCODED_SHA256 = {
    128: "7e538088f2300efa933a16aa450adaf92b051093b8862c99239a5f56bde47783",
    256: "d864a8e6e1a77063ec20bd9f5af72808a273ecdf8ab7976af196b724c70141b8",
}
_IQ_SAMPLE_RATE = 50_000.0
_IQ_SYMBOL_RATE = 2_500.0


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


def _payload(frame_size: int) -> bytes:
    return bytes((index * 37 + frame_size) & 0xFF for index in range(frame_size))


def _stream(
    frame_size: int,
    *,
    inverted: bool = False,
    sync_errors: int = 0,
    ra_error_fraction: float = 0.0,
) -> tuple[np.ndarray, bytes]:
    payload = _payload(frame_size)
    sync = _SYNC.copy()
    sync[:sync_errors] ^= 1
    sync_soft = np.where(sync != 0, 0.9, -0.9)
    capture = ra_wire_soft(payload, magnitude=0.8)
    if ra_error_fraction:
        rng = np.random.default_rng(0x2DD4 + frame_size)
        positions = rng.choice(
            capture.size,
            int(capture.size * ra_error_fraction),
            replace=False,
        )
        capture[positions] *= -1
    stream = np.concatenate((_PREFIX, sync_soft, capture))
    return (-stream if inverted else stream), payload


@pytest.mark.parametrize("frame_size", [128, 256])
@pytest.mark.parametrize("inverted", [False, True])
def test_smogp_ra_sizes_chunks_polarity_offsets_and_literal_encoder(
    frame_size: int, inverted: bool
) -> None:
    encoded = encode_ra(_payload(frame_size)).astype("<u2", copy=False).tobytes()
    assert len(encoded) == {128: 260, 256: 514}[frame_size]
    assert hashlib.sha256(encoded).hexdigest() == _ENCODED_SHA256[frame_size]

    stream, expected = _stream(frame_size, inverted=inverted)
    decoder = build_decoder("SMOG-P RA", {"frame_size": frame_size})
    frames = []
    for start in range(0, stream.size, 137):
        frames += decoder.push(stream[start : start + 137])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.source_start == _PREFIX.size and frame.source_end == stream.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.integrity is IntegrityStatus.NOT_PRESENT
    assert frame.corrected_symbols is None
    assert frame.metadata["variant"] == "SMOG-P"
    assert frame.metadata["frame_size"] == frame_size
    assert frame.metadata["recode_bit_errors"] == 0


def test_smogp_ra_corrects_noise_and_applies_recode_gate() -> None:
    stream, expected = _stream(128, ra_error_fraction=0.03)
    frames = build_decoder("SMOG-P RA").push(stream)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["recode_bit_errors"] > 0
    assert frames[0].metadata["recode_error_fraction"] < 0.35
    assert build_decoder(
        "SMOG-P RA", {"error_threshold": 0.02}
    ).push(stream) == []


@pytest.mark.parametrize("frame_size", [128, 256])
def test_smogp_ra_fsk_file_iq_replay_routes_parameterized_soft_native_profile(
    tmp_path: Path, frame_size: int,
) -> None:
    soft_stream, expected = _stream(frame_size)
    capture = _fsk_capture(np.asarray(soft_stream >= 0, dtype=np.uint8))
    path = tmp_path / f"smogp-ra-{frame_size}-fsk.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("SMOG-P RA",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        framing_parameters={
            "modulation": "fsk",
            "mod_index": 0.8,
            # Mirrors protobuf Struct JSON semantics: integer-valued protocol
            # parameters arrive at the flowgraph boundary as doubles.
            "frame_size": float(frame_size),
        },
        window_s=3.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "smogp_ra"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


def test_smogp_ra_sync_threshold_corruption_truncation_and_flush() -> None:
    accepted, payload = _stream(128, sync_errors=1)
    assert build_decoder("SMOG-P RA").push(accepted) == []
    frames = build_decoder("SMOG-P RA", {"sync_threshold": 1}).push(accepted)
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].sync_distance == pytest.approx(1.0)

    corrupted, _ = _stream(128, ra_error_fraction=0.40)
    assert build_decoder("SMOG-P RA").push(corrupted) == []

    clean, payload = _stream(128)
    decoder = build_decoder("SMOG-P RA")
    assert decoder.push(clean[:-1]) == []
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(clean)] == [payload]


def test_smogp_ra_registry_contract_and_bounds() -> None:
    profile = REGISTRY.resolve("SMOGP RA")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported and not profile.post_pass_supported
    assert ra_config(128).code_length == 130
    assert ra_config(256).code_length == 257
    with pytest.raises(ValueError):
        build_decoder("SMOG-P RA", {"frame_size": 126})
    with pytest.raises(ValueError):
        build_decoder("SMOG-P RA", {"error_threshold": 0.36})

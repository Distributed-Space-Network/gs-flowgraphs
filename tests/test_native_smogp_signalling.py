"""Construction tests for explicit-only SMOG-P signalling extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.fsk_audio import FskAudioConfig, decode_fsk_audio_mm_profile
from native_framing.profiles.smogp import RX_SYNCWORD, TX_SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity
from scipy.io import wavfile

_PAYLOAD = bytes(range(64))
_ROOT = Path(__file__).resolve().parents[2]
_SMOGP_WAV = _ROOT / "related-projects/satellite-recordings/smog_p.wav"
_SMOGP_WAV_SHA256 = "3df5616c8b66b2ae566369627f4441dbe29cec5a8bf01b200f52852dffe5fdad"
_CAPTURE_PAYLOAD = bytes.fromhex(
    "6d08f7835d9e5982c0fd1dcaad3b5bebd493e14a04d228ddf90153d2e66c5b25"
    "6531c57ce7f138612d5c033ac68890db8c8c42f3517543a083930000ff0000ff"
)


def _stream(syncword: str, payload: bytes = _PAYLOAD) -> np.ndarray:
    sync = np.fromiter((char == "1" for char in syncword), dtype=np.uint8)
    return np.concatenate((sync, np.unpackbits(np.frombuffer(payload, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 63, 64, 255, 1024])
def test_smogp_rx_sync_chunk_boundaries_and_no_integrity_metadata(step: int):
    stream = _stream(RX_SYNCWORD)
    decoder = build_decoder("SMOG-P Signalling", {"sync_threshold": 0})
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].integrity is IntegrityStatus.NOT_PRESENT
    assert frames[0].metadata["sync_variant"] == "rx"
    assert "never autodetect" in frames[0].metadata["false_positive_policy"]


def test_smogp_new_protocol_option_controls_tx_observation_sync():
    stream = _stream(TX_SYNCWORD)
    assert build_decoder("SMOG-P Signalling", {"sync_threshold": 0}).push(stream) == []
    frames = build_decoder(
        "SMOG-P Signalling", {"sync_threshold": 0, "new_protocol": True}
    ).push(stream)
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].metadata["sync_variant"] == "tx-observation"


def test_smogp_payload_mutation_is_emitted_because_protocol_has_no_integrity():
    mutated = bytearray(_PAYLOAD)
    mutated[10] ^= 1
    frames = build_decoder("SMOG-P Signalling", {"sync_threshold": 0}).push(
        _stream(RX_SYNCWORD, bytes(mutated))
    )
    assert [frame.payload for frame in frames] == [bytes(mutated)]
    assert frames[0].integrity is IntegrityStatus.NOT_PRESENT


def test_smogp_signalling_is_not_enabled_for_production_or_autodetect():
    profile = REGISTRY.resolve("SMOGP Signalling")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported
    assert "never autodetected" in profile.integrity_policy


def test_smogp_signalling_published_wav_replays_exact_zero_error_sync() -> None:
    assert hashlib.sha256(_SMOGP_WAV.read_bytes()).hexdigest() == _SMOGP_WAV_SHA256
    sample_rate, audio = wavfile.read(_SMOGP_WAV)
    assert sample_rate == 48_000 and audio.dtype == np.int16 and audio.ndim == 1

    symbols, decoded = decode_fsk_audio_mm_profile(
        audio,
        FskAudioConfig(sample_rate, 1_250),
        "SMOG-P Signalling",
        {"sync_threshold": 0},
        cutoff_hz=1_200,
        transition_hz=100,
        gain_mu=0.05,
        omega_relative_limit=0.01,
    )

    assert symbols.phase_samples == 0
    assert len(decoded) == 1
    located = decoded[0]
    frame = located.frame
    assert frame.payload == _CAPTURE_PAYLOAD
    assert hashlib.sha256(frame.payload).hexdigest() == (
        "bbd10abfac153fc70ccf5037148cccd67afdc295fbfe0b14326d57acd6033a91"
    )
    assert (frame.source_start, frame.source_end) == (774, 1_350)
    assert (located.source_sample_start, located.source_sample_end) == (
        29_706,
        51_824,
    )
    assert frame.sync_distance == 0
    assert frame.polarity is Polarity.NORMAL
    assert frame.integrity is IntegrityStatus.NOT_PRESENT
    assert frame.metadata["sync_variant"] == "rx"
    assert "never autodetect" in frame.metadata["false_positive_policy"]

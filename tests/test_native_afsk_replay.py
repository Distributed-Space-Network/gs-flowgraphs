"""Deterministic binary-AFSK replay, including the pinned S-NET capture."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from native_framing import AfskConfig, decode_afsk_profile, demodulate_afsk
from native_framing.output import utc_from_sample_offset
from scipy.io import wavfile

_SNET_WAV = (
    Path(__file__).resolve().parents[2]
    / "related-projects"
    / "satellite-recordings"
    / "snet_a.wav"
)
_SNET_WAV_SHA256 = "0ab9ec292dd45014ab73360d4ab2596d0bad18dc9eeb934eb75c857f705f2e2e"
_SNET_CONFIG = AfskConfig(48_000.0, 1_200.0, one_hz=1_200.0, zero_hz=1_800.0)
_FIRST_PAYLOAD = bytes.fromhex(
    "f3501ae0240a2c660a873a448f0dd101a40ab063a9079210f73f5c0018001400"
    "11008b001d001a5e3211000751001600c34b1600ce17000f905b7f08625e091b"
    "f41b1816000ed3110f00000708005c000d03fe02080066005e00000070058b00"
    "460000000564ea65975b6303000000000000"
)


def _fractional_afsk(bits: np.ndarray, config: AfskConfig) -> np.ndarray:
    boundaries = [
        round(index * config.sample_rate_hz / config.symbol_rate_hz)
        for index in range(bits.size + 1)
    ]
    output = np.empty(boundaries[-1], dtype=np.float64)
    for index, bit in enumerate(bits):
        start, end = boundaries[index : index + 2]
        time = np.arange(end - start, dtype=np.float64) / config.sample_rate_hz
        frequency = config.one_hz if bit else config.zero_hz
        output[start:end] = np.sin(2 * np.pi * frequency * time + 0.31)
    return output


def test_fractional_sample_clock_afsk_demodulation_and_validation() -> None:
    config = AfskConfig(44_100.0, 1_200.0, one_hz=1_200.0, zero_hz=2_200.0)
    expected = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
    decoded = demodulate_afsk(_fractional_afsk(expected, config), config)
    np.testing.assert_array_equal(decoded.hard_bits, expected)
    assert decoded.sample_offset(4) == 147
    assert decoded.sample_offset(expected.size) == 441
    with pytest.raises(ValueError, match="one-dimensional"):
        demodulate_afsk(np.zeros((4, 2)), config)
    with pytest.raises(ValueError, match="Nyquist"):
        AfskConfig(8_000, 1_200, one_hz=4_000, zero_hz=1_200)
    with pytest.raises(ValueError, match="finite positive"):
        AfskConfig("invalid", 1_200, one_hz=1_200, zero_hz=2_200)


def test_pinned_snet_wav_replays_to_exact_native_frames_and_source_time() -> None:
    assert hashlib.sha256(_SNET_WAV.read_bytes()).hexdigest() == _SNET_WAV_SHA256
    sample_rate, audio = wavfile.read(_SNET_WAV)
    assert sample_rate == 48_000 and audio.ndim == 1

    symbols, decoded = decode_afsk_profile(
        audio,
        _SNET_CONFIG,
        "SNET",
        {"buggy_crc": True},
        phase_samples=0,
    )
    assert len(decoded) == 2
    first = decoded[0]
    assert first.frame.payload == _FIRST_PAYLOAD
    assert (first.frame.source_start, first.frame.source_end) == (663, 4_791)
    assert (first.source_sample_start, first.source_sample_end) == (26_520, 191_640)
    assert symbols.sample_offset(first.frame.source_start) == first.source_sample_start

    pass_start = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    assert utc_from_sample_offset(pass_start, first.source_sample_start, sample_rate) == (
        pass_start + timedelta(seconds=0.5525)
    )
    _, standards_only = decode_afsk_profile(audio, _SNET_CONFIG, "SNET", phase_samples=0)
    assert standards_only == []

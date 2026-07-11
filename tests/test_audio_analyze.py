"""SatNOGS discriminator-audio symbol recovery."""

from __future__ import annotations

import numpy as np
from audio_analyze import DEFAULT_BAUDS, decode_audio, decode_ax25_audio, slice_symbols

from gfsk_ax25 import ax25, endurosat_link
from gfsk_ax25 import framing as ax25_framing


def _discriminator_audio(bits: np.ndarray, sps: int = 5) -> np.ndarray:
    # Gaussian-like transitions are not needed to test the audio boundary: this
    # represents the positive/negative discriminator levels presented to it.
    levels = bits.astype(np.float32) * 2.0 - 1.0
    return np.repeat(levels, sps)


def test_default_baud_sweep_covers_practical_satnogs_rates() -> None:
    assert DEFAULT_BAUDS == (1200.0, 2400.0, 4800.0, 9600.0, 19200.0)


def test_slice_symbols_recovers_five_sample_symbols() -> None:
    expected = np.tile(np.array([0, 1, 1, 0], dtype=np.uint8), 100)
    audio = _discriminator_audio(expected)
    got = slice_symbols(audio, 48_000.0, 9600.0)
    np.testing.assert_array_equal(got[: len(expected)], expected)


def test_decode_ax25_audio_recovers_crc_valid_g3ruh_frame() -> None:
    body = ax25.encode_ui(dest="CQ", src="DSN", info=b"SATNOGS AUDIO")
    bits = ax25_framing.encode(body, scramble=True, nrzi=True)
    audio = np.concatenate([
        np.zeros(12_000, dtype=np.float32),
        _discriminator_audio(bits),
        np.zeros(12_000, dtype=np.float32),
    ])
    frames = decode_ax25_audio(audio, 48_000.0, 9600.0, window_s=1.0)
    assert any(frame == body for _, frame in frames)


def test_decode_ax25_audio_tracks_symbol_clock_error() -> None:
    body = ax25.encode_ui(dest="CQ", src="DSN", info=b"CLOCK RECOVERY" * 8)
    bits = ax25_framing.encode(body, scramble=True, nrzi=True)
    actual_sps = 5.004  # 800 ppm error accumulates appreciably over the frame
    sample_count = int(len(bits) * actual_sps)
    indices = np.minimum((np.arange(sample_count) / actual_sps).astype(int), len(bits) - 1)
    audio = (bits[indices].astype(np.float32) * 2.0 - 1.0)
    frames = decode_ax25_audio(audio, 48_000.0, 9600.0, window_s=1.0)
    assert any(frame == body for _, frame in frames)


def test_decode_audio_recovers_endurosat_crc_valid_frame() -> None:
    payload = bytes(range(32))
    bits = endurosat_link.frame_bits(payload)
    audio = np.concatenate([
        np.zeros(12_000, dtype=np.float32),
        _discriminator_audio(bits),
        np.zeros(12_000, dtype=np.float32),
    ])
    results = decode_audio(audio, 48_000.0, 9600.0, window_s=1.0)
    assert any(payload in frame for _, frame in results["endurosat"])
    assert results["ax25"] == []


def test_decode_ax25_audio_does_not_forge_frames_from_noise() -> None:
    noise = np.random.default_rng(4).normal(0, 1, 48_000).astype(np.float32)
    assert decode_ax25_audio(noise, 48_000.0, 9600.0, window_s=1.0) == []

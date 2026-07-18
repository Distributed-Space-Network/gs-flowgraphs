"""Generated construction tests for native AO-40 uncoded framing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from native_framing import (
    BpskAudioConfig,
    BpskConfig,
    build_decoder,
    decode_bpsk_audio_profile,
)
from native_framing.crc import CRC16_CCITT_FALSE
from native_framing.profiles.ao40 import UNCODED_SYNCWORD
from native_framing.provenance import load_manifest
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parents[2]
_RECORDING = _ROOT / "related-projects/satellite-recordings/ao40_uncoded.wav"
_MANIFEST = Path(__file__).parent / "fixtures/native_framing/MANIFEST.csv"

_SYNC = np.fromiter((char == "1" for char in UNCODED_SYNCWORD), dtype=np.uint8)
_PAYLOAD = bytes(range(256)) * 2
_WIRE = CRC16_CCITT_FALSE.append(_PAYLOAD, byteorder="big")
_STREAM = np.concatenate((_SYNC, np.unpackbits(np.frombuffer(_WIRE, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 17, 31, 512, 8192])
@pytest.mark.parametrize("inverted", [False, True])
def test_ao40_uncoded_chunk_polarity_and_crc(step: int, inverted: bool):
    stream = 1 - _STREAM if inverted else _STREAM
    decoder = build_decoder("AO-40 uncoded")
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frames[0].source_start == 0
    assert frames[0].source_end == stream.size


def test_ao40_uncoded_rejects_corruption_threshold_and_truncation():
    corrupted = _STREAM.copy()
    corrupted[32 + 100] ^= 1
    assert build_decoder("AO-40 uncoded").push(corrupted) == []

    sync_bad = _STREAM.copy()
    sync_bad[[0, 1, 2, 3]] ^= 1
    assert build_decoder("AO-40 uncoded").push(sync_bad) == []

    decoder = build_decoder("AO-40 uncoded")
    assert decoder.push(_STREAM[:-1]) == []
    assert decoder.flush() == []


def test_ao40_uncoded_is_available_without_overclaiming_completion():
    profile = REGISTRY.resolve("AO40 uncoded")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported


def test_ao40_uncoded_published_wav_replays_crc_valid_frames_byte_exactly() -> None:
    assert hashlib.sha256(_RECORDING.read_bytes()).hexdigest() == (
        "6f02b79fe5c6085276ebb6b847f365c8e39d4f7d46671d4785d931cac4d95d9d"
    )
    with pytest.warns(wavfile.WavFileWarning, match="non-data"):
        sample_rate, audio = wavfile.read(_RECORDING)
    assert sample_rate == 48_000 and audio.dtype == np.int16 and audio.ndim == 1

    replay = decode_bpsk_audio_profile(
        audio,
        BpskAudioConfig(
            sample_rate,
            400,
            differential=True,
            manchester=True,
            clock_search_ppm=250,
            clock_search_steps=5,
        ),
        "AO-40 uncoded",
    )

    assert replay.selected_symbol_rate_hz == pytest.approx(399.95)
    assert replay.selected_clock_error_ppm == pytest.approx(-125.0)
    assert replay.selected_manchester_phase == 0
    assert replay.carrier_frequency_min_hz == pytest.approx(1481.356, abs=0.001)
    assert replay.carrier_frequency_max_hz == pytest.approx(1662.696, abs=0.001)
    expected = [
        (
            "d343bf60e001e4bc122df13da25308bf3320fbd415cf85445f72c547216fc15a",
            548_054,
            1_045_396,
            b"A  HI, THIS IS AMSAT OSCAR-40       2003-03-14  23:37:55",
        ),
        (
            "a8c406b1c62012ad5f1d4f7f3e03a0e5edb2ff53f7d5823e4cd05438d9f211c0",
            1_157_730,
            1_655_072,
            b"L  Whole Orbit Data V2.0   Samples: 2",
        ),
        (
            "79d2c5ddd677268d4e3568c9db41ac74416f2fcf787dd84c0550874f128fb1e3",
            1_812_532,
            2_309_874,
            b"A  HI, THIS IS AMSAT OSCAR-40       2003-03-14  23:38:21",
        ),
    ]
    assert len(replay.frames) == len(expected)
    for located, (digest, sample_start, sample_end, prefix) in zip(
        replay.frames, expected, strict=True
    ):
        frame = located.frame
        assert hashlib.sha256(frame.payload).hexdigest() == digest
        assert len(frame.payload) == 512 and frame.payload.startswith(prefix)
        assert (located.source_sample_start, located.source_sample_end) == (
            sample_start,
            sample_end,
        )
        assert frame.integrity is IntegrityStatus.PASSED
        assert frame.sync_distance == 0
        assert frame.polarity is Polarity.NORMAL
        assert frame.metadata == {"crc": "CRC-16/CCITT-FALSE", "frame_size_bytes": 514}

    # The article published exact wire bytes for the first A and L frames.
    # These hashes are independent of the native decoder and include their CRCs.
    assert [
        hashlib.sha256(
            CRC16_CCITT_FALSE.append(item.frame.payload, byteorder="big")
        ).hexdigest()
        for item in replay.frames[:2]
    ] == [
        "5e8c4d7948adb78c8178c1f2614f0588a188a6900711bf0212ca1ee137946070",
        "9ede3ad8be8de739a0ece61f0623019026d6059a71fdf53f47e780680a2f1de8",
    ]


def test_ao40_uncoded_capture_manifest_and_audio_search_fail_closed() -> None:
    artifacts = {artifact.artifact_id: artifact for artifact in load_manifest(_MANIFEST)}
    capture = artifacts["satrec-ao40-uncoded-wav"]
    assert capture.source_commit == "952ddfe53f62a150c53559249c83370630254cab"
    assert capture.sha256 == (
        "6f02b79fe5c6085276ebb6b847f365c8e39d4f7d46671d4785d931cac4d95d9d"
    )
    assert capture.license == "Unlicense"
    assert capture.evidence_class == "real_capture"
    assert "independently match" in capture.expected_output

    with pytest.raises(ValueError, match="odd integer"):
        BpskAudioConfig(48_000, 400, clock_search_steps=4)
    with pytest.raises(ValueError, match="Manchester"):
        BpskConfig(48_000, 400, manchester_phase=0)
    with pytest.raises(ValueError, match="measurable BPSK carrier"):
        decode_bpsk_audio_profile(
            np.zeros(4_800),
            BpskAudioConfig(
                48_000,
                400,
                differential=True,
                manchester=True,
                clock_search_steps=1,
            ),
            "AO-40 uncoded",
        )

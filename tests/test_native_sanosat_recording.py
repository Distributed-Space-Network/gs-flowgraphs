"""Mission-source and hardware-recording qualification for SanoSat-1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.provenance import load_manifest
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

_GS_ROOT = Path(__file__).resolve().parents[2]
_MISSION = _GS_ROOT / "related-projects" / "sanosat-1"
_RECORDING = (
    _MISSION
    / "Transmission Protocol"
    / "Recordings"
    / "AllInOneRecording"
    / "18-55-44_436233kHz.wav"
)
_MANIFEST = Path(__file__).parent / "fixtures" / "native_framing" / "MANIFEST.csv"
_EXPECTED_PAYLOAD = b"NPQDIGIPEATER TEST SANOSAT\x00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sanosat_mission_pdf_generator_and_recording_are_hash_pinned() -> None:
    artifacts = {artifact.artifact_id: artifact for artifact in load_manifest(_MANIFEST)}
    expected = {
        "sanosat-protocol-pdf": (
            "53b7a049b90cdf9d493a960d741066894065782b503b04752ce7292a64a9e40c"
        ),
        "sanosat-packet-generator": (
            "ba763da038f0d2bbfcd588dfe5c686309e5890e9ff60b3a579b746ef453baa8e"
        ),
        "sanosat-hardware-recording": (
            "0df819b14a6e9d787111c89ef02e1ed6ca14bbb80bf01c4b1148bd50c985ae39"
        ),
    }
    assert {key: artifacts[key].sha256 for key in expected} == expected
    assert {artifacts[key].source_commit for key in expected} == {
        "dfa5d131e2b41a02721cad0d4856b8ed2049f38f"
    }
    assert {artifacts[key].license for key in expected} == {"MIT"}
    assert all(
        "Project-authorized MIT treatment" in artifacts[key].expected_output
        and "no license file" in artifacts[key].expected_output
        for key in expected
    )

    local_sources = {
        "sanosat-protocol-pdf": _MISSION
        / "Transmission Protocol"
        / "Transmission Protocol.pdf",
        "sanosat-packet-generator": _MISSION / "sources" / "sanosatdigipacketparser.py",
        "sanosat-hardware-recording": _RECORDING,
    }
    for key, path in local_sources.items():
        if path.exists():
            assert _sha256(path) == expected[key]


@pytest.mark.skipif(not _RECORDING.exists(), reason="pinned SanoSat source checkout absent")
def test_sanosat_hardware_iq_recording_recovers_repeated_crc_valid_packets() -> None:
    sample_rate, stereo = wavfile.read(_RECORDING)
    assert sample_rate == 37_500
    assert stereo.dtype == np.int16 and stereo.ndim == 2 and stereo.shape[1] == 2

    iq = stereo[:, 0].astype(np.float64) + 1j * stereo[:, 1].astype(np.float64)
    discriminator = np.angle(iq[1:] * np.conj(iq[:-1]))
    samples_per_symbol = sample_rate // 500
    centered = discriminator - uniform_filter1d(
        discriminator, size=samples_per_symbol * 32, mode="nearest"
    )
    phase_samples = 30
    symbol_count = (centered.size - phase_samples) // samples_per_symbol
    soft = centered[
        phase_samples : phase_samples + symbol_count * samples_per_symbol
    ].reshape(symbol_count, samples_per_symbol).mean(axis=1)

    decoder = build_decoder("SanoSat")
    frames = decoder.push(soft > 0) + decoder.flush()
    matching = [frame for frame in frames if frame.payload == _EXPECTED_PAYLOAD]

    assert len(matching) == 2
    assert [frame.source_start for frame in matching] == [19_579, 21_398]
    assert all(frame.metadata["crc1"] == frame.metadata["crc2"] == "passed" for frame in matching)

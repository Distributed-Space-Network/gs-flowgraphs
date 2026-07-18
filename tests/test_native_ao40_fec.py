"""Pinned upstream parity and construction tests for native AO-40 FEC."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_ARC
from native_framing.fsk_audio import FskAudioConfig, decode_fsk_audio_mm_profile
from native_framing.linecode import ccsds_randomize
from native_framing.profiles.ao40_fec import SHORT_SYNCWORD, SYNCWORD
from native_framing.provenance import load_manifest
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, Polarity, SymbolInput
from native_framing.viterbi import ConvolutionalCode
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM = _ROOT / "related-projects/gr-satellites/python/components/deframers"
_RECORDINGS = _ROOT / "related-projects/satellite-recordings"
_MANIFEST = Path(__file__).parent / "fixtures/native_framing/MANIFEST.csv"
_VITERBI = ConvolutionalCode("CCSDS")

# These payload digests are also present in Daniel Estévez's independently
# published 2019 SMOG-P decoder output.  Keeping the entire capture replay in
# this test proves that the native demodulator and AO-40 chains reach the same
# bytes without copying the published payload dumps into the repository.
_SMOGP_CAPTURES = [
    (
        "smog_p.wav",
        "3df5616c8b66b2ae566369627f4441dbe29cec5a8bf01b200f52852dffe5fdad",
        "AO-40 FEC short",
        True,
        [
            (
                "f9b65e5cd2fb6e9f636e0fa42c18f20e1691b8d8d31b5f125a53b733037e9a2a",
                2_131,
                4_783,
                81_903,
                183_745,
            ),
            (
                "736d9bec8cb65a9e4091a899e245250dc742136ae8dccc2dca64b64b8e7af8fb",
                5_273,
                7_925,
                202_568,
                304_416,
            ),
        ],
    ),
    (
        "smog_p_long.wav",
        "0d4641d47330b682174d72cbfe91648da3e5b74b61cff44d733d2063b11c3b7c",
        "AO-40 FEC",
        False,
        [
            (
                "6be4b10e0c537123db176eaea8860e61bc3b5b961738fb04f29d654082915721",
                563,
                5_763,
                21_625,
                221_285,
            ),
            (
                "c82d39558a0c31fd9715aa0354988a3b7c53d3adfea6c76e84ad292fdfc64be6",
                6_429,
                11_629,
                246_881,
                446_567,
            ),
            (
                "c4635e8f8f6966626c6e1a580399ee1a8b596dd47f9ed3c512c1bfdf4fd3885e",
                12_293,
                17_493,
                472_082,
                671_757,
            ),
        ],
    ),
]


def _upstream_frame_reference() -> bytes:
    source = ast.parse((_UPSTREAM / "qa_ao40_fec_deframer.py").read_text())
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "frame_reference"
        ):
            return bytes(ast.literal_eval(node.value))
    raise AssertionError("pinned AO-40 QA source has no frame_reference")


def _constructed_stream(
    *,
    short: bool,
    inverted: bool = False,
    crc: bool = False,
    valid_crc: bool = True,
    rs_error_positions: tuple[int, ...] = (),
    sync_error_positions: tuple[int, ...] = (),
) -> tuple[np.ndarray, bytes, int]:
    rows = 51 if short else 80
    columns = 52 if short else 65
    output_size = 2572 if short else 5132
    output_skip = 80 if short else 65
    step = rows
    syncword = SHORT_SYNCWORD if short else SYNCWORD
    payload_size = 128 if short else 256
    body_size = payload_size - 2 if crc else payload_size
    body = bytes((index * 43 + (0x29 if short else 0x71)) & 0xFF for index in range(body_size))
    frame = bytearray(CRC16_ARC.append(body, byteorder="little") if crc else body)
    if crc and not valid_crc:
        frame[-1] ^= 0x80

    rs = CcsdsReedSolomon(basis="conventional", interleaving=1 if short else 2)
    codeword = bytearray(rs.encode(bytes(frame)))
    for index, position in enumerate(rs_error_positions):
        codeword[position] ^= index + 1
    randomized = ccsds_randomize(bytes(codeword))
    encoded = _VITERBI.encode(
        np.unpackbits(np.frombuffer(randomized, dtype=np.uint8)), mode="terminated"
    )
    assert len(encoded) == output_size

    deinterleaved = np.zeros(rows * columns, dtype=np.float64)
    deinterleaved[output_skip : output_skip + output_size] = (
        2.0 * np.asarray(encoded, dtype=np.float64) - 1.0
    )
    capture = deinterleaved.reshape((rows, columns)).T.ravel()
    sync = np.fromiter((char == "1" for char in syncword), dtype=np.uint8)
    capture[np.arange(len(sync)) * step] = 2.0 * sync - 1.0
    if sync_error_positions:
        capture[np.asarray(sync_error_positions) * step] *= -1.0
    prefix = np.asarray([-0.31, 0.22, -0.48, -0.19, 0.37], dtype=np.float64)
    stream = np.concatenate((prefix, capture, np.asarray([-0.2, 0.3])))
    if inverted:
        stream = -stream
    return stream, bytes(frame), prefix.size


@pytest.mark.parametrize("inverted", [False, True])
def test_ao40_fec_exact_pinned_upstream_soft_fixture(inverted: bool) -> None:
    symbols = np.fromfile(_UPSTREAM / "qa_ao40_fec_deframer_symbols.f32", dtype="<f4")
    synchronized = np.fromfile(_UPSTREAM / "qa_ao40_fec_deframer_frame.f32", dtype="<f4")
    assert np.array_equal(symbols[129:5329], synchronized)
    if inverted:
        symbols = -symbols
    decoder = build_decoder("AO40 FEC")
    frames = []
    for start in range(0, symbols.size, 211):
        frames += decoder.push(symbols[start : start + 211])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_upstream_frame_reference()]
    frame = frames[0]
    assert frame.source_start == 129 and frame.source_end == 5329
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.corrected_symbols == 0
    assert frame.metadata["rs_interleaving"] == 2
    assert frame.metadata["matrix_rows"] == 80


@pytest.mark.parametrize("short,label", [(False, "AO-40 FEC"), (True, "AO-40 FEC short")])
@pytest.mark.parametrize("inverted", [False, True])
def test_ao40_fec_generated_long_short_chunks_and_polarity(
    short: bool, label: str, inverted: bool
) -> None:
    stream, expected, prefix_size = _constructed_stream(short=short, inverted=inverted)
    decoder = build_decoder(label)
    frames = []
    for start in range(0, stream.size, 73):
        frames += decoder.push(stream[start : start + 73])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.source_start == prefix_size
    assert frame.source_end == prefix_size + (2652 if short else 5200)
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.metadata["short_frames"] is short


@pytest.mark.parametrize(
    "short,correctable,uncorrectable",
    [
        (True, tuple(range(16)), tuple(range(17))),
        (False, tuple(range(32)), tuple(range(0, 34, 2))),
    ],
)
def test_ao40_fec_rs_correction_boundaries(
    short: bool, correctable: tuple[int, ...], uncorrectable: tuple[int, ...]
) -> None:
    accepted, expected, _ = _constructed_stream(
        short=short, rs_error_positions=correctable
    )
    label = "AO-40 FEC short" if short else "AO-40 FEC"
    frames = build_decoder(label).push(accepted)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].corrected_symbols == len(correctable)

    rejected, _, _ = _constructed_stream(
        short=short, rs_error_positions=uncorrectable
    )
    assert build_decoder(label).push(rejected) == []


def test_ao40_fec_short_crc_sync_threshold_and_truncation() -> None:
    stream, expected, _ = _constructed_stream(
        short=True, crc=True, sync_error_positions=tuple(range(8))
    )
    frames = build_decoder("AO-40 FEC short", {"crc": True}).push(stream)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["crc"] == CRC16_ARC.name
    assert frames[0].metadata["crc_preserved"] is True
    assert frames[0].sync_distance == 8

    bad_crc, _, _ = _constructed_stream(short=True, crc=True, valid_crc=False)
    assert build_decoder("AO-40 FEC short", {"crc": True}).push(bad_crc) == []
    bad_sync, _, _ = _constructed_stream(
        short=True, sync_error_positions=tuple(range(9))
    )
    assert build_decoder("AO-40 FEC short").push(bad_sync) == []

    decoder = build_decoder("AO-40 FEC short")
    assert decoder.push(stream[:-3]) == []
    assert decoder.flush() == []
    with pytest.raises(ValueError):
        build_decoder("AO-40 FEC short", {"sync_threshold": 26})


def test_ao40_fec_registry_contracts_without_production_claim() -> None:
    for label in ("AO-40 FEC", "AO-40 FEC short"):
        profile = REGISTRY.resolve(label)
        assert profile is not None
        assert profile.disposition is DecodeDisposition.IN_PROGRESS
        assert profile.symbol_input is SymbolInput.SOFT_SYMBOLS
        assert profile.decoder_available
        assert not profile.live_supported and not profile.post_pass_supported


@pytest.mark.parametrize(
    ("filename", "capture_sha256", "label", "short", "expected"),
    _SMOGP_CAPTURES,
)
def test_ao40_fec_published_smogp_wavs_replay_byte_exactly(
    filename: str,
    capture_sha256: str,
    label: str,
    short: bool,
    expected: list[tuple[str, int, int, int, int]],
) -> None:
    path = _RECORDINGS / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == capture_sha256
    sample_rate, audio = wavfile.read(path)
    assert sample_rate == 48_000 and audio.dtype == np.int16 and audio.ndim == 1

    symbols, decoded = decode_fsk_audio_mm_profile(
        audio,
        FskAudioConfig(sample_rate, 1_250),
        label,
        cutoff_hz=900,
        transition_hz=100,
        gain_mu=0.05,
        omega_relative_limit=0.01,
    )

    assert symbols.phase_samples == 0
    assert len(decoded) == len(expected)
    for located, (digest, start, end, sample_start, sample_end) in zip(
        decoded, expected, strict=True
    ):
        frame = located.frame
        assert hashlib.sha256(frame.payload).hexdigest() == digest
        assert len(frame.payload) == (128 if short else 256)
        assert (frame.source_start, frame.source_end) == (start, end)
        assert (located.source_sample_start, located.source_sample_end) == (
            sample_start,
            sample_end,
        )
        assert frame.polarity is Polarity.NORMAL
        assert frame.corrected_symbols == 0
        assert frame.metadata["short_frames"] is short
        assert frame.metadata["rs_interleaving"] == (1 if short else 2)


def test_ao40_smogp_manifest_pins_capture_and_published_output_oracles() -> None:
    artifacts = {artifact.artifact_id: artifact for artifact in load_manifest(_MANIFEST)}
    expected = {
        "satrec-smogp-short-wav": (
            "952ddfe53f62a150c53559249c83370630254cab",
            "3df5616c8b66b2ae566369627f4441dbe29cec5a8bf01b200f52852dffe5fdad",
            "Unlicense",
            "real_capture",
        ),
        "satrec-smogp-long-wav": (
            "952ddfe53f62a150c53559249c83370630254cab",
            "0d4641d47330b682174d72cbfe91648da3e5b74b61cff44d733d2063b11c3b7c",
            "Unlicense",
            "real_capture",
        ),
        "smogp-short-published-decode": (
            "34a7d6adc46497d2431b0232b500a52958c6670b",
            "cb5ba05e933a930256e9f42f7a492f84debb3d42d40c1e376b2fb71f2655e710",
            "MIT",
            "independent_oracle",
        ),
        "smogp-long-published-decode": (
            "0d85612ba2498d248c9a599b73ebd5dbcb4c04eb",
            "3e048253d18926129756cbefeb46c131aa7efba5a28e54761299bb988c8a2bf7",
            "MIT",
            "independent_oracle",
        ),
    }
    assert {
        artifact_id: (
            artifacts[artifact_id].source_commit,
            artifacts[artifact_id].sha256,
            artifacts[artifact_id].license,
            artifacts[artifact_id].evidence_class,
        )
        for artifact_id in expected
    } == expected
    assert all(
        "project-authorized mit treatment"
        in artifacts[artifact_id].expected_output.lower()
        for artifact_id in (
            "smogp-short-published-decode",
            "smogp-long-published-decode",
        )
    )

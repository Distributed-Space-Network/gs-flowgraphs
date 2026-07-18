"""NF-MODEM-001 deterministic BPSK/DBPSK file-IQ replay coverage."""

from __future__ import annotations

import ast
from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import BpskConfig, demodulate_bpsk, manchester_sync_symbols
from native_framing.crc import CRC16_CCITT_FALSE
from native_framing.linecode import ccsds_randomize
from native_framing.profiles.ao40 import UNCODED_SYNCWORD
from native_framing.profiles.ao40_fec import SHORT_SYNCWORD
from native_framing.rs import CcsdsReedSolomon
from native_framing.viterbi import ConvolutionalCode
from scipy.signal import resample_poly

from gfsk_ax25 import ax25
from gfsk_ax25 import framing as ax25_framing

_SOURCE_RATE = 48_000.0
_SYMBOL_RATE = 1_200.0
_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM = _ROOT / "related-projects/gr-satellites/python/components/deframers"
_VITERBI = ConvolutionalCode("CCSDS")


def _differential_points(bits: np.ndarray) -> np.ndarray:
    state = 0
    output = [-1.0]
    for bit in bits:
        state ^= int(bit)
        output.append(2.0 * state - 1.0)
    return np.asarray(output)


def _rectangular_bpsk(
    bits: np.ndarray,
    *,
    differential: bool,
    manchester: bool = False,
    symbol_rate_hz: float = _SYMBOL_RATE,
    capture_rate_hz: float = _SOURCE_RATE,
    frequency_offset_hz: float = 533.0,
    phase_radians: float = 0.83,
    guard_samples: int = 2_000,
) -> np.ndarray:
    logical = np.asarray(bits, dtype=np.uint8)
    points = (
        _differential_points(logical)
        if differential
        else logical.astype(np.float64) * 2.0 - 1.0
    )
    if manchester:
        points = np.repeat(points, 2) * np.tile([1.0, -1.0], points.size)
    samples_per_symbol = _SOURCE_RATE / symbol_rate_hz
    assert samples_per_symbol.is_integer()
    samples_per_point = int(samples_per_symbol) // (2 if manchester else 1)
    assert samples_per_point >= 1
    source = np.concatenate(
        (
            np.zeros(guard_samples),
            np.repeat(points, samples_per_point),
            np.zeros(guard_samples),
        )
    )
    if capture_rate_hz != _SOURCE_RATE:
        assert capture_rate_hz == 44_100.0
        source = resample_poly(source, 147, 160)
    time = np.arange(source.size, dtype=np.float64) / capture_rate_hz
    return (
        source * np.exp(1j * (phase_radians + 2.0 * np.pi * frequency_offset_hz * time))
    ).astype(np.complex64)


def _contains(bits: np.ndarray, expected: np.ndarray) -> bool:
    return any(
        np.array_equal(bits[start : start + expected.size], expected)
        for start in range(bits.size - expected.size + 1)
    )


def _upstream_ao40_frame_reference() -> bytes:
    source = ast.parse((_UPSTREAM / "qa_ao40_fec_deframer.py").read_text())
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "frame_reference"
        ):
            return bytes(ast.literal_eval(node.value))
    raise AssertionError("pinned AO-40 QA source has no frame_reference")


def _constructed_ao40_short_symbols() -> tuple[np.ndarray, bytes]:
    body = bytes((index * 43 + 0x29) & 0xFF for index in range(128))
    rs = CcsdsReedSolomon(basis="conventional", interleaving=1)
    randomized = ccsds_randomize(rs.encode(body))
    encoded = np.asarray(
        _VITERBI.encode(
            np.unpackbits(np.frombuffer(randomized, dtype=np.uint8)), mode="terminated"
        ),
        dtype=np.uint8,
    )
    assert encoded.size == 2_572

    rows, columns = 51, 52
    deinterleaved = np.zeros(rows * columns, dtype=np.float64)
    deinterleaved[80 : 80 + encoded.size] = 2.0 * encoded - 1.0
    symbols = deinterleaved.reshape((rows, columns)).T.ravel()
    sync = np.fromiter((char == "1" for char in SHORT_SYNCWORD), dtype=np.uint8)
    symbols[np.arange(sync.size) * rows] = 2.0 * sync - 1.0
    return symbols, body


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_manchester_sync_matches_pinned_gr_satellites_qa_construction(
    complex_input: bool, offset: bool
) -> None:
    bits = 2 * np.random.default_rng(0x40).integers(0, 2, 4_096) - 1
    encoded = np.repeat(bits, 2) * np.tile([1, -1], bits.size)
    if offset:
        encoded[:-1] = encoded[1:]
    if complex_input:
        encoded = encoded.astype(np.complex128)
    recovered = manchester_sync_symbols(encoded, block_size=32)
    if offset:
        assert np.array_equal(recovered[1:], bits[1:])
    else:
        assert np.array_equal(recovered, bits)


@pytest.mark.parametrize("differential", [False, True])
@pytest.mark.parametrize("manchester", [False, True])
@pytest.mark.parametrize("capture_rate_hz", [_SOURCE_RATE, 44_100.0])
def test_bpsk_replay_recovers_fractional_clock_cfo_phase_and_differential_bits(
    differential: bool, manchester: bool, capture_rate_hz: float
) -> None:
    expected = np.random.default_rng(0xB5).integers(0, 2, 500, dtype=np.uint8)
    iq = _rectangular_bpsk(
        expected,
        differential=differential,
        manchester=manchester,
        capture_rate_hz=capture_rate_hz,
    )
    replay = demodulate_bpsk(
        iq,
        BpskConfig(
            capture_rate_hz,
            _SYMBOL_RATE,
            differential=differential,
            manchester=manchester,
        ),
    )
    # Multiply-conjugate DBPSK is positive for no transition, the complement
    # of the modulus-2 input bit. Coherent BPSK retains a global Costas ambiguity.
    target = 1 - expected if differential and not manchester else expected
    assert _contains(replay.hard_bits, target) or _contains(replay.hard_bits, 1 - target)
    assert replay.estimated_frequency_hz == pytest.approx(533.0, abs=2.0)
    assert 0 <= replay.phase_samples < np.ceil(capture_rate_hz / _SYMBOL_RATE)
    assert np.all(np.diff(replay.sample_boundaries) > 0)


@pytest.mark.parametrize(
    ("modulation", "differential", "manchester"),
    [
        ("bpsk", False, False),
        ("dbpsk", True, False),
        ("BPSK Manchester", False, True),
        ("DBPSK Manchester", True, True),
    ],
)
def test_postpass_bpsk_ax25_file_iq_replay_preserves_payload_and_source_time(
    tmp_path: Path, modulation: str, differential: bool, manchester: bool
) -> None:
    body = ax25.encode_ui(dest="APRS", src="N0CALL", info=b"native-bpsk-replay")
    bits = ax25_framing.encode(body, scramble=False, nrzi=True)
    capture = _rectangular_bpsk(
        bits, differential=differential, manchester=manchester
    )
    path = tmp_path / f"{modulation}.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_SOURCE_RATE,
        symbol_rate_hz=_SYMBOL_RATE,
        framings_to_try=("AX.25",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation=modulation,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [body]
    assert records[0]["framing"] == "ax25"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["source_sample_offset"] > 0
    assert records[0]["timestamp"].startswith("2026-01-01T00:00:")


def test_postpass_dbpsk_manchester_routes_ao40_uncoded_profile(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 2
    wire = CRC16_CCITT_FALSE.append(payload, byteorder="big")
    sync = np.fromiter((char == "1" for char in UNCODED_SYNCWORD), dtype=np.uint8)
    bits = np.concatenate((sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))
    capture = _rectangular_bpsk(bits, differential=True, manchester=True)
    path = tmp_path / "ao40-dbpsk-manchester.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_SOURCE_RATE,
        symbol_rate_hz=_SYMBOL_RATE,
        framings_to_try=("AO-40 uncoded",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="DBPSK Manchester",
        window_s=6.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [payload]
    assert records[0]["framing"] == "ao40_uncoded"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


def test_postpass_dbpsk_manchester_routes_ao40_fec_soft_fixture(tmp_path: Path) -> None:
    soft_fixture = np.fromfile(
        _UPSTREAM / "qa_ao40_fec_deframer_symbols.f32", dtype="<f4"
    )
    hard_bits = np.asarray(soft_fixture >= 0, dtype=np.uint8)
    capture = _rectangular_bpsk(
        hard_bits,
        differential=True,
        manchester=True,
        symbol_rate_hz=400.0,
    )
    path = tmp_path / "ao40-fec-dbpsk-manchester.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_SOURCE_RATE,
        symbol_rate_hz=400.0,
        framings_to_try=("AO-40 FEC",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="DBPSK Manchester",
        window_s=15.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [
        _upstream_ao40_frame_reference()
    ]
    assert records[0]["framing"] == "ao40_fec"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["source_sample_offset"] > 0


def test_postpass_dbpsk_manchester_routes_ao40_fec_short_profile(
    tmp_path: Path,
) -> None:
    soft_symbols, expected = _constructed_ao40_short_symbols()
    capture = _rectangular_bpsk(
        np.asarray(soft_symbols >= 0, dtype=np.uint8),
        differential=True,
        manchester=True,
        symbol_rate_hz=400.0,
    )
    path = tmp_path / "ao40-fec-short-dbpsk-manchester.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_SOURCE_RATE,
        symbol_rate_hz=400.0,
        framings_to_try=("AO-40 FEC short",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="DBPSK Manchester",
        window_s=10.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "ao40_fec_short"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


def test_bpsk_replay_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        BpskConfig(3_000, 1_000)
    with pytest.raises(ValueError, match="at least 4"):
        BpskConfig(7_000, 1_000, manchester=True)
    with pytest.raises(ValueError, match="positive integer"):
        BpskConfig(48_000, 1_200, manchester_block_size=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        demodulate_bpsk(np.zeros((4, 2)), BpskConfig(48_000, 1_200))
    with pytest.raises(ValueError, match="quarter sample rate"):
        BpskConfig(48_000, 1_200, frequency_offset_hz=12_000)
    short = demodulate_bpsk(
        np.ones(100, dtype=np.complex64), BpskConfig(48_000, 1_200)
    )
    assert short.sample_boundaries[-1] <= 100

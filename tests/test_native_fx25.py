"""Construction tests for the Astrocast-compatible native FX.25 profile."""

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_X25
from native_framing.linecode import nrzi_encode, reflect_bytes
from native_framing.profiles.fx25 import CAPTURE_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PREFIX = np.asarray([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
_RS = CcsdsReedSolomon(basis="dual", interleaving=1)
_PAYLOAD = bytes((index * 19 + 3) % 0x7D for index in range(91))
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 9_600.0


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


def _stream(
    *,
    nrzi: bool,
    line_inverted: bool = False,
    payload: bytes = _PAYLOAD,
    rs_error_positions: tuple[int, ...] = (),
    valid_crc: bool = True,
    closing_flag: bool = True,
    sync_error_positions: tuple[int, ...] = (),
) -> np.ndarray:
    framed = bytearray(CRC16_X25.append(payload, byteorder="little"))
    if not valid_crc:
        framed[0] ^= 0x80
    inner = b"\x7e" + bytes(framed) + (b"\x7e" if closing_flag else b"\x55")
    inner = inner.ljust(223, b"\x55")
    codeword = bytearray(_RS.encode(inner))
    for index, position in enumerate(rs_error_positions):
        codeword[position] ^= index + 1
    wire = reflect_bytes(bytes(codeword))
    sync = _SYNC.copy()
    if sync_error_positions:
        sync[list(sync_error_positions)] ^= 1
    logical = np.concatenate(
        (_PREFIX, sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8)))
    )
    line = nrzi_encode(logical, initial=1) if nrzi else logical
    return 1 - line if line_inverted else line


@pytest.mark.parametrize("nrzi", [False, True])
@pytest.mark.parametrize("line_inverted", [False, True])
def test_fx25_nrz_nrzi_chunks_polarity_offsets_and_metadata(
    nrzi: bool, line_inverted: bool
) -> None:
    stream = _stream(nrzi=nrzi, line_inverted=line_inverted)
    decoder = build_decoder("FX25 NRZI", {"nrzi": nrzi})
    frames = []
    for start in range(0, stream.size, 37):
        frames += decoder.push(stream[start : start + 37])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.source_start == _PREFIX.size
    assert frame.source_end == _PREFIX.size + len(SYNCWORD) + CAPTURE_SIZE * 8
    expected_polarity = (
        Polarity.AMBIGUOUS
        if nrzi
        else (Polarity.INVERTED if line_inverted else Polarity.NORMAL)
    )
    assert frame.polarity is expected_polarity
    assert frame.metadata["nrzi"] is nrzi
    assert frame.metadata["line_polarity_unobservable"] is nrzi
    assert frame.metadata["rs_basis"] == "dual"
    assert frame.metadata["byte_reflection"] is True


def test_fx25_rs_boundary_crc_and_closing_flag_gates() -> None:
    corrected = build_decoder("FX.25 NRZI", {"nrzi": False}).push(
        _stream(nrzi=False, rs_error_positions=tuple(range(16)))
    )
    assert [frame.payload for frame in corrected] == [_PAYLOAD]
    assert corrected[0].corrected_symbols == 16
    assert build_decoder("FX.25 NRZI", {"nrzi": False}).push(
        _stream(nrzi=False, rs_error_positions=tuple(range(17)))
    ) == []


@pytest.mark.parametrize("nrzi", [False, True])
def test_fx25_fsk_file_iq_replay_routes_parameterized_nrz_and_nrzi_profiles(
    tmp_path: Path, nrzi: bool
) -> None:
    path = tmp_path / f"astrocast-fx25-{'nrzi' if nrzi else 'nrz'}-fsk.cf32"
    _fsk_capture(_stream(nrzi=nrzi)).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("FX.25 NRZI",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        native_evaluation=True,
        framing_parameters={
            "modulation": "fsk",
            "mod_index": 0.8,
            "nrzi": nrzi,
        },
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [_PAYLOAD]
    assert records[0]["framing"] == "fx25_nrzi"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert build_decoder("FX.25 NRZI", {"nrzi": False}).push(
        _stream(nrzi=False, valid_crc=False)
    ) == []
    assert build_decoder("FX.25 NRZI", {"nrzi": False}).push(
        _stream(nrzi=False, closing_flag=False)
    ) == []


def test_fx25_sync_threshold_configuration_cross_rejection_and_flush() -> None:
    accepted = _stream(nrzi=True, sync_error_positions=tuple(range(8)))
    frames = build_decoder("Astrocast FX.25 NRZ-I").push(accepted)
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].sync_distance == 8
    rejected = _stream(nrzi=True, sync_error_positions=tuple(range(9)))
    assert build_decoder("FX.25 NRZI").push(rejected) == []

    nrz = _stream(nrzi=False)
    assert build_decoder("FX.25 NRZI", {"nrzi": True}).push(nrz) == []

    decoder = build_decoder("FX.25 NRZI")
    complete = _stream(nrzi=True)
    assert decoder.push(complete[:-1]) == []
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(complete)] == [_PAYLOAD]
    with pytest.raises(ValueError):
        build_decoder("FX.25 NRZI", {"sync_threshold": 32})


def test_fx25_registry_contract_without_generic_or_production_claim() -> None:
    profile = REGISTRY.resolve("Astrocast FX.25 NRZ")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert "Astrocast" in profile.integrity_policy or "Astrocast" in profile.output_semantics
    assert not profile.live_supported and not profile.post_pass_supported

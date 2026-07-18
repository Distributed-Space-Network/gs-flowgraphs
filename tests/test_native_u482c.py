"""Generated construction tests for the native U482C profile."""

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.codes.golay24 import encode_golay24
from native_framing.linecode import ccsds_randomize
from native_framing.profiles.u482c import CAPTURE_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity
from native_framing.viterbi import ConvolutionalCode

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PAYLOAD = bytes((index * 13 + 0x42) & 0xFF for index in range(48))
_RS = CcsdsReedSolomon(basis="conventional", interleaving=1)
_VITERBI = ConvolutionalCode("NASA-DSN uninverted")
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
    viterbi: bool,
    randomize: bool,
    rs: bool,
    header_errors: tuple[int, ...] = (),
    packet_errors: tuple[int, ...] = (),
    payload: bytes = _PAYLOAD,
    declared_length: int | None = None,
) -> np.ndarray:
    packet = _RS.encode(payload) if rs else payload
    if randomize:
        packet = ccsds_randomize(packet)
    if viterbi:
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        encoded = _VITERBI.encode(bits, mode="terminated")
        packet = bytes(np.packbits(np.asarray(encoded, dtype=np.uint8), bitorder="big"))
    damaged = bytearray(packet)
    for index, position in enumerate(packet_errors):
        damaged[position] ^= index + 1
    length = len(damaged) if declared_length is None else declared_length
    flags = (int(viterbi) << 8) | (int(randomize) << 9) | (int(rs) << 10)
    header = encode_golay24((length & 0xFF) | flags)
    for position in header_errors:
        header ^= 1 << position
    capture = header.to_bytes(3, "big") + bytes(damaged)
    capture = capture[:CAPTURE_SIZE].ljust(CAPTURE_SIZE, b"\x00")
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(capture, dtype=np.uint8))))


@pytest.mark.parametrize(
    "viterbi,randomize,rs",
    [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("inverted", [False, True])
def test_u482c_header_selected_stage_matrix_and_inversion(
    viterbi: bool, randomize: bool, rs: bool, inverted: bool
) -> None:
    stream = _stream(viterbi=viterbi, randomize=randomize, rs=rs)
    if inverted:
        stream = 1 - stream
    decoder = build_decoder("GOMspace U482C")
    frames = []
    for start in range(0, stream.size, 37):
        frames += decoder.push(stream[start : start + 37])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.metadata["viterbi"] is viterbi
    assert frame.metadata["randomizer"] == ("CCSDS" if randomize else "none")
    assert frame.metadata["rs"] is rs
    assert frame.integrity is (IntegrityStatus.PASSED if rs else IntegrityStatus.NOT_PRESENT)
    assert frame.corrected_symbols == (0 if rs else None)


def test_u482c_golay_and_rs_correction_metadata() -> None:
    frames = build_decoder("U482C").push(
        _stream(
            viterbi=False,
            randomize=False,
            rs=True,
            header_errors=(0, 11, 23),
            packet_errors=tuple(range(0, 48, 3)),
        )
    )
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].metadata["golay_corrected_bits"] == 3
    assert frames[0].corrected_symbols == 16

    assert build_decoder("U482C").push(
        _stream(
            viterbi=False,
            randomize=False,
            rs=True,
            packet_errors=tuple(range(20)),
        )
    ) == []


def test_u482c_fsk_file_iq_replay_routes_full_header_selected_native_chain(
    tmp_path: Path,
) -> None:
    wire = _stream(viterbi=True, randomize=True, rs=True)
    path = tmp_path / "u482c-full-fsk.cf32"
    _fsk_capture(wire).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("U482C",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="fsk",
        mod_index=0.8,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [_PAYLOAD]
    assert records[0]["framing"] == "u482c"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


def test_u482c_without_rs_reports_absent_integrity_and_emits_mutation() -> None:
    mutated = bytearray(_PAYLOAD)
    mutated[5] ^= 0x80
    frames = build_decoder("U482C").push(
        _stream(viterbi=False, randomize=False, rs=False, payload=bytes(mutated))
    )
    assert [frame.payload for frame in frames] == [bytes(mutated)]
    assert frames[0].integrity is IntegrityStatus.NOT_PRESENT
    assert frames[0].metadata["false_positive_policy"].startswith("explicit-profile")


def test_u482c_rejects_invalid_lengths_truncation_and_parameters() -> None:
    assert build_decoder("U482C").push(
        _stream(viterbi=False, randomize=False, rs=False, declared_length=0)
    ) == []
    assert build_decoder("U482C").push(
        _stream(viterbi=True, randomize=False, rs=False, declared_length=2)
    ) == []

    decoder = build_decoder("U482C")
    assert decoder.push(_stream(viterbi=True, randomize=True, rs=True)[:-1]) == []
    assert decoder.flush() == []
    with pytest.raises(ValueError):
        build_decoder("U482C", {"sync_threshold": 33})


def test_u482c_is_available_without_production_or_completion_claim() -> None:
    profile = REGISTRY.resolve("U482C")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported and not profile.post_pass_supported

"""Official-vector and construction tests for native NGHam receive profiles."""

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.codes.ngham import (
    NON_RS_SIZES,
    PARITY_SIZES,
    RS_SIZES,
    SIZE_TAGS,
    classify_ngham_size,
    encode_ngham_rs,
)
from native_framing.crc import CRC16_X25
from native_framing.linecode import ccsds_randomize
from native_framing.profiles.ngham import SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PREFIX = np.asarray([0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
_OFFICIAL_TEST_FRAME = bytes.fromhex(
    "aaaaaaaa5de62a7e3b49cde71c4b93ce5968bc8e2c93ada7b746ce5a977dcc32"
    "a2bf3e0a10f18894cdeae0f7f92426d158630b25683caf9794d5"
)
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
    size_index: int,
    *,
    rs: bool = False,
    inverted: bool = False,
    padding: int = 7,
    tag_error_bits: tuple[int, ...] = (),
    rs_error_positions: tuple[int, ...] = (),
    valid_crc: bool = True,
    sync_errors: tuple[int, ...] = (),
) -> tuple[np.ndarray, bytes]:
    non_rs_size = NON_RS_SIZES[size_index]
    application_size = non_rs_size - padding - 3
    if application_size < 0:
        raise ValueError("test packet does not fit selected NGHam size")
    header = (0xA0 | padding) & 0xFF
    body = bytes([header]) + bytes(
        (index * 17 + size_index * 23) & 0xFF for index in range(application_size)
    )
    packet = bytearray(CRC16_X25.append(body, byteorder="big"))
    if not valid_crc:
        packet[1] ^= 0x80
    padded = bytes(packet).ljust(non_rs_size, b"\x00")
    size = classify_ngham_size(SIZE_TAGS[size_index].to_bytes(3, "big"))
    assert size is not None
    codeword = bytearray(
        encode_ngham_rs(padded, size)
        if rs
        else padded.ljust(RS_SIZES[size_index], b"\x00")
    )
    for index, position in enumerate(rs_error_positions):
        codeword[position] ^= index + 1

    tag = SIZE_TAGS[size_index]
    for bit in tag_error_bits:
        tag ^= 1 << bit
    capture = tag.to_bytes(3, "big") + ccsds_randomize(bytes(codeword))
    sync = _SYNC.copy()
    if sync_errors:
        sync[list(sync_errors)] ^= 1
    stream = np.concatenate((_PREFIX, sync, np.unpackbits(np.frombuffer(capture, dtype=np.uint8))))
    if inverted:
        stream = 1 - stream
    return stream, body


def test_ngham_exact_official_reference_vector() -> None:
    assert len(_OFFICIAL_TEST_FRAME) == 58
    stream = np.unpackbits(np.frombuffer(_OFFICIAL_TEST_FRAME[4:], dtype=np.uint8))
    frames = build_decoder("NGHam").push(stream)
    assert [frame.payload for frame in frames] == [b"\x18TEST"]
    frame = frames[0]
    assert frame.source_start == 0 and frame.source_end == stream.size
    assert frame.corrected_symbols == 0
    assert frame.metadata["rs"] is True
    assert frame.metadata["rs_parity_symbols"] == 16
    assert frame.metadata["padding"] == 24


@pytest.mark.parametrize(
    ("label", "canonical", "bits", "expected"),
    [
        (
            "NGHam",
            "ngham",
            np.unpackbits(np.frombuffer(_OFFICIAL_TEST_FRAME[4:], dtype=np.uint8)),
            b"\x18TEST",
        ),
        (
            "NGHam no Reed Solomon",
            "ngham_no_rs",
            *_stream(3),
        ),
    ],
)
def test_ngham_fsk_file_iq_replay_routes_official_rs_and_generated_no_rs_profiles(
    tmp_path: Path,
    label: str,
    canonical: str,
    bits: np.ndarray,
    expected: bytes,
) -> None:
    path = tmp_path / f"{canonical}-fsk.cf32"
    _fsk_capture(bits).tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=(label,),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="fsk",
        mod_index=0.8,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == canonical
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


@pytest.mark.parametrize("size_index", range(7))
@pytest.mark.parametrize("inverted", [False, True])
def test_ngham_no_rs_all_size_tags_chunks_polarity_and_metadata(
    size_index: int, inverted: bool
) -> None:
    stream, expected = _stream(size_index, inverted=inverted)
    decoder = build_decoder("NGHam no RS")
    frames = []
    for start in range(0, stream.size, 37):
        frames += decoder.push(stream[start : start + 37])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.source_start == _PREFIX.size
    assert frame.source_end == stream.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.metadata["size_index"] == size_index
    assert frame.metadata["rs"] is False
    assert frame.metadata["rs_slot_size"] == RS_SIZES[size_index]
    assert frame.metadata["non_rs_size"] == NON_RS_SIZES[size_index]
    assert frame.metadata["padding"] == 7


@pytest.mark.parametrize("size_index", range(7))
@pytest.mark.parametrize("inverted", [False, True])
def test_ngham_rs_all_sizes_chunks_polarity_and_metadata(
    size_index: int, inverted: bool
) -> None:
    stream, expected = _stream(size_index, rs=True, inverted=inverted)
    decoder = build_decoder("NGHam")
    frames = []
    for start in range(0, stream.size, 41):
        frames += decoder.push(stream[start : start + 41])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.source_start == _PREFIX.size and frame.source_end == stream.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.corrected_symbols == 0
    assert frame.metadata["rs"] is True
    assert frame.metadata["rs_parity_symbols"] == PARITY_SIZES[size_index]


def test_ngham_size_tag_six_bit_boundary_and_threshold() -> None:
    tag = SIZE_TAGS[0]
    for bit in range(6):
        tag ^= 1 << bit
    classified = classify_ngham_size(tag.to_bytes(3, "big"))
    assert classified is not None
    assert classified.index == 0 and classified.tag_distance == 6

    stream, expected = _stream(0, tag_error_bits=tuple(range(6)))
    frames = build_decoder("NGHam no Reed Solomon").push(stream)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["size_tag_distance"] == 6
    assert build_decoder("NGHam no RS", {"tag_threshold": 0}).push(stream) == []

    seven_errors = SIZE_TAGS[0]
    for bit in range(7):
        seven_errors ^= 1 << bit
    assert classify_ngham_size(seven_errors.to_bytes(3, "big")) is None


def test_ngham_crc_zero_padding_sync_and_truncation_fail_closed() -> None:
    invalid_crc, _ = _stream(2, valid_crc=False)
    assert build_decoder("NGHam no RS").push(invalid_crc) == []

    zero_padding, expected = _stream(2, padding=0)
    frames = build_decoder("NGHam no RS").push(zero_padding)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["padding"] == 0

    accepted, _ = _stream(2, sync_errors=(0, 1, 2, 3))
    rejected, _ = _stream(2, sync_errors=(0, 1, 2, 3, 4))
    assert len(build_decoder("NGHam no RS").push(accepted)) == 1
    assert build_decoder("NGHam no RS").push(rejected) == []

    stream, expected = _stream(2)
    decoder = build_decoder("NGHam no RS")
    assert decoder.push(stream[:-1]) == []
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(stream)] == [expected]


@pytest.mark.parametrize(
    "size_index,correctable,uncorrectable",
    [
        (0, tuple(range(8)), tuple(range(9))),
        (3, tuple(range(16)), tuple(range(17))),
    ],
)
def test_ngham_rs16_rs32_correction_boundaries(
    size_index: int,
    correctable: tuple[int, ...],
    uncorrectable: tuple[int, ...],
) -> None:
    accepted, expected = _stream(
        size_index, rs=True, rs_error_positions=correctable
    )
    frames = build_decoder("NGHam").push(accepted)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].corrected_symbols == len(correctable)

    rejected, _ = _stream(size_index, rs=True, rs_error_positions=uncorrectable)
    assert build_decoder("NGHam").push(rejected) == []


def test_ngham_registry_contract_and_profile_separation() -> None:
    no_rs = REGISTRY.resolve("NGHam no Reed Solomon")
    with_rs = REGISTRY.resolve("NGHam")
    assert no_rs is not None and with_rs is not None and no_rs is not with_rs
    assert no_rs.disposition is DecodeDisposition.IN_PROGRESS and no_rs.decoder_available
    assert with_rs.disposition is DecodeDisposition.IN_PROGRESS and with_rs.decoder_available
    assert not no_rs.live_supported and not no_rs.post_pass_supported
    with pytest.raises(ValueError):
        build_decoder("NGHam no RS", {"tag_threshold": 7})

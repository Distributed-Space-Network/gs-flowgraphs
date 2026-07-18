"""Construction and pinned-behavior tests for the native OpenLST profile."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.codes.openlst import (
    decode_openlst_fec,
    encode_openlst_fec,
    interleave_openlst_chunk,
)
from native_framing.crc import CRC16_CC11XX
from native_framing.linecode import pn9_bytes
from native_framing.profiles.openlst import CAPTURE_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PREAMBLE = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
_PAYLOAD = bytes((index * 29 + 0x37) & 0xFF for index in range(93))


def _capture(
    payload: bytes = _PAYLOAD,
    *,
    fec_error_bits: tuple[int, ...] = (),
    valid_crc: bool = True,
) -> np.ndarray:
    body = bytes([len(payload) + 2]) + payload
    frame = bytearray(CRC16_CC11XX.append(body, byteorder="little"))
    if not valid_crc:
        frame[1] ^= 0x80
    encoded = np.unpackbits(
        np.frombuffer(
            encode_openlst_fec(pn9_bytes(bytes(frame)), encoded_size=CAPTURE_SIZE),
            dtype=np.uint8,
        )
    )
    if fec_error_bits:
        encoded[list(fec_error_bits)] ^= 1
    return encoded


def _stream(
    payload: bytes = _PAYLOAD,
    *,
    fec_error_bits: tuple[int, ...] = (),
    valid_crc: bool = True,
    sync_errors: tuple[int, ...] = (),
) -> np.ndarray:
    sync = _SYNC.copy()
    if sync_errors:
        sync[list(sync_errors)] ^= 1
    return np.concatenate(
        (_PREAMBLE, sync, _capture(payload, fec_error_bits=fec_error_bits, valid_crc=valid_crc))
    )


def test_openlst_fec_literal_vector_interleave_and_error_correction() -> None:
    raw = b"OpenLST"
    encoded = bytes.fromhex("1bd53b50625643e08319f8b00c0a033a00000000")
    assert encode_openlst_fec(raw) == encoded
    assert decode_openlst_fec(encoded) == raw
    assert interleave_openlst_chunk(bytes.fromhex("01234567")) == bytes.fromhex("dd508850")
    interleaved = interleave_openlst_chunk(bytes.fromhex("deadbeef"))
    assert interleave_openlst_chunk(interleaved) == bytes.fromhex("deadbeef")

    damaged = bytearray(encoded)
    damaged[5] ^= 0x80
    assert decode_openlst_fec(bytes(damaged)) == raw

    with pytest.raises(ValueError):
        decode_openlst_fec(b"\x00" * 3)
    with pytest.raises(ValueError):
        encode_openlst_fec(raw, encoded_size=16)


@pytest.mark.parametrize("step", [1, 37, 521, 8192])
@pytest.mark.parametrize("inverted", [False, True])
def test_openlst_chunks_polarity_offsets_crc_crop_and_metadata(step: int, inverted: bool) -> None:
    stream = _stream()
    if inverted:
        stream = 1 - stream
    decoder = build_decoder("Open LST")
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.source_start == _PREAMBLE.size
    assert frame.source_end == stream.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.metadata["declared_length"] == len(_PAYLOAD) + 3
    assert frame.metadata["crc"] == CRC16_CC11XX.name
    assert frame.metadata["crc_byteorder"] == "little"


def test_openlst_fec_correction_crc_gate_and_sync_threshold() -> None:
    corrected = build_decoder("OpenLST").push(_stream(fec_error_bits=(40, 381, 1900)))
    assert [frame.payload for frame in corrected] == [_PAYLOAD]
    assert build_decoder("OpenLST").push(_stream(valid_crc=False)) == []

    assert len(build_decoder("OpenLST").push(_stream(sync_errors=(0, 1, 2, 3)))) == 1
    assert build_decoder("OpenLST").push(_stream(sync_errors=(0, 1, 2, 3, 4))) == []


def test_openlst_length_bounds_truncation_flush_and_parameters() -> None:
    maximum_payload = bytes(index & 0xFF for index in range(253))
    assert [frame.payload for frame in build_decoder("OpenLST").push(_stream(maximum_payload))] == [
        maximum_payload
    ]
    assert [frame.payload for frame in build_decoder("OpenLST").push(_stream(b""))] == [b""]

    decoder = build_decoder("OpenLST")
    assert decoder.push(_stream()[:-1]) == []
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(_stream())] == [_PAYLOAD]
    with pytest.raises(ValueError):
        build_decoder("OpenLST", {"sync_threshold": 33})


def test_openlst_is_available_without_production_or_completion_claim() -> None:
    profile = REGISTRY.resolve("OpenLST")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported and not profile.post_pass_supported

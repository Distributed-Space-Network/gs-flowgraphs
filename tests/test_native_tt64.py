"""Generated construction tests for native TT-64 framing."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_ARC
from native_framing.profiles.tt64 import PARITY_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

from gfsk_ax25.reedsolomon import RSCodec

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PAYLOAD = bytes((index * 19 + 7) & 0xFF for index in range(46))
_DATA = CRC16_ARC.append(_PAYLOAD, byteorder="little")
_CODEC = RSCodec(PARITY_SIZE, prim=0x11D, fcr=1, generator=2)
_WIRE = _CODEC.encode(_DATA)
_STREAM = np.concatenate((_SYNC, np.unpackbits(np.frombuffer(_WIRE, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 15, 16, 127, 4096])
@pytest.mark.parametrize("inverted", [False, True])
def test_tt64_chunk_polarity_rs_crc_and_offsets(step: int, inverted: bool) -> None:
    stream = 1 - _STREAM if inverted else _STREAM
    decoder = build_decoder("TT64")
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].corrected_symbols == 0
    assert frames[0].source_start == 0
    assert frames[0].source_end == stream.size
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)


def test_tt64_corrects_eight_rs_symbols_then_checks_crc() -> None:
    corrupted = bytearray(_WIRE)
    for index in range(8):
        corrupted[index * 5] ^= index + 1
    bits = np.concatenate((_SYNC, np.unpackbits(np.frombuffer(bytes(corrupted), dtype=np.uint8))))
    frames = build_decoder("TT-64").push(bits)
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    assert frames[0].corrected_symbols == 8

    crc_bad_data = bytearray(_DATA)
    crc_bad_data[-1] ^= 1
    crc_bad_wire = _CODEC.encode(crc_bad_data)
    crc_bad = np.concatenate(
        (_SYNC, np.unpackbits(np.frombuffer(crc_bad_wire, dtype=np.uint8)))
    )
    assert build_decoder("TT-64").push(crc_bad) == []


def test_tt64_rejects_uncorrectable_threshold_truncation_and_bad_parameters() -> None:
    corrupted = bytearray(_WIRE)
    for index in range(17):
        corrupted[index * 3] ^= 0xA5
    bits = np.concatenate((_SYNC, np.unpackbits(np.frombuffer(bytes(corrupted), dtype=np.uint8))))
    assert build_decoder("TT-64").push(bits) == []

    sync_bad = _STREAM.copy()
    sync_bad[[0, 1]] ^= 1
    assert build_decoder("TT-64").push(sync_bad) == []

    decoder = build_decoder("TT-64")
    assert decoder.push(_STREAM[:-1]) == []
    assert decoder.flush() == []
    with pytest.raises(ValueError):
        build_decoder("TT-64", {"sync_threshold": 17})


def test_tt64_is_available_without_overclaiming_completion() -> None:
    profile = REGISTRY.resolve("TT-64")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported

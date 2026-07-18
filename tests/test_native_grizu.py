"""Generated construction tests for the pinned Grizu-263A receive chain."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_CC11XX
from native_framing.linecode import pn9_bytes, reflect_bytes
from native_framing.profiles.grizu import FRAME_SIZE, SCRAMBLER_SEED, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)


def _wire(payload: bytes = b"grizu-payload") -> bytes:
    packet_without_crc = bytes([len(payload) + 3, 0xAA, 0x55]) + payload + b"\x7e"
    packet = CRC16_CC11XX.append(packet_without_crc, byteorder="big")
    decoded_capture = packet + bytes(FRAME_SIZE - len(packet))
    # Inverse of reflect -> PN9(seed 0x100) -> reflect.
    return reflect_bytes(pn9_bytes(reflect_bytes(decoded_capture), seed=SCRAMBLER_SEED))


def _stream(wire: bytes) -> np.ndarray:
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 17, 63, 257, 4096])
@pytest.mark.parametrize("inverted", [False, True])
def test_grizu_chunk_polarity_transform_order_and_crc(step: int, inverted: bool):
    expected = b"grizu-payload"
    stream = _stream(_wire(expected))
    if inverted:
        stream = 1 - stream
    decoder = build_decoder("Grizu-263A")
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frames[0].metadata["pn9_seed"] == 0x100
    assert frames[0].metadata["reflection_stages"] == 2


def test_grizu_rejects_crc_length_threshold_and_truncation():
    corrupted = bytearray(_wire())
    corrupted[10] ^= 0x01
    assert build_decoder("Grizu-263A").push(_stream(bytes(corrupted))) == []

    decoded = bytearray(
        reflect_bytes(pn9_bytes(reflect_bytes(_wire()), seed=SCRAMBLER_SEED))
    )
    decoded[0] = 0xFF
    malformed = reflect_bytes(pn9_bytes(reflect_bytes(decoded), seed=SCRAMBLER_SEED))
    assert build_decoder("Grizu-263A").push(_stream(malformed)) == []

    bad_sync = _stream(_wire())
    bad_sync[:9] ^= 1
    assert build_decoder("Grizu-263A").push(bad_sync) == []

    decoder = build_decoder("Grizu-263A")
    assert decoder.push(_stream(_wire())[:-1]) == []
    assert decoder.flush() == []


def test_grizu_is_available_without_payload_schema_overclaim():
    profile = REGISTRY.resolve("Grizu 263A")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported

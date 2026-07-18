"""Generated construction tests for the shared PN9/CC11xx profile family.

These tests exercise native composition and boundaries.  They intentionally do
not claim the independent mission vectors still required by NF-FRM-021/028.
"""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_CC11XX, CRC16_X25, CrcSpec
from native_framing.linecode import pn9_bytes
from native_framing.profiles.cc11xx import FRAME_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)


def _wire(spec: CrcSpec, byteorder: str, payload: bytes = b"mission-payload") -> bytes:
    without_crc = bytes([len(payload) + 3, 0xAA, 0x55]) + payload + b"\x7e"
    packet = spec.append(without_crc, byteorder=byteorder)
    decoded_capture = packet + bytes(FRAME_SIZE - len(packet))
    return pn9_bytes(decoded_capture)


def _bits(wire: bytes) -> np.ndarray:
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


@pytest.mark.parametrize(
    ("label", "spec", "byteorder"),
    [
        ("Reaktor Hello World", CRC16_CC11XX, "big"),
        ("AALTO-1", CRC16_X25, "little"),
    ],
)
@pytest.mark.parametrize("step", [1, 17, 31, 256, 4096])
@pytest.mark.parametrize("inverted", [False, True])
def test_cc11xx_profiles_chunk_polarity_and_crop(
    label: str, spec: CrcSpec, byteorder: str, step: int, inverted: bool
):
    expected = b"mission-payload"
    stream = _bits(_wire(spec, byteorder, expected))
    if inverted:
        stream = 1 - stream
    decoder = build_decoder(label)
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frames[0].metadata["packet_length"] == len(expected) + 6


@pytest.mark.parametrize(
    ("label", "spec", "byteorder"),
    [
        ("Reaktor Hello World", CRC16_CC11XX, "big"),
        ("AALTO-1", CRC16_X25, "little"),
    ],
)
def test_cc11xx_profiles_reject_crc_length_and_truncation(
    label: str, spec: CrcSpec, byteorder: str
):
    wire = bytearray(_wire(spec, byteorder))
    wire[8] ^= 0x01
    assert build_decoder(label).push(_bits(bytes(wire))) == []

    # After PN9 reversal this claims a packet longer than the capture.
    decoded = bytearray(pn9_bytes(_wire(spec, byteorder)))
    decoded[0] = 0xFF
    malformed = pn9_bytes(bytes(decoded))
    assert build_decoder(label).push(_bits(malformed)) == []

    decoder = build_decoder(label)
    assert decoder.push(_bits(_wire(spec, byteorder))[:-5]) == []
    assert decoder.flush() == []


def test_cc11xx_profiles_are_available_but_not_claimed_complete():
    for label in ("Reaktor Hello World", "AALTO-1"):
        profile = REGISTRY.resolve(label)
        assert profile is not None
        assert profile.disposition is DecodeDisposition.IN_PROGRESS
        assert profile.decoder_available
        assert not profile.live_supported
        assert not profile.post_pass_supported

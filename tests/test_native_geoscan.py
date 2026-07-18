"""Out-of-bench tests for the first new native hard-bit profile."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_CC11XX
from native_framing.linecode import pn9_bytes
from native_framing.profiles.geoscan import SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)

# Static construction vector for the pinned upstream chain parameters:
# decoded packet bytes 00..3f + big-endian CRC-16/CC11XX, then GNU Radio /
# TinyGS-compatible PN9.  This proves deterministic stage composition but is
# not claimed as the independent/public vector still required by NF-FRM-009.
_PAYLOAD = bytes(range(64))
_WIRE = bytes.fromhex(
    "ffe01f99e9803523e273d8327c9a5905446c3fcb7918ac987f40ddb9a329d407"
    "1072b1fcb6c98132a2f5dead7963360e70f5f6e6f2a4bcfadfe874320e2ae1bcbcd9"
)


def _stream(wire: bytes = _WIRE) -> np.ndarray:
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 7, 8, 17, 31, 64, 257, 4096])
@pytest.mark.parametrize("inverted", [False, True])
def test_geoscan_clean_vector_is_chunk_and_polarity_invariant(step: int, inverted: bool):
    bits = _stream()
    if inverted:
        bits = 1 - bits
    decoder = build_decoder("GEOSCAN")
    frames = []
    for start in range(0, bits.size, step):
        frames += decoder.push(bits[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    frames += decoder.flush()
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.integrity is IntegrityStatus.PASSED
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.source_start == 0
    assert frame.source_end == bits.size
    assert frame.sync_distance == 0


def test_geoscan_crc_mutation_and_truncation_are_rejected():
    corrupted = bytearray(_WIRE)
    corrupted[20] ^= 0x01
    assert build_decoder("GEOSCAN").push(_stream(bytes(corrupted))) == []

    truncated = _stream()[:-1]
    decoder = build_decoder("GEOSCAN")
    assert decoder.push(truncated) == []
    assert decoder.flush() == []
    assert decoder.retained_symbols == 0


def test_geoscan_sync_threshold_edge_and_one_beyond():
    edge = _stream()
    edge[[0, 4, 9, 31]] ^= 1
    frames = build_decoder("GEOSCAN").push(edge)
    assert len(frames) == 1 and frames[0].sync_distance == 4

    beyond = edge.copy()
    beyond[14] ^= 1
    assert build_decoder("GEOSCAN").push(beyond) == []


def test_geoscan_parameterized_frame_size_and_fail_closed_parameters():
    payload = b"12345678"
    decoded = CRC16_CC11XX.append(payload, byteorder="big")
    wire = pn9_bytes(decoded)
    frames = build_decoder("GeoScan", {"frame_size": 10, "sync_threshold": 0}).push(
        _stream(wire)
    )
    assert [frame.payload for frame in frames] == [payload]
    with pytest.raises(ValueError, match="<= 258"):
        build_decoder("GEOSCAN", {"frame_size": 259})
    with pytest.raises(ValueError, match="<= 32"):
        build_decoder("GEOSCAN", {"sync_threshold": 33})


def test_geoscan_registry_disposition_does_not_overclaim_completion():
    profile = REGISTRY.resolve("GEOSCAN")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported

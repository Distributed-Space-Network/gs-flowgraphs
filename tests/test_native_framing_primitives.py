"""Independent and boundary tests for Phase-1 native framing primitives."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing.crc import (
    CRC16_ARC,
    CRC16_CC11XX,
    CRC16_CCITT_FALSE,
    CRC16_X25,
    CrcSpec,
)
from native_framing.crop import cc11xx_packet, fixed, head_tail
from native_framing.linecode import (
    differential_decode,
    differential_encode,
    pn9_bytes,
    reflect_bytes,
)
from native_framing.sync import StreamingSync
from native_framing.types import Polarity, SymbolInput


def test_hard_sync_exact_threshold_edge_and_one_beyond():
    pattern = "10010011000010110101000111011110"
    clean = np.fromiter((char == "1" for char in pattern), dtype=np.uint8)
    edge = clean.copy()
    edge[[0, 5, 17, 31]] ^= 1
    beyond = edge.copy()
    beyond[9] ^= 1

    assert StreamingSync(pattern, threshold=4).push(clean)[0].distance == 0
    match = StreamingSync(pattern, threshold=4).push(edge)
    assert len(match) == 1 and match[0].distance == 4
    assert StreamingSync(pattern, threshold=4).push(beyond) == []


def test_sync_is_invariant_to_every_single_chunk_boundary():
    pattern = "1100101011110001"
    stream = np.array([0, 0, *map(int, pattern), 1, 0], dtype=np.uint8)
    for split in range(stream.size + 1):
        sync = StreamingSync(pattern)
        matches = sync.push(stream[:split]) + sync.push(stream[split:])
        matches += sync.flush()
        assert [(match.source_start, match.source_end) for match in matches] == [(2, 18)]
        assert sync.retained_symbols == 0


def test_sync_reports_overlap_without_duplicate_reemission():
    sync = StreamingSync("10101")
    first = sync.push([1, 0, 1])
    second = sync.push([0, 1, 0, 1])
    third = sync.push([])
    assert first == []
    assert [(match.source_start, match.source_end) for match in second] == [(0, 5), (2, 7)]
    assert third == []
    assert sync.retained_symbols <= sync.max_retained_symbols == 4


@pytest.mark.parametrize("scale", [0.25, 1.0, 12.0])
def test_soft_sync_distance_is_scale_invariant_and_detects_inversion(scale: float):
    pattern = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    symbols = (pattern * 2.0 - 1.0) * scale
    normal = StreamingSync(
        pattern, threshold=0.001, symbol_input=SymbolInput.SOFT_SYMBOLS, accept_inverted=True
    ).push(symbols)
    inverted = StreamingSync(
        pattern, threshold=0.001, symbol_input=SymbolInput.SOFT_SYMBOLS, accept_inverted=True
    ).push(-symbols)
    assert normal[0].distance == pytest.approx(0)
    assert normal[0].polarity is Polarity.NORMAL
    assert inverted[0].distance == pytest.approx(0)
    assert inverted[0].polarity is Polarity.INVERTED


def test_sync_input_validation_and_random_noise_false_positive_bound():
    with pytest.raises(ValueError, match="binary"):
        StreamingSync("10x1")
    with pytest.raises(ValueError, match="threshold"):
        StreamingSync("101", threshold=4)
    with pytest.raises(ValueError, match="only 0 and 1"):
        StreamingSync("101").push([0, 2, 1])
    with pytest.raises(ValueError, match="finite"):
        StreamingSync("101", symbol_input=SymbolInput.SOFT_SYMBOLS).push([1.0, np.nan])

    # Exact 64-bit sync in this fixed 100k-bit noise corpus has no hit.  This
    # is a deterministic regression corpus, not a statistical protocol claim.
    noise = np.random.default_rng(0x51C).integers(0, 2, 100_000, dtype=np.uint8)
    sync = StreamingSync("1001001100001011010100011101111011010010110011110010110101000011")
    assert sync.push(noise) == []


def test_pn9_matches_pinned_gnuradio_and_tinygs_sequence_and_is_involutive():
    # GNU Radio lfsr(mask=0x21, seed=0x1ff, reg_len=8), packed LSB first;
    # independently the pinned TinyGS BitCode::pn9 yields the same sequence.
    expected = bytes.fromhex("ff e1 1d 9a ed 85 33 24 ea 7a d2 39 70 97 57 0a")
    assert pn9_bytes(bytes(16)) == expected
    payload = bytes(range(128))
    assert pn9_bytes(pn9_bytes(payload)) == payload
    with pytest.raises(ValueError, match="seed"):
        pn9_bytes(payload, seed=0)


def test_reflection_and_differential_vectors():
    assert reflect_bytes(bytes([0x00, 0x01, 0x96, 0xFF])) == bytes([0x00, 0x80, 0x69, 0xFF])
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    encoded = differential_encode(bits, initial=1)
    assert differential_decode(encoded, initial=1).tolist() == bits.tolist()
    with pytest.raises(ValueError, match="one-dimensional"):
        differential_encode(np.zeros((2, 2), dtype=np.uint8))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (CRC16_CCITT_FALSE, 0x29B1),
        (CRC16_X25, 0x906E),
        (CRC16_ARC, 0xBB3D),
        (CRC16_CC11XX, 0xAEE7),
    ],
)
def test_crc_catalog_published_check_values(spec: CrcSpec, expected: int):
    assert spec.compute(b"123456789") == expected


@pytest.mark.parametrize("byteorder", ["big", "little"])
def test_crc_append_check_strip_and_mutation(byteorder: str):
    payload = b"native framing crc"
    frame = CRC16_CCITT_FALSE.append(payload, byteorder=byteorder)
    assert CRC16_CCITT_FALSE.strip_if_valid(frame, byteorder=byteorder) == payload
    corrupted = bytearray(frame)
    corrupted[3] ^= 0x01
    assert CRC16_CCITT_FALSE.strip_if_valid(corrupted, byteorder=byteorder) is None
    assert CRC16_CCITT_FALSE.strip_if_valid(b"x", byteorder=byteorder) is None
    with pytest.raises(ValueError, match="byteorder"):
        CRC16_CCITT_FALSE.strip_if_valid(frame, byteorder="middle")


def test_crc_spec_rejects_malformed_parameters():
    with pytest.raises(ValueError, match="multiple of eight"):
        CrcSpec("bad", 7, 1, 0, 0, False, False)
    with pytest.raises(ValueError, match="does not fit"):
        CrcSpec("bad", 8, 0x101, 0, 0, False, False)


def test_cc11xx_and_head_tail_crop_bounds():
    packet = bytes([3]) + b"abc" + b"\x12\x34" + b"ignored"
    assert cc11xx_packet(packet) == bytes([3]) + b"abc" + b"\x12\x34"
    assert cc11xx_packet(bytes([20]) + b"short") is None
    assert cc11xx_packet(b"") is None
    assert cc11xx_packet(packet, maximum=5) is None
    assert head_tail(b"abcdef", head=2, tail=1) == b"cde"
    assert head_tail(b"abc", head=2, tail=2) is None
    assert fixed(b"abcdef", size=4, maximum=4) == b"abcd"
    assert fixed(b"abc", size=4, maximum=4) is None
    with pytest.raises(ValueError, match="bounds"):
        fixed(b"abc", size=5, maximum=4)

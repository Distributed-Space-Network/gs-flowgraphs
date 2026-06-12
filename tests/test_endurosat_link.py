"""EnduroSat chip-packet link tests: framing, CRC, and TX->RX round trips."""

from __future__ import annotations

import struct

import numpy as np
from scipy.signal import resample_poly

from gfsk_ax25 import endurosat_link as el

_TX_SR = 153_600.0  # 16 samples/symbol at 9600
_CAPTURE_SR = 128_000.0  # the VSA capture rate (13.33 sps) we must also handle


def test_frame_structure_and_crc():
    payload = bytes(range(32))
    raw = el.frame_bytes(payload, preamble_len=5)
    assert raw[:5] == b"\xaa\xaa\xaa\xaa\xaa"  # preamble
    assert raw[5] == 0x7E  # sync
    assert raw[6] == len(payload)  # length byte
    assert raw[7 : 7 + 32] == payload
    body = raw[6 : 7 + 32]  # length + payload
    assert raw[-2:] == struct.pack(">H", el.crc16(body))  # CRC-16/CCITT-FALSE, big-endian


def test_deframe_roundtrip_and_multiframe():
    a = bytes(range(20))
    b = bytes([0xAA, 0x55] * 16)  # 32 bytes incl. byte values that look like preamble
    bits = np.concatenate([el.frame_bits(a), el.frame_bits(b)])
    got = el.deframe(bits)
    assert a in got
    assert b in got


def test_deframe_rejects_corruption():
    raw = bytearray(el.frame_bytes(b"hello endurosat payload"))
    raw[10] ^= 0xFF  # corrupt a payload byte
    bits = np.unpackbits(np.frombuffer(bytes(raw), dtype=np.uint8))
    assert el.deframe(bits) == []  # CRC catches it


def _awgn(iq, sigma, seed):
    rng = np.random.default_rng(seed)
    return (iq + rng.normal(0, sigma, len(iq)) + 1j * rng.normal(0, sigma, len(iq))).astype(
        np.complex64
    )


def _with_guard(iq, n=512):
    """A real burst slice has settling room around it (the captures include a few
    ms of guard); receive() is always called that way, so the tests mirror it."""
    return np.concatenate([np.zeros(n, np.complex64), iq, np.zeros(n, np.complex64)])


def test_tx_rx_roundtrip_clean():
    payload = b"AIRMAC-encrypted-blob-or-anything"
    iq = _with_guard(el.transmit(payload, _TX_SR))
    assert payload in el.receive(iq, _TX_SR)


def test_tx_rx_roundtrip_awgn():
    payload = bytes(range(40))
    iq = _awgn(_with_guard(el.transmit(payload, _TX_SR)), sigma=0.08, seed=3)
    assert payload in el.receive(iq, _TX_SR)


def test_stream_decoder_segments_bursts():
    a = b"frame-A-payload"
    b = b"frame-B-different-payload"
    gap = np.zeros(2000, dtype=np.complex64)
    iq = np.concatenate(
        [_with_guard(el.transmit(a, _TX_SR)), gap, _with_guard(el.transmit(b, _TX_SR)), gap]
    ).astype(np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    third = len(iq) // 3
    out: list[bytes] = []
    dec.push(iq[:third])
    out += dec.decode_new()
    dec.push(iq[third : 2 * third])
    out += dec.decode_new()
    dec.push(iq[2 * third :])
    out += dec.flush()
    assert a in out
    assert b in out
    assert len(out) == len(set(out)) == 2


def test_rx_at_capture_rate_with_cfo():
    # The hard case the captures pose: transmit at a clean 16 sps, then resample
    # to the 128 kHz VSA rate (non-integer sps), add a carrier offset + noise,
    # and confirm the capture-robust receive() recovers it.
    payload = bytes((np.arange(48) * 7 % 256).astype(np.uint8).tolist())
    iq = resample_poly(_with_guard(el.transmit(payload, _TX_SR)), 5, 6).astype(np.complex64)
    n = np.arange(len(iq))
    cfo = np.exp(1j * 2 * np.pi * 1500.0 * n / _CAPTURE_SR)  # +1.5 kHz carrier offset
    iq = _awgn((iq * cfo).astype(np.complex64), sigma=0.05, seed=9)
    assert payload in el.receive(iq, _CAPTURE_SR)

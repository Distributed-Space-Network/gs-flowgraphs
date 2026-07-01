"""docs/08 Tier 3 — KISS / SLIP TNC framing (byte-oriented, reversible)."""
from __future__ import annotations

import framings
import numpy as np

from gfsk_ax25 import kiss


def test_kiss_roundtrip():
    frame = b"\x01\x02\x03hello"
    frames = kiss.kiss_decode(kiss.kiss_encode(frame))
    assert frames == [frame]


def test_kiss_escapes_reserved_bytes():
    frame = bytes([kiss.FEND, kiss.FESC, 0x41, kiss.FEND])  # payload contains the delimiters
    wire = kiss.kiss_encode(frame)
    assert kiss.FEND not in wire[1:-1]  # delimiters only at the frame ends
    assert kiss.kiss_decode(wire) == [frame]


def test_slip_roundtrip_and_escapes():
    frame = bytes([0x00, kiss.FEND, 0xDB, kiss.FESC, 0xFF])
    assert kiss.slip_decode(kiss.slip_encode(frame)) == [frame]


def test_multiple_frames():
    stream = kiss.kiss_encode(b"AAA") + kiss.kiss_encode(b"BBB")
    assert kiss.kiss_decode(stream) == [b"AAA", b"BBB"]


def test_through_framing_registry():
    frame = b"payload-\xc0-\xdb-end"
    wire = kiss.kiss_encode(frame)
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8))
    frames, matched = framings.deframe(bits, "kiss")
    assert matched == "kiss" and frames == [frame]

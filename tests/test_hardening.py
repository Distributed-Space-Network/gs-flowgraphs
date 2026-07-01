"""docs/08 — hardening regressions from the post-build rigorous recheck.

These lock in edge cases the per-module tests didn't cover: exhaustive RS single-error correction,
exhaustive BCH single+double correction, CCSDS behaviour under a mismatched RS flag / truncation /
back-to-back frames, and the Morse single-element timing ambiguity.
"""
from __future__ import annotations

import numpy as np

from gfsk_ax25 import argos, ccsds, morse
from gfsk_ax25.reedsolomon import RSCodec


def test_rs_corrects_every_single_symbol_position():
    rs = RSCodec(32)
    msg = bytes(range(223))
    cw = rs.encode(msg)
    for pos in range(255):
        for val in (0x01, 0x7F, 0xFF):
            c = bytearray(cw)
            c[pos] ^= val
            assert rs.decode(bytes(c)) == msg


def test_bch_exhaustive_single_and_double():
    for m in (0, 1, 0xFFFFF, 0x155555, 0xAAAAA):
        c = argos.bch3121_encode(m)
        for i in range(31):
            assert argos.bch3121_decode(c ^ (1 << i)) == m
        for i in range(31):
            for j in range(i + 1, 31):
                assert argos.bch3121_decode(c ^ (1 << i) ^ (1 << j)) == m


_H = ccsds.TMHeader(0, 0x100, 1, 0, 5, 6, 0, 0, 0)


def test_ccsds_rs_false_recovers_systematic_payload_of_clean_frame():
    # RS is systematic, so the message bytes are in the clear even without RS decoding; the FECF
    # still validates a clean frame. (This documents the graceful behaviour under a wrong rs flag.)
    data = b"PAYLOAD-CHECK-1234567890"
    b = ccsds.build_tm_frame(_H, data, frame_len=223, rs=True, randomize=True, fecf=True)
    frames = ccsds.deframe_tm(b, frame_len=223, rs=False)
    assert len(frames) == 1 and frames[0][6:6 + len(data)] == data


def test_ccsds_rs_false_rejects_when_message_byte_corrupted():
    data = b"PAYLOAD-CHECK-1234567890"
    b = ccsds.build_tm_frame(_H, data, frame_len=223, rs=True, randomize=True, fecf=True)
    by = bytearray(np.packbits(b))
    by[4 + 10] ^= 0xFF  # corrupt a message byte (RS parity would fix it, but rs=False can't)
    noisy = np.unpackbits(np.frombuffer(bytes(by), dtype=np.uint8))
    assert ccsds.deframe_tm(noisy, frame_len=223, rs=False) == []      # FECF rejects
    assert ccsds.deframe_tm(noisy, frame_len=223, rs=True)[0][6:6 + len(data)] == data  # RS fixes


def test_ccsds_truncated_frame_is_safe():
    b = ccsds.build_tm_frame(_H, b"X" * 50, frame_len=223)
    assert ccsds.deframe_tm(b[:-40], frame_len=223) == []  # no crash, no partial frame


def test_ccsds_back_to_back_frames_all_recovered():
    stream = np.concatenate(
        [ccsds.build_tm_frame(_H, bytes([i]) * 10, frame_len=223) for i in range(3)])
    frames = ccsds.deframe_tm(stream, frame_len=223)
    assert len(frames) == 3 and [f[6] for f in frames] == [0, 1, 2]


def test_ccsds_build_rejects_oversized_data():
    import pytest
    with pytest.raises(ValueError):
        ccsds.build_tm_frame(_H, b"Z" * 300, frame_len=223)


def test_morse_single_element_needs_explicit_unit():
    # 'T' (one dash) vs 'E' (one dot) is indistinguishable without a time reference: the auto unit
    # estimate treats the lone element as a dot. Supplying unit= resolves it.
    assert morse.decode(morse.encode("T")) == "E"                    # documented ambiguity
    assert morse.decode(morse.encode("T", unit=4), unit=4) == "T"    # disambiguated
    # any message containing a dot estimates the unit correctly:
    assert morse.decode(morse.encode("THE QUICK BROWN FOX 123")) == "THE QUICK BROWN FOX 123"

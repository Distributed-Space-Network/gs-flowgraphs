"""docs/08 Tier 1 — Argos PTT/PMT-A3: rigorous BCH(31,21) codec + deframer structure.

The BCH codec is checked as coding theory (systematic property, corrects all ≤2-bit errors,
rejects 3-error miscorrection). The deframer is checked structurally: given a known sync it
locates frames, BCH-decodes the platform ID, and rejects noise (the BCH syndrome is the FCS).
"""
from __future__ import annotations

import numpy as np
import pytest

from gfsk_ax25 import argos


def test_bch_is_systematic():
    for m in (0, 1, 0x1FFFFF, 0x155555):
        code = argos.bch3121_encode(m)
        assert code >> 10 == m  # message occupies the high 21 bits
        assert code < (1 << 31)


def test_bch_corrects_all_single_and_double_errors():
    rng = np.random.default_rng(7)
    for _ in range(500):
        m = int(rng.integers(0, 1 << 21))
        code = argos.bch3121_encode(m)
        for weight in (0, 1, 2):
            r = code
            for p in rng.choice(31, size=weight, replace=False):
                r ^= 1 << int(p)
            assert argos.bch3121_decode(r) == m


def test_bch_does_not_miscorrect_triple_errors():
    rng = np.random.default_rng(9)
    miscorrect = 0
    for _ in range(500):
        m = int(rng.integers(0, 1 << 21))
        code = argos.bch3121_encode(m)
        r = code
        for p in rng.choice(31, size=3, replace=False):
            r ^= 1 << int(p)
        if argos.bch3121_decode(r) == m:
            miscorrect += 1
    assert miscorrect == 0  # 3 errors is beyond t=2 → never silently corrected to the original


def _frame_bits(sync, sync_bits, msg21):
    s = [(sync >> (sync_bits - 1 - i)) & 1 for i in range(sync_bits)]
    code = argos.bch3121_encode(msg21)
    c = [(code >> (31 - 1 - i)) & 1 for i in range(31)]
    return np.array(s + c, dtype=np.uint8)


# A realistic long frame-sync (24 bits) is the false-alarm gate — an 8-bit sync matches noise
# far too often (BCH alone accepts ~48% of randoms, so it cannot backstop a short sync).
_SYNC, _SYNC_BITS = 0xABCDEF, 24


def test_deframe_locates_frame_and_recovers_id():
    msg = 0x123AB
    stream = np.concatenate([
        np.zeros(30, dtype=np.uint8),  # clean lead (0x000000 ≠ the 24-bit sync)
        _frame_bits(_SYNC, _SYNC_BITS, msg),
        np.zeros(20, dtype=np.uint8),
    ])
    frames = argos.deframe(stream, sync=_SYNC, sync_bits=_SYNC_BITS)
    assert frames
    assert int.from_bytes(frames[0][:3], "big") == msg


def test_deframe_corrects_two_bit_errors_in_the_id_field():
    msg = 0x0FACE
    bits = _frame_bits(_SYNC, _SYNC_BITS, msg)
    bits[_SYNC_BITS + 3] ^= 1  # two errors inside the 31-bit BCH codeword
    bits[_SYNC_BITS + 17] ^= 1
    frames = argos.deframe(bits, sync=_SYNC, sync_bits=_SYNC_BITS)
    assert frames and int.from_bytes(frames[0][:3], "big") == msg


def test_a_long_sync_gates_out_noise():
    # With a 24-bit sync, accidental syncs in random bits are ~N/2^24 → effectively none, so no
    # spurious frames escape (the sync length, NOT the BCH syndrome, is the frame gate).
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(50):
        noise = rng.integers(0, 2, 400).astype(np.uint8)
        hits += len(argos.deframe(noise, sync=_SYNC, sync_bits=_SYNC_BITS))
    assert hits == 0


@pytest.mark.parametrize("bad", [1 << 21, -1])
def test_bch_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        argos.bch3121_encode(bad)

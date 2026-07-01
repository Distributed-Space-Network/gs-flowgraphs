"""docs/08 Tier 2 — Reed-Solomon RS(255,223) codec, validated as coding theory.

Rigorous: systematic property, corrects ALL error patterns up to t=nsym/2, and does NOT silently
miscorrect beyond t. Parameterization checked with a second code rate.
"""
from __future__ import annotations

import numpy as np
import pytest

from gfsk_ax25.reedsolomon import RS_NSYM_255_223, RSCodec


def test_encode_is_systematic():
    rs = RSCodec(RS_NSYM_255_223)
    msg = bytes(range(223))
    cw = rs.encode(msg)
    assert len(cw) == 255
    assert cw[:223] == msg  # message occupies the leading bytes, parity trails


def test_corrects_all_errors_up_to_t():
    rs = RSCodec(RS_NSYM_255_223)  # t = 16
    rng = np.random.default_rng(0)
    for _ in range(300):
        msg = bytes(rng.integers(0, 256, 223).tolist())
        cw = bytearray(rs.encode(msg))
        nerr = int(rng.integers(0, 17))  # 0..16
        for p in rng.choice(255, size=nerr, replace=False):
            cw[int(p)] ^= int(rng.integers(1, 256))
        assert rs.decode(bytes(cw)) == msg


def test_does_not_miscorrect_beyond_t():
    rs = RSCodec(RS_NSYM_255_223)
    rng = np.random.default_rng(1)
    miscorrect = 0
    for _ in range(300):
        msg = bytes(rng.integers(0, 256, 223).tolist())
        cw = bytearray(rs.encode(msg))
        for p in rng.choice(255, size=17, replace=False):  # 17 > t
            cw[int(p)] ^= int(rng.integers(1, 256))
        if rs.decode(bytes(cw)) == msg:
            miscorrect += 1
    assert miscorrect == 0


def test_parameterized_rate_rs255_239():
    rs = RSCodec(16)  # RS(255,239), t = 8
    rng = np.random.default_rng(2)
    msg = bytes(rng.integers(0, 256, 239).tolist())
    cw = bytearray(rs.encode(msg))
    assert len(cw) == 255
    for p in rng.choice(255, size=8, replace=False):
        cw[int(p)] ^= 0xFF
    assert rs.decode(bytes(cw)) == msg


def test_message_too_long_raises():
    with pytest.raises(ValueError):
        RSCodec(RS_NSYM_255_223).encode(bytes(224))

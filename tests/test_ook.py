"""docs/08 Tier 3 — OOK / ASK envelope keying (numpy codec)."""
from __future__ import annotations

import numpy as np

from gfsk_ax25 import ook


def test_ook_roundtrip_clean():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8)
    iq = ook.modulate(bits, sps=8)
    out = ook.demodulate(iq, sps=8)
    np.testing.assert_array_equal(out, bits)


def test_ook_roundtrip_with_noise_and_gain():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 200).astype(np.uint8)
    iq = ook.modulate(bits, sps=10, amp=3.0)
    iq = iq + (rng.normal(0, 0.3, iq.shape) + 1j * rng.normal(0, 0.3, iq.shape))
    out = ook.demodulate(iq, sps=10)
    assert np.mean(out == bits) > 0.99  # adaptive threshold + gain independence


def test_mask_4level_roundtrip():
    syms = np.array([0, 1, 2, 3, 3, 2, 1, 0, 2, 1], dtype=np.uint8)
    iq = ook.modulate(syms, sps=6, amp=1.0, levels=4)
    out = ook.demodulate(iq, sps=6, levels=4)
    np.testing.assert_array_equal(out, syms)


def test_flat_input_yields_no_symbols_set():
    iq = np.ones(80, dtype=np.complex128)
    assert not ook.demodulate(iq, sps=8).any()

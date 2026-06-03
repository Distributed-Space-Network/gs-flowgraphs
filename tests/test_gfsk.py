"""GFSK modem tests: modulate -> (channel) -> demodulate bit recovery."""

from __future__ import annotations

import numpy as np

from gfsk_ax25.gfsk import GfskParams, demodulate, modulate

_SR = 99_840.0  # 8 samples/symbol at 12 480 sym/s


def _params() -> GfskParams:
    return GfskParams(sample_rate_hz=_SR, symbol_rate_hz=12_480.0)


def _min_ber(tx: np.ndarray, rx: np.ndarray, *, search: int = 4) -> float:
    """BER under a small alignment search (the demod drops edge symbols and may
    sit one symbol off; framing handles this for real frames via flag sync)."""
    best = 1.0
    for off in range(-search, search + 1):
        a, b = (tx[off:], rx) if off >= 0 else (tx, rx[-off:])
        m = min(len(a), len(b))
        if m < 64:
            continue
        a2, b2 = a[5 : m - 5], b[5 : m - 5]
        best = min(best, float(np.mean(a2 != b2)))
    return best


def _awgn(iq: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, len(iq)) + 1j * rng.normal(0, sigma, len(iq))
    return (iq + noise).astype(np.complex64)


def test_perfect_channel():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    iq = modulate(bits, _params())
    rx = demodulate(iq, _params(), recover_timing=False)
    assert _min_ber(bits, rx) < 0.005


def test_awgn_channel():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    iq = modulate(bits, _params())
    rx = demodulate(_awgn(iq, 0.25, seed=11), _params(), recover_timing=False)
    assert _min_ber(bits, rx) < 0.02


def test_frequency_offset_doppler():
    # An FSK discriminator turns a carrier/Doppler offset into a DC bias, which
    # the demod's slow-mean removal cancels. 2.5 kHz ~= UHF LEO residual.
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 3000).astype(np.uint8)
    iq = modulate(bits, _params())
    n = np.arange(len(iq))
    iq_off = (iq * np.exp(1j * 2 * np.pi * 2_500.0 * n / _SR)).astype(np.complex64)
    rx = demodulate(iq_off, _params(), recover_timing=False)
    assert _min_ber(bits, rx) < 0.02


def test_gardner_timing_recovery():
    # Fractional-sample delay + AWGN; Gardner must still track the clock.
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    iq = modulate(bits, _params())
    # Fractional delay of 0.37 samples via linear interpolation.
    delay = 0.37
    idx = np.arange(len(iq)) - delay
    re = np.interp(idx, np.arange(len(iq)), iq.real)
    im = np.interp(idx, np.arange(len(iq)), iq.imag)
    iq_d = _awgn((re + 1j * im).astype(np.complex64), 0.2, seed=5)
    rx = demodulate(iq_d, _params(), recover_timing=True)
    assert _min_ber(bits, rx) < 0.03

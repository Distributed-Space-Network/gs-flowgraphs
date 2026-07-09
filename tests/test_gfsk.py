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


def test_estimate_cfo_is_amplitude_weighted():
    # estimate_cfo_rad must be AMPLITUDE-weighted (angle of the mean lag-1 product), not a mean of
    # per-sample angles. On a real capture the wanted signal is a loud burst sitting in long quiet
    # gaps; a mean-of-angles estimate weights every low-amplitude gap sample equally and drifts onto
    # whatever tiny tone lives in the gaps (measured as a bogus +18 kHz CFO on cmd_107). The
    # amplitude-weighted form lets the loud burst dominate.
    from gfsk_ax25.gfsk import estimate_cfo_rad

    fs = 96_000.0
    n = np.arange(6000)
    burst = np.exp(2j * np.pi * 5000.0 * n / fs).astype(np.complex64)  # loud tone at +5 kHz
    rng = np.random.default_rng(0)
    gap = (0.01 * (rng.normal(0, 1, 3000) + 1j * rng.normal(0, 1, 3000))).astype(np.complex64)
    sig = np.concatenate([gap, burst, gap]).astype(np.complex64)
    cfo_hz = estimate_cfo_rad(sig) * fs / (2 * np.pi)
    assert abs(cfo_hz - 5000.0) < 100.0  # locks the loud burst, not the noisy gaps


def test_channel_filter_recovers_preamble_beside_carrier():
    # The cmd_107 failure mode: a STRONG continuous carrier tens of kHz off-channel captures the
    # wideband FM discriminator (which has no channel filter of its own), corrupting every symbol.
    # demodulate_capture(channel_bw_hz=...) low-passes at DC first and recovers the signal — shown
    # here by the 0xAA preamble surviving intact only WITH the filter.
    from gfsk_ax25.gfsk import GfskParams, demodulate_capture

    fs, baud = 96_000.0, 9600.0
    pre = modulate(np.tile([1, 0], 400).astype(np.uint8),  # 800-bit 0xAA preamble at DC
                   GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    n = np.arange(len(pre))
    rx = (pre + 10.0 * np.exp(2j * np.pi * 25_000.0 * n / fs)).astype(np.complex64)  # 10x @ +25 kHz
    no_filter = demodulate_capture(rx, fs, symbol_rate_hz=baud, recover_timing=False)
    filtered = demodulate_capture(rx, fs, symbol_rate_hz=baud, channel_bw_hz=16_000.0,
                                  recover_timing=False)
    from iq_analyze import _longest_alt_run

    assert _longest_alt_run(no_filter) < 20  # interferer shreds the preamble (noise-level run)
    assert _longest_alt_run(filtered) > 500  # channel filter recovers the long clean preamble

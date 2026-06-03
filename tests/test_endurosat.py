"""End-to-end EnduroSat UHF link: AX.25 frame -> GFSK IQ -> channel -> frame.

Proves the whole stack (AX.25/HDLC/NRZI/G3RUH/GFSK) is self-consistent and
decodable under the impairments a real UHF LEO downlink sees: AWGN, residual
Doppler, and clock/timing offset. This is the working verification of our
spec interpretation, since we cannot test against the radio itself.
"""

from __future__ import annotations

import numpy as np

from gfsk_ax25 import ax25, endurosat

_SR = 99_840.0  # 8 samples/symbol


def _frame() -> bytes:
    return ax25.encode_ui(dest="DSN0", src="ES1", info=b"BEACON volt=7.4 temp=21 mode=NOMINAL")


def _awgn(iq: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (iq + rng.normal(0, sigma, len(iq)) + 1j * rng.normal(0, sigma, len(iq))).astype(
        np.complex64
    )


def test_occupied_bandwidth_matches_symbol_rate():
    # Carson's rule cross-checks 12 480 sym/s @ h=0.5 against the spec's 18.7 kHz.
    dev = endurosat.MOD_INDEX * endurosat.SYMBOL_RATE_HZ / 2.0
    carson = 2.0 * (dev + endurosat.SYMBOL_RATE_HZ / 2.0)
    assert abs(carson - endurosat.OCCUPIED_BANDWIDTH_HZ) / endurosat.OCCUPIED_BANDWIDTH_HZ < 0.02


def test_clean_link():
    body = _frame()
    iq = endurosat.transmit(body, _SR)
    frames = endurosat.receive(iq, _SR, recover_timing=False)
    assert body in frames
    ui = ax25.decode_ui(frames[0])
    assert ui is not None and ui.src == "ES1" and ui.info.startswith(b"BEACON")


def test_link_with_awgn():
    body = _frame()
    iq = _awgn(endurosat.transmit(body, _SR), sigma=0.18, seed=3)
    assert body in endurosat.receive(iq, _SR, recover_timing=False)


def test_link_with_doppler_offset():
    body = _frame()
    iq = endurosat.transmit(body, _SR)
    n = np.arange(len(iq))
    iq = (iq * np.exp(1j * 2 * np.pi * 3_000.0 * n / _SR)).astype(np.complex64)
    assert body in endurosat.receive(iq, _SR, recover_timing=False)


def test_link_with_timing_and_awgn():
    body = _frame()
    iq = endurosat.transmit(body, _SR)
    idx = np.arange(len(iq)) - 0.41  # fractional clock phase
    iq = (np.interp(idx, np.arange(len(iq)), iq.real)
          + 1j * np.interp(idx, np.arange(len(iq)), iq.imag)).astype(np.complex64)
    iq = _awgn(iq, sigma=0.12, seed=9)
    assert body in endurosat.receive(iq, _SR, recover_timing=True)


def test_stream_decoder_emits_each_frame_once():
    a = ax25.encode_ui(dest="DSN0", src="ES1", info=b"frame-A")
    b = ax25.encode_ui(dest="DSN0", src="ES1", info=b"frame-B-different")
    gap = np.zeros(2000, dtype=np.complex64)
    iq = np.concatenate(
        [endurosat.transmit(a, _SR), gap, endurosat.transmit(b, _SR), gap]
    ).astype(np.complex64)
    dec = endurosat.StreamDecoder(_SR, recover_timing=False)
    # Feed in three chunks; only new frames come back, none duplicated.
    third = len(iq) // 3
    emitted: list[bytes] = []
    dec.push(iq[:third])
    emitted += dec.decode_new()
    dec.push(iq[third : 2 * third])
    emitted += dec.decode_new()
    dec.push(iq[2 * third :])
    emitted += dec.flush()
    assert a in emitted
    assert b in emitted
    assert len(emitted) == len(set(emitted)) == 2


def test_unscrambled_profile_roundtrip():
    # If telemetry later shows the link is NOT G3RUH-scrambled, the toggle works.
    profile = endurosat.LinkProfile(scramble=False, nrzi=True)
    body = _frame()
    iq = endurosat.transmit(body, _SR, profile=profile)
    assert body in endurosat.receive(iq, _SR, profile=profile, recover_timing=False)

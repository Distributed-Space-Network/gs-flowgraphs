"""2-GFSK modulator + demodulator (numpy/scipy).

Continuous-phase 2-level Gaussian FSK, parameterised by symbol rate, modulation
index ``h`` and Gaussian ``BT``. For the EnduroSat UHF link the spec gives a
12 480 sym/s channel at ~18.7 kHz occupied bandwidth, which is consistent with
``h ~= 0.5`` (Carson: 2*(h*Rs/2 + Rs/2) = 1.5*Rs ~= 18.7 kHz).

The demodulator is a classic noncoherent chain: phase-difference discriminator
-> slow-mean removal (kills residual carrier/Doppler offset, which an FSK
discriminator turns into a DC bias) -> Gaussian matched filter -> Gardner symbol
timing recovery -> hard bits. It operates only on baseband IQ, so it is fully
testable with synthetic signals; no SDR or GNU Radio required.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GfskParams:
    sample_rate_hz: float
    symbol_rate_hz: float = 12_480.0
    mod_index: float = 0.5  # peak deviation = mod_index * symbol_rate / 2
    bt: float = 0.5  # Gaussian bandwidth-time product
    pulse_span_symbols: int = 4

    @property
    def sps(self) -> float:
        return self.sample_rate_hz / self.symbol_rate_hz

    @property
    def deviation_hz(self) -> float:
        return self.mod_index * self.symbol_rate_hz / 2.0


def gaussian_taps(params: GfskParams) -> np.ndarray:
    """Unit-area Gaussian pulse sampled at the working sample rate."""
    sps = params.sps
    # sigma (in samples) from BT: sigma_t = sqrt(ln2)/(2*pi*B), B = bt*Rs.
    b = params.bt * params.symbol_rate_hz
    sigma_samples = math.sqrt(math.log(2.0)) / (2.0 * math.pi * b) * params.sample_rate_hz
    half = int(round(params.pulse_span_symbols * sps / 2.0))
    n = np.arange(-half, half + 1, dtype=float)
    taps = np.exp(-(n**2) / (2.0 * sigma_samples**2))
    taps /= np.sum(taps)
    return taps


def modulate(bits: np.ndarray, params: GfskParams) -> np.ndarray:
    """Bits (0/1) -> complex64 baseband IQ at ``params.sample_rate_hz``."""
    bits = np.asarray(bits)
    if bits.size == 0:
        return np.empty(0, dtype=np.complex64)  # empty in → empty out (else np.convolve raises)
    sps = params.sps
    if abs(sps - round(sps)) > 1e-9:
        msg = f"sample_rate/symbol_rate must be integer for modulate (got {sps})"
        raise ValueError(msg)
    sps_i = int(round(sps))
    symbols = 2.0 * np.asarray(bits, dtype=float) - 1.0  # +/-1 NRZ
    nrz = np.repeat(symbols, sps_i)
    shaped = np.convolve(nrz, gaussian_taps(params), mode="same")
    inst_freq = params.deviation_hz * shaped  # Hz
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / params.sample_rate_hz
    return np.exp(1j * phase).astype(np.complex64)


def modulate_bytes(payload: bytes, params: GfskParams, *, bitorder: str = "big") -> np.ndarray:
    """Raw bytes -> 2-GFSK IQ, VERBATIM (no framing added). ``bitorder`` is numpy's ``"big"``
    (MSB-first, the default and the convention ``endurosat_link.frame_bits`` uses) or ``"little"``
    (LSB-first). For transmitting a blob that is already a complete on-air bitstream."""
    order = {"msb": "big", "lsb": "little"}.get(bitorder, bitorder)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder=order)
    return modulate(bits, params)


def modulate_bytes_zero_gaps(
    payload: bytes, params: GfskParams, *, bitorder: str = "big", min_gap_bytes: int = 0
) -> np.ndarray:
    """Like :func:`modulate_bytes`, but a run of ``>= min_gap_bytes`` zero bytes becomes
    zero-amplitude IQ (silence) of the SAME time duration instead of full-power FSK ``0`` symbols.

    For a pre-framed packet TRAIN that uses long ``0x00`` pads as inter-packet gaps: the padding
    becomes real silence (helps the receiver re-lock per packet) while its timing is preserved.
    ``min_gap_bytes <= 0`` -> plain verbatim raw (short zero runs stay legal FSK ``0`` bits)."""
    if min_gap_bytes <= 0:
        return modulate_bytes(payload, params, bitorder=bitorder)
    sps = params.sps
    if abs(sps - round(sps)) > 1e-9:
        msg = f"sample_rate/symbol_rate must be integer for raw gaps (got {sps})"
        raise ValueError(msg)
    sps_i = int(round(sps))
    parts: list[np.ndarray] = []
    pos = scan = 0
    n = len(payload)
    while scan < n:
        if payload[scan] != 0:
            scan += 1
            continue
        end = scan + 1
        while end < n and payload[end] == 0:
            end += 1
        run = end - scan
        if run >= min_gap_bytes:
            if pos < scan:
                parts.append(modulate_bytes(payload[pos:scan], params, bitorder=bitorder))
            parts.append(np.zeros(run * 8 * sps_i, dtype=np.complex64))
            pos = end
        scan = end
    if pos < n:
        parts.append(modulate_bytes(payload[pos:], params, bitorder=bitorder))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.complex64)


def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.zeros_like(x)
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def _gardner(soft_in: np.ndarray, sps: float, *, loop_gain: float = 0.05) -> np.ndarray:
    """Gardner symbol-timing recovery with linear interpolation.

    Returns one soft value per recovered symbol. Non-data-aided, so it works on
    the discriminator output directly; converges within a few tens of symbols.
    """
    x = np.asarray(soft_in, dtype=float)
    n = len(x)
    if n < 2 * sps:
        return np.empty(0, dtype=float)
    # Normalise to unit RMS so the loop gain is signal-amplitude independent.
    rms = math.sqrt(float(np.mean(x**2))) or 1.0
    x = x / rms

    def interp(pos: float) -> float:
        if pos <= 0.0:
            return float(x[0])
        if pos >= n - 1:
            return float(x[-1])
        k = int(pos)
        frac = pos - k
        return float(x[k] * (1.0 - frac) + x[k + 1] * frac)

    out: list[float] = []
    period = float(sps)
    half = period / 2.0
    t = period  # first full symbol instant
    while t < n - 1:
        curr = interp(t)
        mid = interp(t - half)
        prev = interp(t - period)
        # Gardner timing error (negative feedback): advances/retards t.
        err = mid * (curr - prev)
        out.append(curr)
        t += period - loop_gain * err
    return np.array(out, dtype=float)


def demodulate(iq: np.ndarray, params: GfskParams, *, recover_timing: bool = True) -> np.ndarray:
    """Baseband IQ -> hard bits (0/1).

    With ``recover_timing=False`` the symbols are taken at the maximum-eye phase
    (adequate for short bursts with negligible clock offset); otherwise Gardner
    timing recovery tracks the clock across the capture.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.empty(0, dtype=np.uint8)
    # Phase-difference discriminator -> instantaneous frequency (rad/sample).
    disc = np.angle(iq[1:] * np.conj(iq[:-1]))
    # Remove slow carrier/Doppler bias (window ~ many symbols).
    win = max(1, int(round(params.sps * 64)))
    if len(disc) < win:
        # Input shorter than the bias-removal kernel: numpy's 'same' convolution
        # returns kernel-length output for a shorter input, so the subtraction
        # below would broadcast-fail. No decodable frame fits in < 64 symbols
        # (the shortest on-air packet of either link layer here — preamble/flags
        # included — is longer), and such inputs always raised before this guard,
        # so returning no bits rejects nothing that ever decoded: these are burst
        # fragments (e.g. a segmenter carry), not frames.
        return np.empty(0, dtype=np.uint8)
    disc = disc - _moving_mean(disc, win)
    # Matched filter (same Gaussian as TX).
    mf = np.convolve(disc, gaussian_taps(params), mode="same")

    soft = _gardner(mf, params.sps) if recover_timing else _max_eye_symbols(mf, params.sps)
    return (soft > 0.0).astype(np.uint8)


def _max_eye_symbols(mf: np.ndarray, sps: float) -> np.ndarray:
    sps_i = int(round(sps))
    nsym = len(mf) // sps_i
    if nsym == 0:
        return np.empty(0, dtype=float)
    grid = mf[: nsym * sps_i].reshape(nsym, sps_i)
    phase = int(np.argmax(np.mean(np.abs(grid), axis=0)))
    return grid[:, phase]


def estimate_cfo_rad(iq: np.ndarray) -> float:
    """Coarse carrier-frequency offset (rad/sample): the mean instantaneous
    frequency of a balanced 2-FSK burst is ~the carrier offset.

    AMPLITUDE-WEIGHTED — ``angle(mean(product))``, not ``mean(angle(product))``. A balanced 2-FSK
    burst still averages to the carrier either way, but the mean-of-angles form weights EVERY sample
    equally, so low-amplitude regions (inter-burst guards, the quiet gaps between packets, the small
    residual of a channel-filtered-out interferer) pull the estimate onto whatever tiny tone lives
    in the gaps — measured as a bogus +18 kHz CFO on a channel-filtered burst whose true offset was
    ~0. Taking the angle of the amplitude-weighted mean product lets the (loud) burst dominate."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return 0.0
    return float(np.angle(np.mean(iq[1:] * np.conj(iq[:-1]))))


def derotate(iq: np.ndarray, cfo_rad: float) -> np.ndarray:
    n = np.arange(len(iq))
    return (np.asarray(iq, dtype=np.complex64) * np.exp(-1j * cfo_rad * n)).astype(np.complex64)


def demodulate_capture(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    symbol_rate_hz: float,
    mod_index: float = 0.5,
    bt: float = 0.5,
    target_sps: int = 16,
    correct_cfo: bool = True,
    recover_timing: bool = False,
    carrier_hz: float = 0.0,
    channel_bw_hz: float = 0.0,
) -> np.ndarray:
    """Robust burst demod for real SDR/VSA captures -> hard bits.

    Tuned on EnduroSat lab captures (measured 20/21 vs 12/21 for the plain
    ``demodulate``, and 21/21 with the small ensemble in
    :func:`endurosat_link.receive`): (1) derotate the coarse carrier offset,
    (2) optionally CHANNEL-SELECT filter to reject an off-channel interferer,
    (3) polyphase resample to an integer ``target_sps`` (real captures land on
    non-integer samples/symbol, e.g. 128 kHz / 9600 = 13.33), (4) sample at the
    maximum-eye phase (``recover_timing=False``) — best for short packets, which
    have no intra-burst clock drift. Pass one burst at a time (with a guard).

    ``channel_bw_hz`` (0 = off, the default → identical to the historical behaviour): after the
    ``carrier_hz`` de-rotation puts the wanted signal at DC, low-pass to ``±channel_bw_hz/2`` so a
    STRONG OFF-CHANNEL CARRIER (e.g. a co-visible satellite tens of kHz away) is removed BEFORE the
    two blocks it would otherwise wreck — the ``correct_cfo`` mean-angle estimate (a loud tone
    dominates the mean instantaneous frequency and drags the derotation onto the interferer) and the
    wideband FM discriminator in :func:`demodulate` (which has no channel filter of its own, so any
    in-band interferer beats against the wanted signal in every symbol). This is what lets a bursty
    GFSK downlink decode next to a continuous carrier; without it the interferer captures the demod.
    """
    from math import gcd

    from scipy.signal import firwin, resample_poly

    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.empty(0, dtype=np.uint8)
    if carrier_hz:
        # Shift a KNOWN coarse carrier offset to DC BEFORE the narrow demod filter. Doppler
        # compensation removes the pass Doppler but NOT the bird's fixed oscillator error (tens of
        # kHz at 400 MHz); left at +carrier_hz the signal sits outside the 0.625*baud channel and
        # the demod sees only noise. ``estimate_cfo_rad`` (mean-angle) can't recover this on a
        # bursty capture — the caller estimates it from the spectrum and passes it here.
        n = np.arange(len(iq), dtype=np.float64)
        iq = (iq * np.exp((-2j * np.pi * float(carrier_hz) / float(sample_rate_hz)) * n)).astype(
            np.complex64
        )
    if channel_bw_hz and channel_bw_hz > 0.0 and len(iq) > 8:
        # One-sided channel select at baseband (the wanted signal is now at DC): a linear-phase FIR
        # low-pass at half the channel bandwidth. Applied here — after de-rotation, before CFO and
        # the discriminator — so an off-channel carrier is gone before it can capture either. Odd
        # tap count scaled to the sample rate for a sharp-enough transition; clamped to the input.
        cutoff = min(float(channel_bw_hz) / 2.0, float(sample_rate_hz) / 2.0 * 0.98)
        ntaps = min(len(iq) - 1 | 1, max(31, int(4.0 * sample_rate_hz / channel_bw_hz) | 1))
        taps = firwin(ntaps, cutoff, fs=sample_rate_hz).astype(np.float64)
        iq = np.convolve(iq, taps, mode="same").astype(np.complex64)
    if correct_cfo:
        iq = derotate(iq, estimate_cfo_rad(iq))
    target_fs = symbol_rate_hz * target_sps
    up, down = int(round(target_fs)), int(round(sample_rate_hz))
    g = gcd(up, down) or 1
    iq = resample_poly(iq, up // g, down // g).astype(np.complex64)
    params = GfskParams(
        sample_rate_hz=target_fs, symbol_rate_hz=symbol_rate_hz, mod_index=mod_index, bt=bt
    )
    return demodulate(iq, params, recover_timing=recover_timing)


__all__ = [
    "GfskParams",
    "demodulate",
    "demodulate_capture",
    "derotate",
    "estimate_cfo_rad",
    "gaussian_taps",
    "modulate",
    "modulate_bytes",
    "modulate_bytes_zero_gaps",
]

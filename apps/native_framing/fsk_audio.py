"""Deterministic replay of real, already-FM-demodulated binary FSK audio.

Copyright 2019 Daniel Estevez <daniel@destevez.net>
Adapted for bounded gs-flowgraphs capture replay in 2026.
SPDX-License-Identifier: GPL-3.0-or-later

The square-pulse and 32-symbol DC-removal policy follows the pinned
gr-satellites real-input FSK demodulator.  This helper deliberately omits a
live clock-recovery loop: callers select a source-sample phase explicitly, so
capture tests remain deterministic and frame offsets map back to the WAV.

The separate Mueller-Muller entry point reproduces the bounded legacy chain
used by the published classic-Mobitex recordings: sharp 3 kHz low-pass,
amplitude-calibrated decision-directed timing, and a +/-0.5 percent clock
period limit.  It is explicit because those old discriminator recordings do
not use the later square-pulse/Gardner frontend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import signal

from native_framing.registry import build_decoder, resolve_profile
from native_framing.sample_clock import SampleClock
from native_framing.types import FrameResult, SymbolInput

MAX_MM_FILTER_TAPS = 65_537


@dataclass(frozen=True)
class FskAudioConfig:
    """Clock and bounded DC-removal configuration for discriminator audio."""

    sample_rate_hz: float
    symbol_rate_hz: float
    dc_block_symbols: int = 32

    def __post_init__(self) -> None:
        for name, value in {
            "sample_rate_hz": self.sample_rate_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
        }.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite positive number") from exc
            if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if (
            isinstance(self.dc_block_symbols, bool)
            or not isinstance(self.dc_block_symbols, int)
            or not 1 <= self.dc_block_symbols <= 4096
        ):
            raise ValueError("dc_block_symbols must be an integer in 1..4096")
        SampleClock(self.sample_rate_hz, self.symbol_rate_hz)


@dataclass(frozen=True)
class FskAudioSymbols:
    """FSK decisions and their exact source-audio symbol boundaries."""

    hard_bits: np.ndarray
    soft_symbols: np.ndarray
    sample_boundaries: np.ndarray
    phase_samples: int

    def __post_init__(self) -> None:
        hard = np.array(self.hard_bits, dtype=np.uint8, copy=True)
        soft = np.array(self.soft_symbols, dtype=np.float64, copy=True)
        boundaries = np.array(self.sample_boundaries, dtype=np.int64, copy=True)
        if hard.ndim != 1 or soft.ndim != 1 or hard.size != soft.size:
            raise ValueError("hard and soft symbols must be equal-length vectors")
        if boundaries.ndim != 1 or boundaries.size != hard.size + 1:
            raise ValueError("sample_boundaries must contain one boundary per symbol edge")
        if boundaries.size and boundaries[0] != self.phase_samples:
            raise ValueError("the first sample boundary must equal phase_samples")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("sample boundaries must be strictly increasing")
        hard.setflags(write=False)
        soft.setflags(write=False)
        boundaries.setflags(write=False)
        object.__setattr__(self, "hard_bits", hard)
        object.__setattr__(self, "soft_symbols", soft)
        object.__setattr__(self, "sample_boundaries", boundaries)

    def sample_offset(self, symbol_offset: int) -> int:
        if isinstance(symbol_offset, bool) or not isinstance(symbol_offset, int):
            raise ValueError("symbol_offset must be an integer")
        if not 0 <= symbol_offset <= self.hard_bits.size:
            raise ValueError("symbol_offset is outside the demodulated stream")
        return int(self.sample_boundaries[symbol_offset])


@dataclass(frozen=True)
class FskAudioDecodedFrame:
    """A native frame located in symbol and source-audio clocks."""

    frame: FrameResult
    source_sample_start: int
    source_sample_end: int


def _full_symbol_boundaries(
    sample_count: int, clock: SampleClock, phase_samples: int
) -> np.ndarray:
    available = sample_count - phase_samples
    if available < 0:
        raise ValueError("phase_samples exceeds the audio length")
    estimate = int(available / clock.samples_per_symbol) + 2
    while estimate and clock.sample_offset_for_symbol(
        estimate, origin_sample=phase_samples
    ) > sample_count:
        estimate -= 1
    return np.fromiter(
        (
            clock.sample_offset_for_symbol(index, origin_sample=phase_samples)
            for index in range(estimate + 1)
        ),
        dtype=np.int64,
        count=estimate + 1,
    )


def demodulate_fsk_audio(
    audio: np.ndarray,
    config: FskAudioConfig,
    *,
    phase_samples: int = 0,
) -> FskAudioSymbols:
    """Convert real discriminator audio to normalized binary soft symbols."""

    source = np.asarray(audio)
    if source.ndim != 1 or np.iscomplexobj(source):
        raise ValueError("audio must be a one-dimensional real vector")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("audio must contain numeric samples")
    samples = source.astype(np.float64, copy=False)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite")
    if isinstance(phase_samples, bool) or not isinstance(phase_samples, int):
        raise ValueError("phase_samples must be an integer")
    if phase_samples < 0:
        raise ValueError("phase_samples must be non-negative")

    clock = SampleClock(config.sample_rate_hz, config.symbol_rate_hz)
    boundaries = _full_symbol_boundaries(samples.size, clock, phase_samples)
    dc_width = max(
        1, math.ceil(clock.samples_per_symbol * config.dc_block_symbols)
    )
    if samples.size:
        baseline = signal.convolve(
            samples, np.full(dc_width, 1.0 / dc_width), mode="same"
        )
        centered = samples - baseline
    else:
        centered = samples
    soft = np.fromiter(
        (
            float(np.mean(centered[start:end]))
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ),
        dtype=np.float64,
        count=boundaries.size - 1,
    )
    rms = float(np.sqrt(np.mean(soft * soft))) if soft.size else 0.0
    if rms > 0:
        soft /= rms
    return FskAudioSymbols(soft > 0, soft, boundaries, phase_samples)


def demodulate_fsk_audio_mm(
    audio: np.ndarray,
    config: FskAudioConfig,
    *,
    cutoff_hz: float = 3_000.0,
    transition_hz: float = 100.0,
    gain_mu: float = 0.175,
    omega_relative_limit: float = 0.005,
) -> FskAudioSymbols:
    """Recover FSK symbols with the legacy GNU Radio Mueller-Muller loop."""

    source = np.asarray(audio)
    if source.ndim != 1 or np.iscomplexobj(source):
        raise ValueError("audio must be a one-dimensional real vector")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("audio must contain numeric samples")
    samples = source.astype(np.float64, copy=False)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite")
    for name, value in {
        "cutoff_hz": cutoff_hz,
        "transition_hz": transition_hz,
        "gain_mu": gain_mu,
        "omega_relative_limit": omega_relative_limit,
    }.items():
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
    nyquist = float(config.sample_rate_hz) / 2
    if float(cutoff_hz) >= nyquist:
        raise ValueError("cutoff_hz must be below Nyquist")
    sps = float(config.sample_rate_hz) / float(config.symbol_rate_hz)
    if sps <= 1:
        raise ValueError("Mueller-Muller timing requires more than one sample per symbol")
    if float(omega_relative_limit) >= 0.5:
        raise ValueError("omega_relative_limit must be below 0.5")

    tap_count = math.ceil(4 * float(config.sample_rate_hz) / float(transition_hz))
    tap_count += 1 - tap_count % 2
    if tap_count > MAX_MM_FILTER_TAPS:
        raise ValueError(f"low-pass design exceeds {MAX_MM_FILTER_TAPS} taps")
    taps = signal.firwin(
        tap_count,
        float(cutoff_hz),
        fs=float(config.sample_rate_hz),
        window="hamming",
    )
    if np.issubdtype(source.dtype, np.integer):
        scale = float(max(abs(np.iinfo(source.dtype).min), np.iinfo(source.dtype).max))
        samples = samples / scale
    # The historical flowgraph converted int16 to full-scale float and then
    # multiplied by ten.  The M&M error and loop gains depend on this scale.
    filtered = signal.lfilter(taps, [1.0], samples * 10.0)

    omega = sps
    omega_mid = sps
    omega_limit = sps * float(omega_relative_limit)
    gain_omega = 0.25 * float(gain_mu) * float(gain_mu)
    mu = 0.5
    index = 0
    last = 0.0
    output: list[float] = []
    positions: list[int] = []
    while index < filtered.size - 2:
        position = index + mu
        left = int(position)
        fraction = position - left
        value = float(
            filtered[left] * (1.0 - fraction) + filtered[left + 1] * fraction
        )
        output.append(value)
        rounded = int(round(position))
        positions.append(
            rounded if not positions or rounded > positions[-1] else positions[-1] + 1
        )
        decision_last = 1.0 if last > 0 else -1.0
        decision_current = 1.0 if value > 0 else -1.0
        error = decision_last * value - decision_current * last
        last = value
        omega = min(
            omega_mid + omega_limit,
            max(omega_mid - omega_limit, omega + gain_omega * error),
        )
        mu += omega + float(gain_mu) * error
        advance = math.floor(mu)
        index += advance
        mu -= advance

    soft = np.asarray(output, dtype=np.float64)
    rms = float(np.sqrt(np.mean(soft * soft))) if soft.size else 0.0
    if rms > 0:
        soft /= rms
    if positions:
        final = min(samples.size, max(positions[-1] + 1, int(round(index + mu))))
        boundaries = np.asarray(positions + [final], dtype=np.int64)
        phase_samples = positions[0]
    else:
        boundaries = np.asarray([0], dtype=np.int64)
        phase_samples = 0
    return FskAudioSymbols(soft > 0, soft, boundaries, phase_samples)


def decode_fsk_audio_profile(
    audio: np.ndarray,
    config: FskAudioConfig,
    framing: str,
    parameters: Mapping[str, object] | None = None,
    *,
    phase_samples: int = 0,
) -> tuple[FskAudioSymbols, list[FskAudioDecodedFrame]]:
    """Demodulate one real-audio capture and feed a native framing profile."""

    profile = resolve_profile(framing)
    if profile is None:
        raise KeyError(f"unknown framing profile: {framing!r}")
    symbols = demodulate_fsk_audio(audio, config, phase_samples=phase_samples)
    decoder = build_decoder(framing, parameters)
    decoder_input = (
        symbols.hard_bits
        if profile.symbol_input is SymbolInput.HARD_BITS
        else symbols.soft_symbols
    )
    frames = decoder.push(decoder_input) + decoder.flush()
    located = [
        FskAudioDecodedFrame(
            frame,
            symbols.sample_offset(frame.source_start),
            symbols.sample_offset(frame.source_end),
        )
        for frame in frames
    ]
    return symbols, located


def decode_fsk_audio_mm_profile(
    audio: np.ndarray,
    config: FskAudioConfig,
    framing: str,
    parameters: Mapping[str, object] | None = None,
    *,
    cutoff_hz: float = 3_000.0,
    transition_hz: float = 100.0,
    gain_mu: float = 0.175,
    omega_relative_limit: float = 0.005,
) -> tuple[FskAudioSymbols, list[FskAudioDecodedFrame]]:
    """Apply configurable legacy M&M timing and feed one native profile."""

    profile = resolve_profile(framing)
    if profile is None:
        raise KeyError(f"unknown framing profile: {framing!r}")
    symbols = demodulate_fsk_audio_mm(
        audio,
        config,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
        gain_mu=gain_mu,
        omega_relative_limit=omega_relative_limit,
    )
    decoder = build_decoder(framing, parameters)
    decoder_input = (
        symbols.hard_bits
        if profile.symbol_input is SymbolInput.HARD_BITS
        else symbols.soft_symbols
    )
    frames = decoder.push(decoder_input) + decoder.flush()
    located = [
        FskAudioDecodedFrame(
            frame,
            symbols.sample_offset(frame.source_start),
            symbols.sample_offset(frame.source_end),
        )
        for frame in frames
    ]
    return symbols, located


__all__ = [
    "FskAudioConfig",
    "FskAudioDecodedFrame",
    "FskAudioSymbols",
    "decode_fsk_audio_profile",
    "decode_fsk_audio_mm_profile",
    "demodulate_fsk_audio",
    "demodulate_fsk_audio_mm",
]

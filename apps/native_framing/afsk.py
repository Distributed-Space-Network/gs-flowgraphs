"""Deterministic, engine-independent binary AFSK replay.

The demodulator deliberately uses bounded non-coherent tone-energy windows.  It
is intended for reproducible capture replay and decoder verification, not as a
replacement for a live timing-recovery loop.  Every symbol boundary is retained
in the original audio-sample clock so decoded frames can be timestamped without
wall-clock inference.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from native_framing.registry import build_decoder, resolve_profile
from native_framing.sample_clock import SampleClock
from native_framing.types import FrameResult, SymbolInput


@dataclass(frozen=True)
class AfskConfig:
    """Binary AFSK tone and clock configuration."""

    sample_rate_hz: float
    symbol_rate_hz: float
    one_hz: float
    zero_hz: float

    def __post_init__(self) -> None:
        values = {
            "sample_rate_hz": self.sample_rate_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
            "one_hz": self.one_hz,
            "zero_hz": self.zero_hz,
        }
        for name, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite positive number") from exc
            if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.one_hz == self.zero_hz:
            raise ValueError("one_hz and zero_hz must be distinct")
        nyquist = self.sample_rate_hz / 2
        if max(self.one_hz, self.zero_hz) >= nyquist:
            raise ValueError("AFSK tones must be below the Nyquist frequency")
        SampleClock(self.sample_rate_hz, self.symbol_rate_hz)


@dataclass(frozen=True)
class AfskSymbols:
    """Demodulated symbols plus exact boundaries in the source sample clock."""

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
class AfskDecodedFrame:
    """A native frame located in both symbol and source-audio clocks."""

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


def _tone_energy(samples: np.ndarray, sample_rate_hz: float, frequency_hz: float) -> float:
    time = np.arange(samples.size, dtype=np.float64) / sample_rate_hz
    cosine = np.cos(2 * np.pi * frequency_hz * time)
    sine = np.sin(2 * np.pi * frequency_hz * time)
    return float(np.dot(samples, cosine) ** 2 + np.dot(samples, sine) ** 2)


def demodulate_afsk(
    audio: np.ndarray,
    config: AfskConfig,
    *,
    phase_samples: int = 0,
) -> AfskSymbols:
    """Convert mono audio into hard bits and normalized soft tone decisions."""

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
    centered = samples - np.mean(samples) if samples.size else samples
    soft = np.empty(boundaries.size - 1, dtype=np.float64)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        window = centered[start:end]
        one_energy = _tone_energy(window, config.sample_rate_hz, config.one_hz)
        zero_energy = _tone_energy(window, config.sample_rate_hz, config.zero_hz)
        total = one_energy + zero_energy
        soft[index] = (one_energy - zero_energy) / total if total > 0 else 0.0
    return AfskSymbols(soft > 0, soft, boundaries, phase_samples)


def decode_afsk_profile(
    audio: np.ndarray,
    config: AfskConfig,
    framing: str,
    parameters: Mapping[str, object] | None = None,
    *,
    phase_samples: int = 0,
) -> tuple[AfskSymbols, list[AfskDecodedFrame]]:
    """Demodulate audio and feed the selected native framing profile once."""

    profile = resolve_profile(framing)
    if profile is None:
        raise KeyError(f"unknown framing profile: {framing!r}")
    symbols = demodulate_afsk(audio, config, phase_samples=phase_samples)
    decoder = build_decoder(framing, parameters)
    decoder_input = (
        symbols.hard_bits
        if profile.symbol_input is SymbolInput.HARD_BITS
        else symbols.soft_symbols
    )
    frames = decoder.push(decoder_input) + decoder.flush()
    located = [
        AfskDecodedFrame(
            frame,
            symbols.sample_offset(frame.source_start),
            symbols.sample_offset(frame.source_end),
        )
        for frame in frames
    ]
    return symbols, located


__all__ = [
    "AfskConfig",
    "AfskDecodedFrame",
    "AfskSymbols",
    "decode_afsk_profile",
    "demodulate_afsk",
]

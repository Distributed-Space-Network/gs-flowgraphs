"""Bounded native replay of real-audio BPSK and DBPSK captures.

The carrier tracker uses the BPSK squaring method: squaring removes the data
sign, so a blockwise phase-increment estimate can follow slow audio-carrier
drift without a payload oracle.  Manchester captures evaluate both fixed
``[1, -1]`` decimation phases, following Daniel Estevez's published AO-40
uncoded receiver, and protocol integrity selects the replay candidate.

Copyright 2017-2026 Daniel Estevez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import signal

from native_framing.psk import BpskConfig, BpskSymbols, demodulate_bpsk
from native_framing.registry import build_decoder, resolve_profile
from native_framing.types import FrameResult, IntegrityStatus, SymbolInput


@dataclass(frozen=True)
class BpskAudioConfig:
    """Carrier and symbol-clock search bounds for one real-audio replay."""

    sample_rate_hz: float
    symbol_rate_hz: float
    differential: bool = False
    manchester: bool = False
    carrier_block_s: float = 0.25
    clock_search_ppm: float = 500.0
    clock_search_steps: int = 9
    manchester_block_size: int = 32
    rrc_rolloff: float | None = 0.35
    rrc_span_symbols: int = 11

    def __post_init__(self) -> None:
        for name in (
            "sample_rate_hz",
            "symbol_rate_hz",
            "carrier_block_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(self.differential, bool) or not isinstance(
            self.manchester, bool
        ):
            raise ValueError("differential and manchester must be booleans")
        if self.sample_rate_hz * self.carrier_block_s < 8:
            raise ValueError("carrier_block_s must contain at least eight samples")
        if (
            isinstance(self.clock_search_ppm, bool)
            or not math.isfinite(float(self.clock_search_ppm))
            or not 0 <= self.clock_search_ppm <= 10_000
        ):
            raise ValueError("clock_search_ppm must be finite and in 0..10000")
        if (
            isinstance(self.clock_search_steps, bool)
            or not isinstance(self.clock_search_steps, int)
            or not 1 <= self.clock_search_steps <= 33
            or self.clock_search_steps % 2 == 0
        ):
            raise ValueError("clock_search_steps must be an odd integer in 1..33")
        # Reuse the base demodulator's complete clock/filter validation.
        BpskConfig(
            self.sample_rate_hz,
            self.symbol_rate_hz,
            differential=self.differential,
            manchester=self.manchester,
            manchester_block_size=self.manchester_block_size,
            rrc_rolloff=self.rrc_rolloff,
            rrc_span_symbols=self.rrc_span_symbols,
        )


@dataclass(frozen=True)
class BpskAudioDecodedFrame:
    """A native frame located in the source-audio sample clock."""

    frame: FrameResult
    source_sample_start: int
    source_sample_end: int


@dataclass(frozen=True)
class BpskAudioReplay:
    """The integrity-selected candidate and its carrier-search diagnostics."""

    symbols: BpskSymbols
    frames: tuple[BpskAudioDecodedFrame, ...]
    selected_symbol_rate_hz: float
    selected_clock_error_ppm: float
    selected_manchester_phase: int | None
    carrier_frequency_min_hz: float
    carrier_frequency_median_hz: float
    carrier_frequency_max_hz: float


def _real_audio(audio: np.ndarray) -> np.ndarray:
    source = np.asarray(audio)
    if source.ndim != 1 or np.iscomplexobj(source):
        raise ValueError("audio must be a one-dimensional real vector")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("audio must contain numeric samples")
    samples = source.astype(np.float64, copy=False)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite")
    return samples


def _track_carrier(
    audio: np.ndarray, config: BpskAudioConfig
) -> tuple[np.ndarray, np.ndarray]:
    if not audio.size:
        return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.float64)
    analytic = signal.hilbert(audio)
    block_samples = max(8, int(round(config.sample_rate_hz * config.carrier_block_s)))
    centers: list[float] = []
    frequencies: list[float] = []
    for start in range(0, analytic.size, block_samples):
        block = analytic[start : start + block_samples]
        squared = np.square(block)
        cross = np.vdot(squared[:-1], squared[1:]) if squared.size > 1 else 0j
        energy = float(np.vdot(squared, squared).real)
        if energy <= 0 or abs(cross) <= np.finfo(np.float64).eps * energy:
            continue
        frequency = float(
            np.angle(cross) * float(config.sample_rate_hz) / (4.0 * np.pi)
        )
        if not math.isfinite(frequency):
            continue
        centers.append(start + 0.5 * (block.size - 1))
        frequencies.append(frequency)
    if not frequencies:
        raise ValueError("audio has no measurable BPSK carrier")
    tracked = np.interp(
        np.arange(analytic.size, dtype=np.float64),
        np.asarray(centers),
        np.asarray(frequencies),
    )
    phase = 2.0 * np.pi * np.cumsum(tracked) / float(config.sample_rate_hz)
    return analytic * np.exp(-1j * phase), np.asarray(frequencies, dtype=np.float64)


def _candidate_rates(config: BpskAudioConfig) -> np.ndarray:
    if config.clock_search_steps == 1 or config.clock_search_ppm == 0:
        return np.asarray([config.symbol_rate_hz], dtype=np.float64)
    ppm = np.linspace(
        -float(config.clock_search_ppm),
        float(config.clock_search_ppm),
        config.clock_search_steps,
    )
    return float(config.symbol_rate_hz) * (1.0 + ppm * 1e-6)


def decode_bpsk_audio_profile(
    audio: np.ndarray,
    config: BpskAudioConfig,
    framing: str,
    parameters: Mapping[str, object] | None = None,
) -> BpskAudioReplay:
    """Track, demodulate, and integrity-select one native framing replay."""

    profile = resolve_profile(framing)
    if profile is None:
        raise KeyError(f"unknown framing profile: {framing!r}")
    samples = _real_audio(audio)
    baseband, carrier_frequencies = _track_carrier(samples, config)
    phases: tuple[int | None, ...] = (0, 1) if config.manchester else (None,)
    best: tuple[
        tuple[int, float, float, int],
        BpskSymbols,
        list[FrameResult],
        float,
        int | None,
    ] | None = None
    for rate in _candidate_rates(config):
        clock_error_ppm = (float(rate) / float(config.symbol_rate_hz) - 1.0) * 1e6
        for phase in phases:
            symbols = demodulate_bpsk(
                baseband,
                BpskConfig(
                    config.sample_rate_hz,
                    float(rate),
                    differential=config.differential,
                    manchester=config.manchester,
                    manchester_block_size=config.manchester_block_size,
                    rrc_rolloff=config.rrc_rolloff,
                    rrc_span_symbols=config.rrc_span_symbols,
                    frequency_offset_hz=0.0,
                    manchester_phase=phase,
                ),
            )
            decoder = build_decoder(framing, parameters)
            decoder_input = (
                symbols.hard_bits
                if profile.symbol_input is SymbolInput.HARD_BITS
                else symbols.soft_symbols
            )
            decoded = decoder.push(decoder_input) + decoder.flush()
            # Carrier/clock hypotheses may only be ranked by a protocol-level
            # integrity pass.  Sync-only or integrity-free profiles would turn
            # random candidate multiplicity into an oracle and are therefore
            # deliberately suppressed here.
            frames = [
                frame
                for frame in decoded
                if frame.integrity is IntegrityStatus.PASSED
            ]
            sync_distance = sum(float(frame.sync_distance) for frame in frames)
            score = (
                len(frames),
                -sync_distance,
                -abs(clock_error_ppm),
                -(phase if phase is not None else 0),
            )
            if best is None or score > best[0]:
                best = (score, symbols, frames, float(rate), phase)
    if best is None:  # the validated candidate grid is always non-empty
        raise RuntimeError("BPSK replay produced no demodulation candidate")
    _, symbols, frames, rate, phase = best
    located = tuple(
        BpskAudioDecodedFrame(
            frame=frame,
            source_sample_start=symbols.sample_offset(frame.source_start),
            source_sample_end=symbols.sample_offset(frame.source_end),
        )
        for frame in frames
    )
    clock_error_ppm = (rate / float(config.symbol_rate_hz) - 1.0) * 1e6
    frequency_min = float(np.min(carrier_frequencies)) if carrier_frequencies.size else 0.0
    frequency_median = (
        float(np.median(carrier_frequencies)) if carrier_frequencies.size else 0.0
    )
    frequency_max = float(np.max(carrier_frequencies)) if carrier_frequencies.size else 0.0
    return BpskAudioReplay(
        symbols=symbols,
        frames=located,
        selected_symbol_rate_hz=rate,
        selected_clock_error_ppm=clock_error_ppm,
        selected_manchester_phase=phase,
        carrier_frequency_min_hz=frequency_min,
        carrier_frequency_median_hz=frequency_median,
        carrier_frequency_max_hz=frequency_max,
    )


__all__ = [
    "BpskAudioConfig",
    "BpskAudioDecodedFrame",
    "BpskAudioReplay",
    "decode_bpsk_audio_profile",
]

"""Deterministic, engine-independent BPSK/DBPSK capture replay.

The replay mirrors the observable stages of the pinned gr-satellites BPSK
demodulator: carrier removal, RRC matched filtering, symbol timing, and either
coherent or multiply-conjugate differential decisions.  It is deliberately a
bounded file-replay implementation; a live adaptive FLL/Costas/timing-loop
backend still requires GNU Radio parity and station evidence.

Manchester recovery is an attributed NumPy adaptation of the pinned
gr-satellites ``manchester_sync`` block: it demodulates at twice the declared
symbol rate, evaluates both half-symbol phases in bounded blocks, and emits the
half-difference having the larger sum-of-magnitudes metric.

Soft symbols follow the native package convention ``positive => hard bit 1``.
For DBPSK the raw multiply-conjugate decision is positive for no transition,
which is the complement of GNU Radio's modulus-2 differential input bit.  This
global complement is intentional and is resolved by framing profiles that
accept both polarities.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from native_framing.sample_clock import SampleClock


@dataclass(frozen=True)
class BpskConfig:
    sample_rate_hz: float
    symbol_rate_hz: float
    differential: bool = False
    manchester: bool = False
    manchester_block_size: int = 32
    rrc_rolloff: float | None = 0.35
    rrc_span_symbols: int = 11
    frequency_offset_hz: float | None = None
    manchester_phase: int | None = None

    def __post_init__(self) -> None:
        for name in ("sample_rate_hz", "symbol_rate_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(self.differential, bool) or not isinstance(self.manchester, bool):
            raise ValueError("differential and manchester must be booleans")
        effective_symbol_rate = self.symbol_rate_hz * (2 if self.manchester else 1)
        SampleClock(
            self.sample_rate_hz,
            effective_symbol_rate,
            minimum_samples_per_symbol=4.0,
        )
        if (
            isinstance(self.manchester_block_size, bool)
            or not isinstance(self.manchester_block_size, int)
            or self.manchester_block_size < 1
        ):
            raise ValueError("manchester_block_size must be a positive integer")
        if self.rrc_rolloff is not None and not 0.0 <= self.rrc_rolloff <= 1.0:
            raise ValueError("rrc_rolloff must be between zero and one or None")
        if (
            isinstance(self.rrc_span_symbols, bool)
            or not isinstance(self.rrc_span_symbols, int)
            or self.rrc_span_symbols < 3
        ):
            raise ValueError("rrc_span_symbols must be an integer of at least three")
        if self.frequency_offset_hz is not None:
            offset = float(self.frequency_offset_hz)
            if not math.isfinite(offset) or abs(offset) >= self.sample_rate_hz / 4:
                raise ValueError(
                    "frequency_offset_hz must be finite and below one quarter sample rate"
                )
        if self.manchester_phase is not None:
            if self.manchester_phase not in (0, 1) or isinstance(
                self.manchester_phase, bool
            ):
                raise ValueError("manchester_phase must be zero, one, or None")
            if not self.manchester:
                raise ValueError("manchester_phase requires Manchester recovery")


@dataclass(frozen=True)
class BpskSymbols:
    hard_bits: np.ndarray
    soft_symbols: np.ndarray
    sample_boundaries: np.ndarray
    phase_samples: int
    estimated_frequency_hz: float

    def __post_init__(self) -> None:
        hard = np.asarray(self.hard_bits, dtype=np.uint8)
        soft = np.asarray(self.soft_symbols, dtype=np.float64)
        boundaries = np.asarray(self.sample_boundaries, dtype=np.int64)
        if hard.ndim != 1 or soft.ndim != 1 or hard.size != soft.size:
            raise ValueError("hard and soft symbols must be equal-length vectors")
        if boundaries.ndim != 1 or boundaries.size != hard.size + 1:
            raise ValueError("sample_boundaries must contain one boundary per symbol edge")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("sample boundaries must be strictly increasing")
        if not math.isfinite(self.estimated_frequency_hz):
            raise ValueError("estimated_frequency_hz must be finite")
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


def root_raised_cosine_taps(
    samples_per_symbol: float, rolloff: float, span_symbols: int
) -> np.ndarray:
    """Return an odd, unit-energy RRC impulse response in symbol-time units."""

    if not math.isfinite(samples_per_symbol) or samples_per_symbol < 4:
        raise ValueError("samples_per_symbol must be finite and at least four")
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError("rolloff must be between zero and one")
    if isinstance(span_symbols, bool) or not isinstance(span_symbols, int) or span_symbols < 3:
        raise ValueError("span_symbols must be an integer of at least three")
    half = int(math.ceil(span_symbols * samples_per_symbol / 2.0))
    time = np.arange(-half, half + 1, dtype=np.float64) / samples_per_symbol
    if rolloff == 0:
        taps = np.sinc(time)
    else:
        taps = np.empty_like(time)
        singular = 1.0 / (4.0 * rolloff)
        for index, value in enumerate(time):
            if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
                taps[index] = 1.0 + rolloff * (4.0 / np.pi - 1.0)
            elif math.isclose(abs(value), singular, rel_tol=0.0, abs_tol=1e-12):
                taps[index] = rolloff / math.sqrt(2.0) * (
                    (1.0 + 2.0 / np.pi) * math.sin(np.pi / (4.0 * rolloff))
                    + (1.0 - 2.0 / np.pi) * math.cos(np.pi / (4.0 * rolloff))
                )
            else:
                numerator = math.sin(np.pi * value * (1.0 - rolloff))
                numerator += 4.0 * rolloff * value * math.cos(
                    np.pi * value * (1.0 + rolloff)
                )
                denominator = np.pi * value * (1.0 - (4.0 * rolloff * value) ** 2)
                taps[index] = numerator / denominator
    energy = float(np.linalg.norm(taps))
    if not energy:
        raise RuntimeError("RRC tap construction produced zero energy")
    return taps / energy


def _boundaries(sample_count: int, clock: SampleClock, phase: int) -> np.ndarray:
    available = sample_count - phase
    estimate = max(0, int(available / clock.samples_per_symbol) + 2)
    while estimate and clock.sample_offset_for_symbol(
        estimate, origin_sample=phase
    ) > sample_count:
        estimate -= 1
    return np.fromiter(
        (
            clock.sample_offset_for_symbol(index, origin_sample=phase)
            for index in range(estimate + 1)
        ),
        dtype=np.int64,
        count=estimate + 1,
    )


def _timed_symbols(
    samples: np.ndarray, clock: SampleClock
) -> tuple[np.ndarray, np.ndarray, int]:
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    phase_count = int(math.ceil(clock.samples_per_symbol))
    base_boundaries = _boundaries(samples.size, clock, 0)
    for phase in range(min(phase_count, samples.size + 1)):
        boundary_count = int(
            np.searchsorted(base_boundaries, samples.size - phase, side="right")
        )
        boundaries = base_boundaries[:boundary_count] + phase
        if boundaries.size < 2:
            continue
        centers = np.rint((boundaries[:-1] + boundaries[1:] - 1) / 2.0).astype(np.int64)
        symbols = samples[centers]
        score = float(np.mean(np.abs(symbols)))
        candidate = (score, -phase, symbols, boundaries)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return (
            np.empty(0, dtype=np.complex128),
            np.asarray([0], dtype=np.int64),
            0,
        )
    return best[2], best[3], -best[1]


def _frequency_offset(samples: np.ndarray, sample_rate_hz: float) -> float:
    squared = np.square(samples)
    cross = np.vdot(squared[:-1], squared[1:]) if squared.size > 1 else 0j
    if abs(cross) <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.angle(cross) * sample_rate_hz / (4.0 * np.pi))


def _normalize_soft(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    nonzero = magnitude[magnitude > np.finfo(np.float64).eps]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    return np.asarray(values / scale, dtype=np.float64)


def _manchester_sync(
    symbols: np.ndarray, boundaries: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recover Manchester symbols with pinned gr-satellites block semantics."""

    recovered: list[np.ndarray] = []
    start_half_symbols: list[np.ndarray] = []
    output_start = 0
    while True:
        base = 2 * output_start
        available = (symbols.size - base - 1) // 2
        if available <= 0:
            break
        count = min(block_size, available)
        phase0 = 0.5 * (
            symbols[base : base + 2 * count : 2]
            - symbols[base + 1 : base + 2 * count : 2]
        )
        phase1 = 0.5 * (
            symbols[base + 1 : base + 2 * count + 1 : 2]
            - symbols[base + 2 : base + 2 * count + 2 : 2]
        )
        use_phase0 = float(np.sum(np.abs(phase0))) > float(np.sum(np.abs(phase1)))
        phase = 0 if use_phase0 else 1
        recovered.append(phase0 if use_phase0 else phase1)
        start_half_symbols.append(base + phase + 2 * np.arange(count, dtype=np.int64))
        output_start += count

    if not recovered:
        return np.empty(0, dtype=np.complex128), boundaries[:1].copy()
    values = np.concatenate(recovered)
    starts = np.concatenate(start_half_symbols)
    source_boundaries = np.concatenate(
        (boundaries[starts], boundaries[starts[-1] + 2 : starts[-1] + 3])
    )
    return values, source_boundaries


def _manchester_sync_fixed_phase(
    symbols: np.ndarray, boundaries: np.ndarray, phase: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recover one fixed Manchester half-symbol phase.

    The original AO-40 uncoded receiver evaluated both decimating ``[1, -1]``
    phases and let the frame CRC select the valid branch.  A fixed branch is
    also useful when the block-energy selector is ambiguous on long runs of
    unscrambled telemetry bytes.
    """

    count = (symbols.size - phase - 1) // 2
    if count <= 0:
        return np.empty(0, dtype=np.complex128), boundaries[:1].copy()
    starts = phase + 2 * np.arange(count, dtype=np.int64)
    values = 0.5 * (symbols[starts] - symbols[starts + 1])
    source_boundaries = np.concatenate(
        (boundaries[starts], boundaries[starts[-1] + 2 : starts[-1] + 3])
    )
    return values, source_boundaries


def manchester_sync_symbols(
    half_symbols: np.ndarray, block_size: int = 32
) -> np.ndarray:
    """Apply the pinned gr-satellites Manchester synchronizer to half-symbols.

    GNU Radio's ``history(2)`` supplies one zero-valued history item before the
    first finite-vector item.  Reproducing that detail is required for exact
    parity with the pinned normal and one-half-symbol-offset QA construction.
    """

    source = np.asarray(half_symbols)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("half_symbols must be a one-dimensional numeric vector")
    if not np.all(np.isfinite(source)):
        raise ValueError("half_symbols must be finite")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    padded = np.concatenate((np.zeros(1, dtype=source.dtype), source))
    values, _ = _manchester_sync(
        padded,
        np.arange(padded.size + 1, dtype=np.int64),
        block_size,
    )
    return values


def demodulate_bpsk(iq: np.ndarray, config: BpskConfig) -> BpskSymbols:
    """Demodulate one already-channelized complex BPSK/DBPSK capture."""

    source = np.asarray(iq)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("IQ must be a one-dimensional numeric vector")
    samples = source.astype(np.complex128, copy=False)
    if not np.all(np.isfinite(samples)):
        raise ValueError("IQ samples must be finite")
    if not samples.size:
        return BpskSymbols(
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            0,
            0.0,
        )

    frequency = (
        _frequency_offset(samples, config.sample_rate_hz)
        if config.frequency_offset_hz is None
        else float(config.frequency_offset_hz)
    )
    time = np.arange(samples.size, dtype=np.float64) / config.sample_rate_hz
    baseband = samples * np.exp(-2j * np.pi * frequency * time)
    effective_symbol_rate = config.symbol_rate_hz * (2 if config.manchester else 1)
    clock = SampleClock(
        config.sample_rate_hz,
        effective_symbol_rate,
        minimum_samples_per_symbol=4.0,
    )
    if config.rrc_rolloff is not None:
        taps = root_raised_cosine_taps(
            clock.samples_per_symbol,
            config.rrc_rolloff,
            config.rrc_span_symbols,
        )
        if baseband.size * taps.size > 1_000_000:
            filtered = signal.fftconvolve(baseband, taps, mode="full")
        else:
            filtered = np.convolve(baseband, taps, mode="full")
        delay = (taps.size - 1) // 2
        baseband = filtered[delay : delay + baseband.size]
    symbols, boundaries, phase = _timed_symbols(baseband, clock)

    if config.manchester:
        symbols = np.concatenate((np.zeros(1, dtype=symbols.dtype), symbols))
        boundaries = np.concatenate((boundaries[:1], boundaries))
        if config.manchester_phase is None:
            symbols, boundaries = _manchester_sync(
                symbols, boundaries, config.manchester_block_size
            )
        else:
            symbols, boundaries = _manchester_sync_fixed_phase(
                symbols, boundaries, config.manchester_phase
            )

    if config.differential:
        decisions = np.real(symbols[1:] * np.conj(symbols[:-1]))
        if config.manchester:
            decisions = -decisions
        boundaries = boundaries[1:]
    else:
        carrier_phase = 0.5 * np.angle(np.sum(np.square(symbols))) if symbols.size else 0.0
        decisions = np.real(symbols * np.exp(-1j * carrier_phase))
    soft = _normalize_soft(decisions)
    return BpskSymbols(soft >= 0, soft, boundaries, phase, frequency)


__all__ = [
    "BpskConfig",
    "BpskSymbols",
    "demodulate_bpsk",
    "manchester_sync_symbols",
    "root_raised_cosine_taps",
]

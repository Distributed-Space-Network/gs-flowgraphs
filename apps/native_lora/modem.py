"""Bounded LoRa chirp synchronization and aligned FFT demodulation.

SPDX-License-Identifier: GPL-3.0-only

Adapted on 2026-07-18 from tapparelj/gr-lora_sdr at commit
862746dd1cf635c9c8a4bfbaa2c3a0ec3a5306c9 (GPL-3.0-only), principally
``utilities.h``, ``frame_sync_impl.cc``, and ``fft_demod_impl.cc``.

This out-of-bench implementation intentionally stops at synthetic-vector
strength.  It estimates bounded integer/fractional CFO and fractional sample
timing/sample-frequency offset from repeated preamble chirps, validates the two
sync chirps and two full downchirps, applies timing compensation while
demodulating, and reports dechirped SNR plus source offsets and hard FFT bins.
Captured-radio parity remains an NF-MODEM-004 closure gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve, find_peaks

from .framing import MAX_SF, MIN_SF

MAX_OVERSAMPLING: Final = 16
DEFAULT_MAX_INPUT_SAMPLES: Final = 1_048_576
FRACTIONAL_BIN_ACQUISITION_FLOOR: Final = 0.35


class LoRaSyncError(ValueError):
    """No bounded, fully validated LoRa synchronization sequence was found."""


@dataclass(frozen=True)
class LoRaModemConfig:
    sf: int
    bandwidth: int
    sample_rate: int
    sync_word: int = 0x12
    preamble_length: int = 8
    inverted_iq: bool = False
    min_correlation: float = 0.55
    sync_tolerance_bins: int = 2
    max_abs_cfo_bins: float = 2.5
    max_abs_sto_samples: float = 1.0
    max_abs_sfo_ppm: float = 250.0
    min_snr_db: float = -8.0
    max_input_samples: int = DEFAULT_MAX_INPUT_SAMPLES

    def __post_init__(self) -> None:
        if not MIN_SF <= self.sf <= MAX_SF:
            raise ValueError(f"sf must be in {MIN_SF}..{MAX_SF}")
        if self.bandwidth <= 0 or self.sample_rate <= 0:
            raise ValueError("bandwidth and sample_rate must be positive")
        if self.sample_rate % self.bandwidth:
            raise ValueError("sample_rate must be an integer multiple of bandwidth")
        if not 1 <= self.oversampling <= MAX_OVERSAMPLING:
            raise ValueError(f"oversampling must be in 1..{MAX_OVERSAMPLING}")
        if not 0 <= self.sync_word <= 0xFF:
            raise ValueError("sync_word must be an octet")
        if self.preamble_length < 5:
            raise ValueError("preamble_length must be at least five")
        if not 0.0 < self.min_correlation <= 1.0:
            raise ValueError("min_correlation must be in (0, 1]")
        if not 0 <= self.sync_tolerance_bins <= 4:
            raise ValueError("sync_tolerance_bins must be in 0..4")
        if not 0.0 <= self.max_abs_cfo_bins <= 8.0:
            raise ValueError("max_abs_cfo_bins must be in 0..8")
        if not 0.0 <= self.max_abs_sto_samples <= self.oversampling:
            raise ValueError("max_abs_sto_samples must be in 0..oversampling")
        if not 0.0 <= self.max_abs_sfo_ppm <= 5_000.0:
            raise ValueError("max_abs_sfo_ppm must be in 0..5000")
        if not -40.0 <= self.min_snr_db <= 60.0:
            raise ValueError("min_snr_db must be in -40..60")
        if self.max_input_samples < self.minimum_sync_samples:
            raise ValueError("max_input_samples is smaller than the synchronization sequence")

    @property
    def oversampling(self) -> int:
        return self.sample_rate // self.bandwidth

    @property
    def bins(self) -> int:
        return 1 << self.sf

    @property
    def samples_per_symbol(self) -> int:
        return self.bins * self.oversampling

    @property
    def sync_symbols(self) -> tuple[int, int]:
        return (((self.sync_word >> 4) & 0xF) << 3, (self.sync_word & 0xF) << 3)

    @property
    def minimum_sync_samples(self) -> int:
        return (self.preamble_length + 4) * self.samples_per_symbol + (
            self.samples_per_symbol // 4
        )


@dataclass(frozen=True)
class LoRaSyncResult:
    preamble_start: int
    payload_start: int
    cfo_bins: float
    preamble_correlation: float
    sync_bins: tuple[int, int]
    downchirp_correlation: float
    inverted_iq: bool
    sto_samples: float
    sfo_ppm: float
    snr_db: float


@dataclass(frozen=True)
class LoRaDemodulatedSymbols:
    bins: tuple[int, ...]
    peak_ratios: tuple[float, ...]
    source_start: int
    source_end: int


def _as_complex_samples(samples: ArrayLike, *, limit: int) -> NDArray[np.complex128]:
    array = np.asarray(samples)
    if array.ndim != 1:
        raise ValueError("IQ samples must be one-dimensional")
    if len(array) > limit:
        raise ValueError("IQ input exceeds the configured sample bound")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("IQ samples must be numeric")
    output = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(output)):
        raise ValueError("IQ samples must be finite")
    return output


def build_upchirp(symbol: int, config: LoRaModemConfig) -> NDArray[np.complex128]:
    """Build the exact pinned gr-lora_sdr chirp for one symbol index."""

    if not isinstance(symbol, int) or isinstance(symbol, bool):
        raise TypeError("symbol must be an integer")
    if not 0 <= symbol < config.bins:
        raise ValueError(f"symbol must be in 0..{config.bins - 1}")
    bins = config.bins
    os_factor = config.oversampling
    count = config.samples_per_symbol
    n = np.arange(count, dtype=np.float64)
    fold = count - symbol * os_factor
    linear = np.where(n < fold, symbol / bins - 0.5, symbol / bins - 1.5)
    phase_cycles = n * n / (2 * bins * os_factor * os_factor) + linear * n / os_factor
    return np.exp(2j * np.pi * phase_cycles)


def _correct_iq(
    samples: NDArray[np.complex128], config: LoRaModemConfig, cfo_bins: float = 0.0
) -> NDArray[np.complex128]:
    corrected = np.conjugate(samples) if config.inverted_iq else samples
    return _apply_cfo(corrected, cfo_bins, config.samples_per_symbol)


def _apply_cfo(
    samples: NDArray[np.complex128], cfo_bins: float, samples_per_symbol: int
) -> NDArray[np.complex128]:
    if not cfo_bins:
        return samples
    n = np.arange(len(samples), dtype=np.float64)
    return samples * np.exp(-2j * np.pi * cfo_bins * n / samples_per_symbol)


def _fft_symbol(
    samples: NDArray[np.complex128], reference: NDArray[np.complex128]
) -> tuple[int, float, NDArray[np.complex128]]:
    spectrum = np.fft.fft(samples * reference)
    powers = np.abs(spectrum) ** 2
    peak = int(np.argmax(powers))
    total = float(np.sum(powers))
    ratio = float(powers[peak] / total) if total > 0.0 else 0.0
    return peak, ratio, spectrum


def _downsample_symbol(
    samples: NDArray[np.complex128], config: LoRaModemConfig
) -> NDArray[np.complex128]:
    # The matched-filter search already resolves the exact synthetic symbol
    # boundary, so the first polyphase arm is the zero-STO arm.  Selecting a
    # midpoint arm would introduce the half-bin term that gr-lora_sdr removes
    # later with its fractional-STO estimator.
    return samples[:: config.oversampling][: config.bins]


def _interpolate_complex(
    samples: NDArray[np.complex128], positions: NDArray[np.float64]
) -> NDArray[np.complex128]:
    if len(positions) == 0:
        return np.empty(0, dtype=np.complex128)
    if positions[0] < 0.0 or positions[-1] > len(samples) - 1:
        raise ValueError("timing-compensated window exceeds available IQ")
    indices = np.arange(len(samples), dtype=np.float64)
    return np.interp(positions, indices, samples.real) + 1j * np.interp(
        positions, indices, samples.imag
    )


def _timing_window(
    samples: NDArray[np.complex128],
    start: float,
    count: int,
    sfo_fraction: float,
) -> NDArray[np.complex128]:
    positions = start + np.arange(count, dtype=np.float64) * (1.0 + sfo_fraction)
    return _interpolate_complex(samples, positions)


def _timing_symbol(
    samples: NDArray[np.complex128],
    start: float,
    config: LoRaModemConfig,
    sfo_fraction: float,
) -> NDArray[np.complex128]:
    positions = start + (
        np.arange(config.bins, dtype=np.float64)
        * config.oversampling
        * (1.0 + sfo_fraction)
    )
    return _interpolate_complex(samples, positions)


def _parabolic_peak(scores: NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= len(scores) - 1:
        return float(index)
    left = float(scores[index - 1])
    middle = float(scores[index])
    right = float(scores[index + 1])
    denominator = left - 2.0 * middle + right
    if abs(denominator) <= np.finfo(np.float64).eps:
        return float(index)
    correction = 0.5 * (left - right) / denominator
    return float(index) + float(np.clip(correction, -0.5, 0.5))


def _estimate_timing(
    scores: NDArray[np.float64],
    start: int,
    config: LoRaModemConfig,
    *,
    enforce_sto: bool = True,
) -> tuple[float, float]:
    max_drift = (
        config.samples_per_symbol
        * config.preamble_length
        * config.max_abs_sfo_ppm
        / 1_000_000.0
    )
    radius = max(2, int(np.ceil(config.max_abs_sto_samples + max_drift)) + 1)
    observed: list[float] = []
    for index in range(config.preamble_length):
        expected = start + index * config.samples_per_symbol
        left = max(0, expected - radius)
        right = min(len(scores), expected + radius + 1)
        if left >= right:
            raise LoRaSyncError("preamble timing search exceeds available IQ")
        peak = left + int(np.argmax(scores[left:right]))
        observed.append(_parabolic_peak(scores, peak))
    symbol_indices = np.arange(config.preamble_length, dtype=np.float64)
    slope, intercept = np.polyfit(symbol_indices, np.asarray(observed), 1)
    sfo_fraction = float(slope / config.samples_per_symbol - 1.0)
    sfo_ppm = sfo_fraction * 1_000_000.0
    sto_samples = float(intercept - start)
    # Parabolic interpolation of an exactly aligned finite chirp can leave
    # sub-ppm/sub-milliperiod numerical residue.  Keep the exact legacy clock
    # in that deadband so zero-drift source offsets and CFO estimates do not
    # move merely because timing qualification is enabled.
    if abs(sfo_ppm) < 1.0:
        sfo_fraction = 0.0
        sfo_ppm = 0.0
    if abs(sto_samples) < 1e-3:
        intercept = float(start)
        sto_samples = 0.0
    if enforce_sto and abs(sto_samples) > config.max_abs_sto_samples + 0.05:
        raise LoRaSyncError("fractional STO exceeds configured bound")
    if abs(sfo_ppm) > config.max_abs_sfo_ppm + 1e-6:
        raise LoRaSyncError("SFO exceeds configured bound")
    return float(intercept), sfo_fraction


def _downchirp_reference(config: LoRaModemConfig) -> NDArray[np.complex128]:
    return np.conjugate(_downsample_symbol(build_upchirp(0, config), config))


def _signed_bin(index: int, width: int) -> int:
    return index if index <= width // 2 else index - width


def _circular_distance(left: int, right: int, modulus: int) -> int:
    delta = abs(left - right) % modulus
    return min(delta, modulus - delta)


def _estimate_cfo(
    samples: NDArray[np.complex128],
    start: float,
    config: LoRaModemConfig,
    sfo_fraction: float,
) -> float:
    sps = config.samples_per_symbol
    downchirp = _downchirp_reference(config)
    spectra: list[NDArray[np.complex128]] = []
    peaks: list[int] = []
    for index in range(config.preamble_length):
        window = _timing_symbol(
            samples,
            start + index * sps * (1.0 + sfo_fraction),
            config,
            sfo_fraction,
        )
        peak, _, spectrum = _fft_symbol(window, downchirp)
        peaks.append(peak)
        spectra.append(spectrum)
    integer_peak = int(np.median(peaks))
    integer_cfo = _signed_bin(integer_peak, config.bins)
    phase_sum = sum(
        spectra[index][integer_peak] * np.conjugate(spectra[index + 1][integer_peak])
        for index in range(len(spectra) - 1)
    )
    fractional_cfo = -float(np.angle(phase_sum)) / (2 * np.pi) if phase_sum else 0.0
    return integer_cfo + fractional_cfo


def _spectrum_snr_db(spectrum: NDArray[np.complex128]) -> float:
    powers = np.abs(spectrum) ** 2
    peak = float(np.max(powers))
    residual = float(np.sum(powers) - peak)
    if peak <= np.finfo(np.float64).tiny:
        return float("-inf")
    if residual <= np.finfo(np.float64).tiny:
        return float("inf")
    return float(10.0 * np.log10(peak / residual))


def _matched_scores(
    samples: NDArray[np.complex128], reference: NDArray[np.complex128]
) -> NDArray[np.float64]:
    correlation = fftconvolve(samples, np.conjugate(reference[::-1]), mode="valid")
    energy = fftconvolve(np.abs(samples) ** 2, np.ones(len(reference)), mode="valid")
    denominator = energy * float(np.vdot(reference, reference).real)
    return np.divide(
        np.abs(correlation) ** 2,
        denominator,
        out=np.zeros_like(energy, dtype=np.float64),
        where=denominator > np.finfo(np.float64).tiny,
    )


def _normalized_correlation(
    reference: NDArray[np.complex128], samples: NDArray[np.complex128]
) -> float:
    numerator = abs(np.vdot(reference, samples)) ** 2
    denominator = float(np.vdot(reference, reference).real) * float(
        np.vdot(samples, samples).real
    )
    return float(numerator / denominator) if denominator else 0.0


def find_lora_sync(samples: ArrayLike, config: LoRaModemConfig) -> LoRaSyncResult:
    """Find and validate the first complete LoRa preamble/sync sequence."""

    iq = _as_complex_samples(samples, limit=config.max_input_samples)
    if len(iq) < config.minimum_sync_samples:
        raise LoRaSyncError("IQ input is shorter than a complete synchronization sequence")
    iq = _correct_iq(iq, config)
    sps = config.samples_per_symbol
    upchirp = build_upchirp(0, config)
    initial_scores = _matched_scores(iq, upchirp)
    acquisition_threshold = min(
        config.min_correlation, FRACTIONAL_BIN_ACQUISITION_FLOOR
    )
    peaks, properties = find_peaks(
        initial_scores,
        height=acquisition_threshold,
        distance=max(1, sps // 2),
    )
    if len(peaks) == 0 and config.max_abs_cfo_bins >= 0.5:
        # A half-bin residual still has ample coherent gain.  Use the small
        # bounded CFO bank only as a fallback so the common near-zero-CFO path
        # pays for one convolution rather than eleven.
        reference_index = np.arange(sps, dtype=np.float64)
        limit = int(np.floor(config.max_abs_cfo_bins * 2.0))
        for half_bin in range(-limit, limit + 1):
            if half_bin == 0:
                continue
            cfo_reference = upchirp * np.exp(
                2j * np.pi * (half_bin / 2.0) * reference_index / sps
            )
            initial_scores = np.maximum(
                initial_scores, _matched_scores(iq, cfo_reference)
            )
        peaks, properties = find_peaks(
            initial_scores,
            height=config.min_correlation,
            distance=max(1, sps // 2),
        )
    if len(peaks) == 0:
        raise LoRaSyncError("no LoRa preamble correlation exceeded the threshold")

    downchirp = _downchirp_reference(config)
    downchirp_wave = np.conjugate(upchirp)
    alignment_radius = int(np.ceil(config.max_abs_cfo_bins * config.oversampling)) + 1
    alignment_deltas = sorted(
        range(-alignment_radius, alignment_radius + 1), key=lambda value: (abs(value), value)
    )
    attempted: set[int] = set()
    best_result: LoRaSyncResult | None = None
    best_quality: tuple[float, float, float] | None = None
    for peak in peaks:
        for delta in alignment_deltas:
            start = int(peak) + delta
            if start in attempted or start < 0:
                continue
            attempted.add(start)
            last_required = start + config.minimum_sync_samples
            if last_required > len(iq):
                continue
            try:
                _, sfo_fraction = _estimate_timing(
                    initial_scores, start, config, enforce_sto=False
                )
            except LoRaSyncError:
                continue
            positions = float(start) + (
                np.arange(config.preamble_length) * sps * (1.0 + sfo_fraction)
            )
            rounded_positions = np.rint(positions).astype(np.int64)
            if rounded_positions[-1] >= len(initial_scores):
                continue
            acquisition_scores = np.asarray(
                [
                    np.max(
                        initial_scores[
                            max(0, position - alignment_radius) : min(
                                len(initial_scores), position + alignment_radius + 1
                            )
                        ]
                    )
                    for position in rounded_positions
                ]
            )
            if np.any(acquisition_scores < acquisition_threshold):
                continue

            try:
                cfo_bins = _estimate_cfo(iq, float(start), config, sfo_fraction)
            except ValueError:
                continue
            if abs(cfo_bins) > config.max_abs_cfo_bins:
                continue
            corrected = _apply_cfo(iq, cfo_bins, sps)
            corrected_scores = _matched_scores(corrected, upchirp)
            try:
                timing_start, sfo_fraction = _estimate_timing(
                    corrected_scores, start, config
                )
            except LoRaSyncError:
                continue
            positions = timing_start + (
                np.arange(config.preamble_length) * sps * (1.0 + sfo_fraction)
            )
            preamble_scores: list[float] = []
            preamble_snr: list[float] = []
            try:
                for position in positions:
                    timing_window = _timing_window(
                        corrected, float(position), sps, sfo_fraction
                    )
                    preamble_scores.append(
                        _normalized_correlation(upchirp, timing_window)
                    )
                    timing_symbol = _timing_symbol(
                        corrected, float(position), config, sfo_fraction
                    )
                    _, _, spectrum = _fft_symbol(timing_symbol, downchirp)
                    preamble_snr.append(_spectrum_snr_db(spectrum))
            except ValueError:
                continue
            preamble_scores_array = np.asarray(preamble_scores)
            if np.any(preamble_scores_array < config.min_correlation):
                continue
            finite_snr = [value for value in preamble_snr if np.isfinite(value)]
            snr_db = (
                float(np.mean(finite_snr)) if finite_snr else float("inf")
            )
            if snr_db < config.min_snr_db:
                continue

            sync_bins: list[int] = []
            sync_ok = True
            for index, expected in enumerate(config.sync_symbols):
                offset = timing_start + (
                    (config.preamble_length + index)
                    * sps
                    * (1.0 + sfo_fraction)
                )
                try:
                    window = _timing_symbol(
                        corrected, offset, config, sfo_fraction
                    )
                except ValueError:
                    sync_ok = False
                    break
                actual, _, _ = _fft_symbol(window, downchirp)
                actual %= config.bins
                sync_bins.append(actual)
                if (
                    _circular_distance(actual, expected, config.bins)
                    > config.sync_tolerance_bins
                ):
                    sync_ok = False
            if not sync_ok:
                continue

            down_scores: list[float] = []
            for index in range(2):
                offset = timing_start + (
                    (config.preamble_length + 2 + index)
                    * sps
                    * (1.0 + sfo_fraction)
                )
                try:
                    window = _timing_window(
                        corrected, offset, sps, sfo_fraction
                    )
                except ValueError:
                    down_scores = []
                    break
                down_scores.append(_normalized_correlation(downchirp_wave, window))
            if len(down_scores) != 2 or min(down_scores) < config.min_correlation:
                continue

            payload_start_float = timing_start + (
                (config.preamble_length + 4.25)
                * sps
                * (1.0 + sfo_fraction)
            )
            canonical_start = int(round(timing_start))
            sto_samples = float(timing_start - canonical_start)
            payload_start = int(np.floor(payload_start_float + 1e-9))
            candidate = LoRaSyncResult(
                preamble_start=canonical_start,
                payload_start=payload_start,
                cfo_bins=cfo_bins,
                preamble_correlation=float(np.mean(preamble_scores_array)),
                sync_bins=(sync_bins[0], sync_bins[1]),
                downchirp_correlation=float(np.mean(down_scores)),
                inverted_iq=config.inverted_iq,
                sto_samples=sto_samples,
                sfo_ppm=float(sfo_fraction * 1_000_000.0),
                snr_db=snr_db,
            )
            quality = (
                -abs(float(timing_start - start)),
                min(down_scores),
                float(np.mean(preamble_scores_array)),
            )
            if best_quality is None or quality > best_quality:
                best_result = candidate
                best_quality = quality

    if best_result is not None:
        return best_result

    peak_score = float(np.max(properties["peak_heights"]))
    raise LoRaSyncError(f"preamble candidate failed sync validation (peak={peak_score:.3f})")


def demodulate_payload_bins(
    samples: ArrayLike,
    config: LoRaModemConfig,
    sync: LoRaSyncResult,
    symbol_count: int,
) -> LoRaDemodulatedSymbols:
    """Demodulate aligned payload chirps to pre-Gray framing-bin values."""

    if not 1 <= symbol_count <= 4096:
        raise ValueError("symbol_count must be in 1..4096")
    iq = _as_complex_samples(samples, limit=config.max_input_samples)
    sfo_fraction = sync.sfo_ppm / 1_000_000.0
    timing_start = sync.preamble_start + sync.sto_samples
    payload_start = timing_start + (
        (config.preamble_length + 4.25)
        * config.samples_per_symbol
        * (1.0 + sfo_fraction)
    )
    end_float = payload_start + (
        symbol_count * config.samples_per_symbol * (1.0 + sfo_fraction)
    )
    last_sample = payload_start + (
        ((symbol_count - 1) * config.samples_per_symbol)
        + ((config.bins - 1) * config.oversampling)
    ) * (1.0 + sfo_fraction)
    if payload_start < 0.0 or last_sample > len(iq) - 1 + 1e-9:
        raise ValueError("IQ input does not contain the requested aligned payload symbols")
    corrected = _correct_iq(iq, config, sync.cfo_bins)
    downchirp = _downchirp_reference(config)
    bins: list[int] = []
    peak_ratios: list[float] = []
    for index in range(symbol_count):
        start = payload_start + (
            index * config.samples_per_symbol * (1.0 + sfo_fraction)
        )
        window = _timing_symbol(
            corrected, start, config, sfo_fraction
        )
        peak, ratio, _ = _fft_symbol(window, downchirp)
        bins.append((peak - 1) % config.bins)
        peak_ratios.append(ratio)
    return LoRaDemodulatedSymbols(
        bins=tuple(bins),
        peak_ratios=tuple(peak_ratios),
        source_start=sync.payload_start,
        source_end=min(len(iq), int(np.ceil(end_float))),
    )


def normalize_framing_symbols(
    bins: tuple[int, ...] | list[int], *, sf: int, ldro: bool
) -> tuple[int, ...]:
    """Apply gr-lora_sdr's reduced-rate division at the framing boundary."""

    if not MIN_SF <= sf <= MAX_SF:
        raise ValueError(f"sf must be in {MIN_SF}..{MAX_SF}")
    modulus = 1 << sf
    normalized: list[int] = []
    for index, value in enumerate(bins):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < modulus:
            raise ValueError(f"bin {index} must be in 0..{modulus - 1}")
        normalized.append(value // 4 if index < 8 or ldro else value)
    return tuple(normalized)


@dataclass
class LoRaSyncBuffer:
    """Bounded chunk adapter that preserves absolute source offsets."""

    config: LoRaModemConfig
    _chunks: list[NDArray[np.complex128]] = field(default_factory=list, init=False)
    _sample_count: int = field(default=0, init=False)

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def feed(self, samples: ArrayLike) -> LoRaSyncResult | None:
        chunk = _as_complex_samples(samples, limit=self.config.max_input_samples)
        if self._sample_count + len(chunk) > self.config.max_input_samples:
            raise ValueError("buffered IQ exceeds the configured sample bound")
        if len(chunk):
            self._chunks.append(chunk.copy())
            self._sample_count += len(chunk)
        if self._sample_count < self.config.minimum_sync_samples:
            return None
        combined = (
            np.concatenate(self._chunks)
            if self._chunks
            else np.empty(0, dtype=np.complex128)
        )
        try:
            return find_lora_sync(combined, self.config)
        except LoRaSyncError:
            return None

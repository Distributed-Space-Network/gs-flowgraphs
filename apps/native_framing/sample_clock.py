"""Exact sample/symbol clocks and receive-rate planning.

The legacy sample-rate compatibility helpers reproduce the observable behavior of
``satnogsclient/radio/grsat.py`` at pinned commit
``60d9902933d86a6133935586a0da4952a5803f9e``.  That AGPL source is comparison-only;
this is a repository-owned implementation expressed from the behavior contract.

Offset conversion uses rational arithmetic and always maps an absolute offset.  It
therefore cannot accumulate the one-sample drift that repeated floating-point or
per-window increments can introduce.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

DEFAULT_MIN_SAMPLES_PER_SYMBOL = 4.0


def _rate_fraction(name: str, value: object, *, allow_zero: bool = False) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return Fraction(str(numeric))


def _offset(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _round_nonnegative(value: Fraction) -> int:
    """Round a non-negative rational to nearest, with exact half values rounded up."""

    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def convert_offset(
    offset: int,
    *,
    from_rate_hz: float,
    to_rate_hz: float,
) -> int:
    """Convert one absolute offset between clock domains without cumulative drift."""

    source_offset = _offset("offset", offset)
    source_rate = _rate_fraction("from_rate_hz", from_rate_hz)
    target_rate = _rate_fraction("to_rate_hz", to_rate_hz)
    return _round_nonnegative(source_offset * target_rate / source_rate)


@dataclass(frozen=True)
class SampleClock:
    """Exact relationship between demodulated symbols and capture samples."""

    sample_rate_hz: float
    symbol_rate_hz: float
    minimum_samples_per_symbol: float = 2.0
    _sample_rate: Fraction = field(init=False, repr=False, compare=False)
    _symbol_rate: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sample_rate = _rate_fraction("sample_rate_hz", self.sample_rate_hz)
        symbol_rate = _rate_fraction("symbol_rate_hz", self.symbol_rate_hz)
        minimum = _rate_fraction(
            "minimum_samples_per_symbol", self.minimum_samples_per_symbol
        )
        samples_per_symbol = sample_rate / symbol_rate
        if samples_per_symbol < minimum:
            raise ValueError(
                f"sample rate {float(sample_rate):g} Hz provides only "
                f"{float(samples_per_symbol):g} samples/symbol; at least "
                f"{float(minimum):g} are required"
            )
        object.__setattr__(self, "_sample_rate", sample_rate)
        object.__setattr__(self, "_symbol_rate", symbol_rate)

    @property
    def samples_per_symbol(self) -> float:
        return float(self._sample_rate / self._symbol_rate)

    def sample_offset_for_symbol(self, symbol_offset: int, *, origin_sample: int = 0) -> int:
        """Map an absolute symbol offset into the capture clock."""

        symbol = _offset("symbol_offset", symbol_offset)
        origin = _offset("origin_sample", origin_sample)
        return origin + _round_nonnegative(symbol * self._sample_rate / self._symbol_rate)

    def symbol_offset_for_sample(self, sample_offset: int) -> int:
        """Map an absolute capture offset into the nearest demodulated symbol."""

        sample = _offset("sample_offset", sample_offset)
        return _round_nonnegative(sample * self._symbol_rate / self._sample_rate)

    def elapsed_seconds_for_sample(self, sample_offset: int) -> Fraction:
        """Return an exact rational duration for a capture offset."""

        sample = _offset("sample_offset", sample_offset)
        return sample / self._sample_rate


def legacy_satnogs_decimation(
    symbol_rate_hz: float,
    *,
    minimum: int = 4,
    audio_sample_rate_hz: float = 48_000.0,
    multiple: int = 2,
) -> int:
    """Return the pinned SatNOGS helper's integer decimation/SPS selection."""

    symbol_rate = _rate_fraction("symbol_rate_hz", symbol_rate_hz)
    audio_rate = _rate_fraction("audio_sample_rate_hz", audio_sample_rate_hz)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("minimum must be a positive integer")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    required = math.ceil(audio_rate / symbol_rate)
    selected = max(minimum, required)
    return selected + (-selected % multiple)


def legacy_satnogs_sample_rate(
    baudrate: object,
    *,
    script: str = "",
    sps: int = 4,
    audio_sample_rate_hz: float = 48_000.0,
) -> float:
    """Reproduce the pinned SatNOGS script-family sample-rate matrix.

    Invalid or sub-one baud values use the historical 9600-baud fallback.  The
    script name is used only to select the same observable family policy.
    """

    try:
        numeric = float(baudrate)
        baud = int(numeric) if math.isfinite(numeric) else 9_600
    except (TypeError, ValueError, OverflowError):
        baud = 9_600
    if baud < 1:
        baud = 9_600
    if isinstance(sps, bool) or not isinstance(sps, int) or sps <= 0:
        raise ValueError("sps must be a positive integer")
    audio_rate = float(_rate_fraction("audio_sample_rate_hz", audio_sample_rate_hz))
    name = str(script or "").casefold()
    if "_bpsk" in name or "_ssb" in name:
        return float(
            legacy_satnogs_decimation(
                baud,
                minimum=2,
                audio_sample_rate_hz=audio_rate,
                multiple=sps,
            )
            * baud
        )
    if "_fsk" in name or "_qubik" in name:
        return float(
            max(
                4,
                legacy_satnogs_decimation(
                    baud,
                    minimum=2,
                    audio_sample_rate_hz=audio_rate,
                    multiple=2,
                ),
            )
            * baud
        )
    if "_sstv" in name or "_apt" in name:
        return float(4 * 4_160 * 4)
    return audio_rate


def select_channel_rate(
    requested_rate_hz: float,
    symbol_rate_hz: float | None,
    capture_rate_hz: float,
    *,
    minimum_samples_per_symbol: float = DEFAULT_MIN_SAMPLES_PER_SYMBOL,
) -> float:
    """Select a bounded channel rate and reject an impossible demodulation clock.

    High-baud widening snaps upward to an integer decimation of an integral capture
    rate.  Unlike the historical helper, a capture that cannot provide the required
    samples/symbol is rejected instead of silently capped to a non-decodable rate.
    """

    requested = _rate_fraction("requested_rate_hz", requested_rate_hz)
    capture = _rate_fraction("capture_rate_hz", capture_rate_hz)
    symbol_value = 0.0 if symbol_rate_hz is None else symbol_rate_hz
    symbol = _rate_fraction("symbol_rate_hz", symbol_value, allow_zero=True)
    minimum = _rate_fraction(
        "minimum_samples_per_symbol", minimum_samples_per_symbol
    )
    if requested > capture:
        raise ValueError("requested channel rate exceeds the capture rate")
    required = symbol * minimum
    if required > capture:
        raise ValueError(
            f"capture rate {float(capture):g} Hz cannot provide "
            f"{float(minimum):g} samples/symbol at {float(symbol):g} baud"
        )
    want = max(requested, required)
    if want <= requested:
        return float(requested)
    if want == capture:
        return float(capture)

    capture_float = float(capture)
    capture_integer = round(capture_float)
    if not math.isclose(capture_float, capture_integer, rel_tol=0.0, abs_tol=1e-9):
        return float(want)
    decimation = math.floor(capture / want)
    while decimation > 1 and capture_integer % decimation:
        decimation -= 1
    return capture_float / decimation if decimation >= 1 else float(want)


__all__ = [
    "DEFAULT_MIN_SAMPLES_PER_SYMBOL",
    "SampleClock",
    "convert_offset",
    "legacy_satnogs_decimation",
    "legacy_satnogs_sample_rate",
    "select_channel_rate",
]

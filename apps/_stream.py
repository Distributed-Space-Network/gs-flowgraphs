"""Streaming DSP + queue primitives shared by the dsp-engine apps (X-02).

R-13/R-14: the per-app ad-hoc versions of these were wrong in ways that only
show at chunk boundaries or on real hardware:

* NCO phase was carried as ``ph[-1]`` — the LAST sample's phase — so every
  chunk boundary REPEATED one phase step instead of advancing it
  (:func:`apply_nco_chunk` advances and wraps correctly).
* ``scipy.signal.resample_poly`` was re-run per chunk with no state, resetting
  the filter at every boundary (:class:`StreamingDecimator` carries FIR state
  across chunks and is sample-exact vs. one-shot processing).
* Modem IQ rates (~96 kHz) were applied directly to XTRX-class hardware whose
  RX floor is ~2.1 Msps (:func:`hardware_rate_for` picks a supported rate that
  is an INTEGER multiple of the modem rate, so the decimation/interpolation
  factor is exact).
* A bounded asyncio queue was fed with ``call_soon_threadsafe(put_nowait)`` —
  overflow silently dropped chunks AND could drop the ``None`` terminator
  (:func:`make_backpressure_put` blocks the producer thread with backpressure
  and stays interruptible so teardown can't hang).

Import-safe: numpy + scipy only (scipy imports are lazy).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
from collections.abc import Callable

import numpy as np

_TWO_PI = 2.0 * math.pi


def apply_nco_chunk(
    chunk: np.ndarray,
    offset_hz: float,
    sample_rate_hz: float,
    phase: float,
) -> tuple[np.ndarray, float]:
    """Mix ``chunk`` DOWN by ``offset_hz`` (multiply by ``e^{j(phase - 2πf n/fs)}``),
    phase-continuous across chunks. Returns ``(mixed_chunk, next_phase)`` where
    ``next_phase`` is the phase of the FIRST sample of the NEXT chunk — i.e.
    advanced one step past the last sample, not a repeat of it (R-13). The
    phase is wrapped so a long pass can't degrade float precision."""
    if not offset_hz or not len(chunk):
        return chunk, phase
    n = np.arange(len(chunk))
    ph = phase - _TWO_PI * offset_hz * n / sample_rate_hz
    out = (chunk * np.exp(1j * ph)).astype(np.complex64)
    next_phase = math.remainder(
        phase - _TWO_PI * offset_hz * len(chunk) / sample_rate_hz, _TWO_PI
    )
    return out, next_phase


def hardware_rate_for(modem_rate_hz: float, min_hardware_rate_hz: float) -> tuple[float, int]:
    """R-14: pick the SDR rate for a modem that wants ``modem_rate_hz``:
    the smallest INTEGER multiple of the modem rate at or above the hardware
    floor (e.g. 96 kHz modem, 2.048 Msps floor → 2.112 Msps, factor 22), so
    decimation/interpolation is exact. ``min_hardware_rate_hz`` <= modem rate
    (or 0 = "hardware streams anything") keeps the modem rate directly.
    Returns ``(hardware_rate_hz, factor)``."""
    if min_hardware_rate_hz <= modem_rate_hz:
        return float(modem_rate_hz), 1
    factor = int(math.ceil(min_hardware_rate_hz / float(modem_rate_hz)))
    return float(modem_rate_hz) * factor, factor


class StreamingDecimator:
    """Integer-factor anti-aliased decimator with FIR state carried across
    chunks — chunked output is sample-exact vs. processing the whole stream
    in one call (R-13: the stateless per-chunk ``resample_poly`` reset its
    filter at every boundary)."""

    def __init__(self, factor: int, *, ntaps: int | None = None) -> None:
        if factor < 1:
            msg = f"decimation factor must be >= 1, got {factor}"
            raise ValueError(msg)
        self.factor = factor
        if factor == 1:
            self._taps = None
            return
        from scipy.signal import firwin  # noqa: PLC0415 — lazy; keeps import cheap

        # Anti-alias LPF at the new Nyquist; 8 taps per decimation arm is the
        # resample_poly-class default trade-off. The filter runs at the FULL
        # input rate via C lfilter (correctness-first; a polyphase that skips
        # the discarded outputs is a bench-profiling optimization — the dsp
        # engine is the backup path, GR is the production default).
        ntaps = ntaps or (8 * factor + 1)
        self._taps = firwin(ntaps, 1.0 / factor).astype(np.float64)
        # FIR state (last ntaps-1 samples) carried across chunks by lfilter's zi.
        self._zi = np.zeros(len(self._taps) - 1, dtype=np.complex128)
        # Output-phase bookkeeping so exactly every ``factor``-th filtered
        # sample is emitted across chunk boundaries.
        self._skip = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return np.asarray(chunk, dtype=np.complex64)
        x = np.asarray(chunk, dtype=np.complex64)
        if not len(x):
            return x
        from scipy.signal import lfilter  # noqa: PLC0415 — lazy

        filtered, self._zi = lfilter(self._taps, 1.0, x, zi=self._zi)
        out = filtered[self._skip :: self.factor]
        consumed = len(filtered) - self._skip
        self._skip = (-consumed) % self.factor
        return out.astype(np.complex64)


def upsample_burst(iq: np.ndarray, factor: int) -> np.ndarray:
    """R-14 TX side: interpolate a SELF-CONTAINED burst from the modem rate to
    the hardware rate by an integer factor. A burst has defined start/end, so
    one-shot polyphase interpolation is correct here (no cross-burst state)."""
    if factor <= 1:
        return np.asarray(iq, dtype=np.complex64)
    from scipy.signal import resample_poly  # noqa: PLC0415 — lazy

    return resample_poly(np.asarray(iq, dtype=np.complex64), factor, 1).astype(np.complex64)


def require_sample_rate(
    dev: object, direction: object, channel: int, rate_hz: float, *, tolerance: float = 0.01
) -> float:
    """R-14 readback validation: after ``setSampleRate``, prove the device
    actually RUNS at the requested rate — a driver that silently clamps an
    unsupported rate desynchronizes the whole modem. Raises ``RuntimeError``
    on a >1% mismatch (fail closed at spawn, R-11). A device without a
    readable rate returns the requested value (can't validate — logged by
    the caller's readback report instead)."""
    getter = getattr(dev, "getSampleRate", None)
    if getter is None:
        return float(rate_hz)
    actual = float(getter(direction, channel))
    if abs(actual - rate_hz) > tolerance * rate_hz:
        msg = (
            f"SDR did not accept sample rate {rate_hz:.0f} Hz "
            f"(readback {actual:.0f} Hz) — modem/hardware rate plan invalid"
        )
        raise RuntimeError(msg)
    return actual


def make_backpressure_put(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_requested: asyncio.Event,
    *,
    poll_s: float = 0.1,
) -> Callable[[object], None]:
    """A producer-thread ``put(item)`` for a bounded asyncio queue: blocks the
    reader thread with BACKPRESSURE (a fast source can't overflow the queue or
    lose the ``None`` terminator), but stays interruptible — if the consumer
    stopped draining, it polls ``stop_requested`` and bails (dropping the
    item) so teardown can't hang (R-13)."""

    def _put(item: object) -> None:
        fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while True:
            try:
                fut.result(timeout=poll_s)
                return
            except concurrent.futures.TimeoutError:
                if stop_requested.is_set():
                    fut.cancel()  # tearing down and nobody is draining
                    return
            except concurrent.futures.CancelledError:
                return

    return _put


__all__ = [
    "StreamingDecimator",
    "apply_nco_chunk",
    "hardware_rate_for",
    "make_backpressure_put",
    "require_sample_rate",
    "upsample_burst",
]

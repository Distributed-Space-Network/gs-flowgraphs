"""X-02 sub-slice A regressions (R-13/R-14): the shared streaming primitives
are chunking-invariant — NCO phase advances across boundaries, the decimator
carries FIR state, rates are exact integer plans with validated readback, and
the producer-side queue put can neither overflow nor lose the terminator."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
from _stream import (
    StreamingDecimator,
    apply_nco_chunk,
    hardware_rate_for,
    make_backpressure_put,
    require_sample_rate,
    upsample_burst,
)

_FS = 96_000.0


def _tone(freq_hz: float, n: int, fs: float = _FS) -> np.ndarray:
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * freq_hz * t).astype(np.complex64)


# ------------------------------------------------------------------- NCO


def test_nco_is_chunking_invariant() -> None:
    """THE R-13 repro: the old carry reused the LAST sample's phase as the
    next chunk's start, so every boundary repeated one phase step. Correct
    carry makes chunked processing IDENTICAL to one-shot processing."""
    x = _tone(1500.0, 4096)
    whole, _ = apply_nco_chunk(x, 1500.0, _FS, 0.0)
    phase = 0.0
    parts = []
    for start in range(0, len(x), 313):  # deliberately odd chunk size
        out, phase = apply_nco_chunk(x[start : start + 313], 1500.0, _FS, phase)
        parts.append(out)
    chunked = np.concatenate(parts)
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-5)


def test_nco_shifts_tone_to_dc() -> None:
    x = _tone(2000.0, 8192)
    out, _ = apply_nco_chunk(x, 2000.0, _FS, 0.0)
    # After mixing down by the tone frequency the result is ~constant (DC).
    assert np.std(np.angle(out[1:] / out[:-1])) < 1e-3


def test_nco_zero_offset_and_empty_chunk_are_passthrough() -> None:
    x = _tone(100.0, 64)
    out, phase = apply_nco_chunk(x, 0.0, _FS, 1.23)
    assert out is x and phase == 1.23
    out2, phase2 = apply_nco_chunk(x[:0], 500.0, _FS, 0.5)
    assert len(out2) == 0 and phase2 == 0.5


# -------------------------------------------------------------- decimator


def test_decimator_is_chunking_invariant() -> None:
    """R-13: per-chunk stateless resample_poly reset the filter every chunk;
    the streaming decimator must produce EXACTLY the same samples regardless
    of how the input is sliced."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(10_000) + 1j * rng.standard_normal(10_000)).astype(np.complex64)
    one_shot = StreamingDecimator(8).process(x)
    dec = StreamingDecimator(8)
    parts = [dec.process(x[s : s + 997]) for s in range(0, len(x), 997)]
    chunked = np.concatenate([p for p in parts if len(p)])
    n = min(len(one_shot), len(chunked))
    np.testing.assert_allclose(chunked[:n], one_shot[:n], rtol=0, atol=1e-6)
    assert abs(len(one_shot) - len(chunked)) <= 1  # tail rounding only


def test_decimator_passes_inband_and_rejects_alias() -> None:
    factor = 8
    fs = 2_112_000.0  # 22 x 96k
    inband = _tone(5_000.0, 65_536, fs)
    alias = _tone(fs / factor * 0.9, 65_536, fs)  # lands out-of-band post-decim
    dec_a, dec_b = StreamingDecimator(factor), StreamingDecimator(factor)
    out_inband = dec_a.process(inband)
    out_alias = dec_b.process(alias)
    p_in = float(np.mean(np.abs(out_inband[200:]) ** 2))
    p_alias = float(np.mean(np.abs(out_alias[200:]) ** 2))
    assert p_in > 0.9  # in-band tone survives (~unit power)
    assert 10 * np.log10(p_alias / p_in) < -40  # alias crushed by the LPF


def test_decimator_factor_one_is_passthrough() -> None:
    x = _tone(100.0, 128)
    out = StreamingDecimator(1).process(x)
    np.testing.assert_array_equal(out, x)


# ------------------------------------------------------------- rate plans


def test_hardware_rate_is_exact_integer_multiple() -> None:
    hw, k = hardware_rate_for(96_000, 2_048_000)
    assert hw == 96_000 * 22 and k == 22  # 2.112 Msps — first multiple >= floor
    assert hw >= 2_048_000
    assert hardware_rate_for(96_000, 0) == (96_000.0, 1)  # env=0: RTL-class
    assert hardware_rate_for(96_000, 96_000) == (96_000.0, 1)
    assert hardware_rate_for(2_000_000, 2_048_000) == (4_000_000.0, 2)


class _RateDev:
    def __init__(self, actual: float) -> None:
        self._actual = actual

    def getSampleRate(self, direction: int, ch: int) -> float:  # noqa: N802
        return self._actual


def test_require_sample_rate_validates_readback() -> None:
    assert require_sample_rate(_RateDev(2_112_000.0), 1, 0, 2_112_000.0) == 2_112_000.0
    with pytest.raises(RuntimeError, match="did not accept sample rate"):
        require_sample_rate(_RateDev(2_048_000.0), 1, 0, 2_112_000.0)  # clamped
    assert require_sample_rate(object(), 1, 0, 96_000.0) == 96_000.0  # no getter


def test_upsample_burst_length_and_tone() -> None:
    x = _tone(1_000.0, 4096)
    up = upsample_burst(x, 4)
    assert len(up) == 4 * len(x)
    # The tone reappears at the same absolute frequency at the higher rate.
    spec = np.abs(np.fft.fft(up * np.hanning(len(up))))
    peak_bin = int(np.argmax(spec[: len(up) // 2]))
    expected_bin = round(1_000.0 * len(up) / (_FS * 4))
    assert abs(peak_bin - expected_bin) <= 1
    assert upsample_burst(x, 1) is not None and len(upsample_burst(x, 1)) == len(x)


# ---------------------------------------------------------- queue put


def test_backpressure_put_loses_nothing_and_delivers_sentinel() -> None:
    async def run() -> list[object]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        stop = asyncio.Event()
        put = make_backpressure_put(queue, loop, stop, poll_s=0.01)

        def producer() -> None:
            for i in range(64):  # floods a 4-slot queue
                put(i)
            put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        got: list[object] = []
        while True:
            item = await queue.get()
            if item is None:
                break
            got.append(item)
        t.join(timeout=5)
        return got

    got = asyncio.run(run())
    assert got == list(range(64))  # nothing lost, order kept, sentinel arrived


def test_backpressure_put_bails_on_stop_instead_of_hanging() -> None:
    async def run() -> bool:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        stop = asyncio.Event()
        put = make_backpressure_put(queue, loop, stop, poll_s=0.01)
        await queue.put("filler")  # nobody will ever drain this
        done = threading.Event()

        def producer() -> None:
            put("blocked-item")  # would park forever without the stop poll
            done.set()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        await asyncio.sleep(0.05)
        assert not done.is_set()  # genuinely blocked on the full queue
        stop.set()
        for _ in range(100):
            if done.is_set():
                break
            await asyncio.sleep(0.01)
        t.join(timeout=5)
        return done.is_set()

    assert asyncio.run(run()) is True

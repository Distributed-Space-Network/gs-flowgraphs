"""R2-17 — a dead SDR must not look like a clean end-of-pass.

The bidir RX reader caught its own exception and its ``finally`` pushed the SAME ``None``
terminator an exhausted stream uses, so ``run_rx`` returned NORMALLY after the SDR died:
zero frames, no error event, a completed pass. A dead radio was indistinguishable from a
quiet sky.

The first fix only CAPTURED the exception into a list, and the "test" I wrote for it
grepped the source for ``reader_error.append(e)``. It passed an implementation that never
raised. This test DRIVES ``run_rx`` with an IQ source that dies, and requires the failure
to propagate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

import cubesat_gfsk_endurosat_bidir as bidir  # noqa: E402
from _spawn_contract import EngineFailure  # noqa: E402


class _Writer:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, data: bytes) -> None:
        import json

        for line in data.decode().splitlines():
            if line.strip():
                self.events.append(json.loads(line))

    async def drain(self) -> None:
        return None


class _Sockets:
    def __init__(self) -> None:
        self.status_writer = _Writer()
        self.data_writer = _Writer()


class _DyingIO:
    """An IQ source that yields a couple of chunks and then dies — a real SDR failure
    (USB drop, driver error) mid-pass, not an exhausted stream."""

    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks

    def rx_chunks(self):
        for _ in range(self._chunks):
            yield np.zeros(256, dtype=np.complex64)
        msg = "SoapySDR: device disconnected"
        raise OSError(msg)

    def close(self) -> None:
        return None


class _HealthyIO:
    """A source that ends CLEANLY (the stream is simply exhausted)."""

    def rx_chunks(self):
        for _ in range(2):
            yield np.zeros(256, dtype=np.complex64)

    def close(self) -> None:
        return None


class _Tx:
    def __init__(self) -> None:
        self.tx_active = asyncio.Event()


class _Args:
    sample_rate = 96_000
    center_freq_hz = 401_500_000
    record_iq = None
    record_formats = ""
    output_dir = None
    sdr_args = "driver=null"


async def _drive(io_obj: object) -> _Sockets:
    sockets = _Sockets()
    stop = asyncio.Event()
    await bidir.run_rx(
        _Args(), sockets, {}, io_obj,
        stop_requested=stop, doppler={"hz": 0.0}, tx=_Tx(),
    )
    return sockets


class TestADeadSdrFailsThePass:
    def test_a_reader_that_dies_raises_engine_failure(self) -> None:
        """THE REPRO: run_rx used to return normally here — the pass then completed with
        zero frames and nobody could tell the radio had died."""
        with pytest.raises(EngineFailure, match="IQ source DIED"):
            asyncio.run(_drive(_DyingIO()))

    def test_the_cause_is_preserved(self) -> None:
        try:
            asyncio.run(_drive(_DyingIO()))
        except EngineFailure as e:
            assert isinstance(e.__cause__, OSError)
        else:  # pragma: no cover
            pytest.fail("a dead SDR must not end the pass cleanly")

    def test_a_clean_end_of_stream_is_still_a_clean_end_of_stream(self) -> None:
        """The other half of the invariant: an exhausted stream must NOT be turned into a
        failure. A guard that fires on the happy path is worse than no guard."""
        asyncio.run(_drive(_HealthyIO()))  # must not raise


class TestTheEngineDeathWatcherIsWired:
    def test_the_bidir_app_watches_its_rx_task(self) -> None:
        """Raising is only half of it: the exception has to REACH the process exit. The
        bidir app was the one engine that never wired watch_engine_death, so its rx_task
        exception was swallowed by gather(return_exceptions=True)."""
        src = (_APPS / "cubesat_gfsk_endurosat_bidir.py").read_text(encoding="utf-8")
        assert "watch_engine_death(rx_task" in src

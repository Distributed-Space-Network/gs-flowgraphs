"""CA-FLOW-005 — engine shutdown must never block on a full data queue.

The FM app's ``_shutdown_engine`` used a blocking ``data_queue.put(None)``. When
the data peer had already closed (the pump exits on ConnectionReset — nothing
consumes) and the GR producer had filled the bounded queue, that put blocked the
asyncio event loop forever: no stopped ack, a frozen engine only SIGKILL ends.
``signal_pump_end`` delivers the sentinel without ever blocking: it skips the
enqueue when the pump is already done, and on a full queue drops the oldest
chunk so the sentinel always fits.

Async coroutines are driven with asyncio.run (the repo has no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import queue
import sys
from pathlib import Path

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

from _spawn_contract import pump_data_queue, signal_pump_end  # noqa: E402


class _ClosedWriter:
    """The data peer is gone: every write fails like a reset socket."""

    def write(self, chunk: bytes) -> None:
        raise ConnectionResetError

    async def drain(self) -> None:
        raise ConnectionResetError


class _LiveWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    async def drain(self) -> None:
        return None


def test_full_queue_with_dead_pump_shuts_down_within_bound() -> None:
    # The CA-FLOW-005 shape: pump exits (peer closed), producer refills the
    # queue to the brim. The old blocking put(None) would freeze here forever.
    async def scenario() -> int:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
        q.put_nowait(b"x")
        pump = asyncio.create_task(pump_data_queue(q, _ClosedWriter()))  # type: ignore[arg-type]
        await asyncio.wait_for(pump, timeout=5.0)  # ends on the reset write
        for _ in range(4):
            q.put_nowait(b"chunk")  # producer filled the queue after the pump died
        assert q.full()

        async def _shutdown() -> None:
            assert signal_pump_end(q, pump) is True
            await asyncio.gather(pump, return_exceptions=True)

        await asyncio.wait_for(_shutdown(), timeout=5.0)
        return q.qsize()

    # Pump done -> the enqueue is SKIPPED (nothing will ever consume it).
    assert asyncio.run(scenario()) == 4


def test_live_pump_receives_sentinel_and_ends() -> None:
    async def scenario() -> list[bytes]:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
        q.put_nowait(b"a")
        q.put_nowait(b"b")
        writer = _LiveWriter()
        pump = asyncio.create_task(pump_data_queue(q, writer))  # type: ignore[arg-type]
        assert signal_pump_end(q, pump) is True
        await asyncio.wait_for(pump, timeout=5.0)
        return writer.chunks

    assert asyncio.run(scenario()) == [b"a", b"b"]


def test_full_queue_live_pump_drops_oldest_for_the_sentinel() -> None:
    # Consumer stalled with a full queue: the sentinel outranks stale audio —
    # the oldest chunk is dropped, the pump then drains and ends cleanly.
    async def scenario() -> list[bytes]:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
        q.put_nowait(b"stale")
        q.put_nowait(b"fresh")
        writer = _LiveWriter()
        pump = asyncio.create_task(pump_data_queue(q, writer))  # type: ignore[arg-type]
        # The pump task has not run yet: the sentinel is delivered synchronously
        # into the still-full queue by dropping the oldest chunk.
        assert signal_pump_end(q, pump) is True
        await asyncio.wait_for(pump, timeout=5.0)
        return writer.chunks

    assert asyncio.run(scenario()) == [b"fresh"]


def test_no_pump_task_still_delivers() -> None:
    q: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
    assert signal_pump_end(q) is True
    assert q.get_nowait() is None

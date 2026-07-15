"""SWEEP-1 (#6): the AX.25 TX app must run its burst as a BACKGROUND task so the control command
loop stays responsive (a mid-burst ``stop`` is dequeued and aborts the burst), while STILL failing
the pass with a nonzero exit when the burst raises (the AUDIT ROUND 4 P0 contract, preserved through
the spawn refactor).

These drive the REAL ``amain`` command loop over an in-memory NDJSON control stream, with only the
heavy pieces (spawn-probe, preflight, the burst itself) faked — the run_command_loop <-> burst-task
interplay under test is the real code.
"""

from __future__ import annotations

import asyncio
from typing import Any

import cubesat_gfsk_ax25_tx as txapp
import numpy as np
from _spawn_contract import SpawnSockets


class _CaptureWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed


def _args() -> Any:
    parser = txapp.build_argparser(prog="cubesat_gfsk_ax25_tx", description="t")
    parser.add_argument("--engine", default="", choices=["", "dsp", "gnuradio"])
    return parser.parse_args([])  # all fields default; sdr_args "" => bench/dsp path


def _harness(monkeypatch: Any, emit_impl: Any) -> tuple[asyncio.StreamReader, _CaptureWriter]:
    reader = asyncio.StreamReader()
    status = _CaptureWriter()
    sockets = SpawnSockets(
        control_reader=reader,
        control_writer=_CaptureWriter(),  # type: ignore[arg-type]
        status_writer=status,  # type: ignore[arg-type]
        data_writer=_CaptureWriter(),  # type: ignore[arg-type]
    )

    async def _connect(_args: Any) -> SpawnSockets:
        return sockets

    monkeypatch.setattr(txapp, "connect_spawn_sockets", _connect)
    monkeypatch.setattr(txapp, "_tx_spawn_probe", lambda _a, _p: (True, {}))
    _iq = np.zeros(4, dtype=np.complex64)
    monkeypatch.setattr(txapp, "_preflight_and_build_iq", lambda *_a: _iq)
    monkeypatch.setattr(txapp, "_prepare_tx_cs16", lambda *_a: None)
    monkeypatch.setattr(txapp, "emit_burst", emit_impl)
    return reader, status


def test_amain_stays_responsive_to_stop_during_burst_and_exits_0(monkeypatch: Any) -> None:
    async def _run() -> tuple[int, bool, _CaptureWriter]:
        running = asyncio.Event()
        aborted = {"v": False}

        async def _emit(_status: Any, *_a: Any, should_abort: Any, **_k: Any) -> None:
            running.set()
            while not should_abort():  # a burst that only ends when `stop` sets stop_requested
                await asyncio.sleep(0.001)
            aborted["v"] = True

        reader, status = _harness(monkeypatch, _emit)
        amain_task = asyncio.create_task(txapp.amain(_args()))
        reader.feed_data(b'{"cmd":"start"}\n')
        await asyncio.wait_for(running.wait(), timeout=5.0)  # burst is running...
        reader.feed_data(b'{"cmd":"stop"}\n')  # ...and `stop` is dequeued WHILE it runs
        rc = await asyncio.wait_for(amain_task, timeout=5.0)
        return rc, aborted["v"], status

    rc, aborted, status = asyncio.run(_run())
    assert aborted, "the burst was never aborted — the command loop was blocked by the burst"
    assert rc == 0
    assert b'"stopped"' in bytes(status.buf), "no stopped ack after a clean stop"


def test_amain_burst_failure_exits_nonzero_even_without_a_stop(monkeypatch: Any) -> None:
    """AUDIT ROUND 4 P0 preserved: a burst that RAISES must fail the pass (nonzero exit). The
    done-callback records the failure and feeds control EOF so amain returns 1 even though no `stop`
    ever arrives."""

    async def _run() -> tuple[int, _CaptureWriter]:
        async def _emit_fail(status: Any, *_a: Any, should_abort: Any, **_k: Any) -> None:
            del should_abort
            await txapp.send_event(status, {"event": "error", "code": "tx-failed", "detail": "x"})
            raise RuntimeError("boom")

        reader, status = _harness(monkeypatch, _emit_fail)
        amain_task = asyncio.create_task(txapp.amain(_args()))
        reader.feed_data(b'{"cmd":"start"}\n')  # NO stop is ever sent
        rc = await asyncio.wait_for(amain_task, timeout=5.0)
        return rc, status

    rc, status = asyncio.run(_run())
    assert rc == 1, "a burst failure must fail the pass (nonzero) even without a stop"
    assert b'"tx-failed"' in bytes(status.buf), "the burst error event was not emitted"

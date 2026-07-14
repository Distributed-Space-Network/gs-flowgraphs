"""Finding #17 regression: a wedged-but-open data-socket peer must not stall the
decode loop / block engine teardown.

emit_frame writes a decoded frame body to the data socket and drains it. If the
consumer stays CONNECTED but stops reading (its TCP receive buffer fills), an
unbounded ``data_writer.drain()`` awaits forever — ConnectionReset/BrokenPipe never
fire — so _decode_loop can't observe stop and run_rx's teardown gather never returns,
wedging the process until the supervisor SIGTERMs it. emit_frame now bounds the drain
and treats a stall as a routine peer failure (drop the body, keep going).
"""

from __future__ import annotations

import asyncio

import cubesat_gfsk_endurosat_bidir as bidir


class _OkWriter:
    """Status writer: write + drain both succeed immediately."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf += data

    async def drain(self) -> None:
        return None


class _WedgedWriter:
    """Data writer for a connected-but-not-reading peer: write() succeeds (buffered)
    but drain() never completes and never raises — exactly the hang the fix bounds."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.drain_calls = 0

    def write(self, data: bytes) -> None:
        self.buf += data

    async def drain(self) -> None:
        self.drain_calls += 1
        await asyncio.Event().wait()  # forever — no data ever consumed, no error


class _Sockets:
    def __init__(self, data_writer) -> None:
        self.status_writer = _OkWriter()
        self.data_writer = data_writer


def test_emit_frame_returns_when_data_peer_wedged(monkeypatch):
    # Keep the test fast: shrink the production drain bound to a few ms. The property
    # under test — emit_frame TERMINATES instead of awaiting drain() forever — is
    # unchanged; only the wait shrinks.
    monkeypatch.setattr(bidir, "_DATA_DRAIN_TIMEOUT_S", 0.05)
    wedged = _WedgedWriter()
    sockets = _Sockets(wedged)

    async def _run() -> None:
        # Hard outer timeout: if the fix regresses, emit_frame hangs on drain() and
        # this raises TimeoutError (test FAILS fast) rather than hanging the suite.
        await asyncio.wait_for(bidir.emit_frame(sockets, b"decoded-frame-body"), timeout=5.0)

    asyncio.run(_run())

    # The body was written to the data socket and the drain WAS attempted...
    assert bytes(wedged.buf) == b"decoded-frame-body"
    assert wedged.drain_calls == 1
    # ...and the frame_received status event still went out on the status socket.
    assert b"frame_received" in bytes(sockets.status_writer.buf)


def test_emit_frame_normal_drain_still_works():
    # A healthy peer path is unchanged: body written, drained, status emitted.
    ok = _OkWriter()
    sockets = _Sockets(ok)

    async def _run() -> None:
        await asyncio.wait_for(bidir.emit_frame(sockets, b"hello"), timeout=5.0)

    asyncio.run(_run())
    assert bytes(ok.buf) == b"hello"
    assert b"frame_received" in bytes(sockets.status_writer.buf)

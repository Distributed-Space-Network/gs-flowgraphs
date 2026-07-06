"""Unit tests for _doppler (flowgraph-owned Doppler, docs/12 v2).

No GNU Radio: the module is stdlib-only, so the conversion, source selection, the gs-orbitd
NDJSON poll (against fake asyncio servers), reconnect-on-drop, and the poll loop are all exercised
here. Async bits run via ``asyncio.run`` so no pytest-asyncio config is needed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
from _doppler import (
    NullDopplerSource,
    OrbitdDopplerSource,
    doppler_shift_hz,
    make_doppler_source,
    run_doppler_poll,
)

_C = 299_792_458.0


# ── conversion ────────────────────────────────────────────────────────────────
def test_doppler_shift_sign_and_magnitude() -> None:
    f0 = 401_000_000.0
    # receding (range_rate > 0) -> received LOWER -> negative offset
    assert doppler_shift_hz(f0, 1000.0) == pytest.approx(-f0 * 1000.0 / _C)
    # approaching (range_rate < 0) -> received HIGHER -> positive offset
    assert doppler_shift_hz(f0, -1000.0) == pytest.approx(+f0 * 1000.0 / _C)
    assert doppler_shift_hz(f0, 0.0) == 0.0
    # ~7.5 km/s at 401 MHz is ~10 kHz
    assert doppler_shift_hz(f0, -7500.0) == pytest.approx(10_034.0, abs=5.0)


# ── source selection (pure, no I/O) ───────────────────────────────────────────
def test_make_source_selection() -> None:
    assert isinstance(make_doppler_source(source="none", center_freq_hz=4e8), NullDopplerSource)
    assert isinstance(
        make_doppler_source(source="orbitd", center_freq_hz=4e8, orbitd_handle="p-1"),
        OrbitdDopplerSource,
    )
    # orbitd with nothing available -> Null (record-only)
    assert isinstance(make_doppler_source(source="orbitd", center_freq_hz=4e8), NullDopplerSource)
    # explicit orbitd but no handle -> falls through to Null (can't build a query)
    assert isinstance(make_doppler_source(source="orbitd", center_freq_hz=4e8), NullDopplerSource)


# ── fake servers ──────────────────────────────────────────────────────────────
async def _serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _shutdown(server) -> None:
    # Python 3.12's Server.wait_closed() blocks on lingering client handlers; bound it so a
    # still-reading fake handler can't hang the test (asyncio.run cancels it on loop teardown).
    server.close()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(server.wait_closed(), timeout=1.0)


def _orbitd_handler(range_rate_mps: float):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            req = json.loads(line)
            assert req["op"] == "ephem_at"
            reply = {
                "handle": req["handle"],
                "sample": {
                    "t_unix_s": req.get("t_unix_s", 0.0), "az_deg": 0.0, "el_deg": 0.0,
                    "az_rate_dps": 0.0, "el_rate_dps": 0.0, "range_m": 0.0,
                    "range_rate_mps": range_rate_mps,
                },
                "clamped": False,
            }
            writer.write((json.dumps(reply) + "\n").encode())
            await writer.drain()
    return handle


def test_orbitd_source_polls_and_converts() -> None:
    async def run() -> float | None:
        server, port = await _serve(_orbitd_handler(range_rate_mps=2000.0))
        try:
            src = OrbitdDopplerSource("127.0.0.1", port, "p-1", 401_000_000.0, now_fn=lambda: 123.0)
            off1 = await src.read_offset_hz()
            off2 = await src.read_offset_hz()  # reuses the persistent connection
            await src.aclose()
            assert off1 == off2  # persistent connection, deterministic
            return off1
        finally:
            await _shutdown(server)

    off = asyncio.run(run())
    assert off == pytest.approx(-401_000_000.0 * 2000.0 / _C)


def test_orbitd_source_omits_t_unix_s_by_default() -> None:
    # Production omits t_unix_s so gs-orbitd interpolates at ITS OWN GPS-disciplined clock — the
    # query instant and the plan then share one time frame (no flowgraph host-clock skew).
    async def run() -> dict:
        seen: dict = {}

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            line = await reader.readline()
            seen["req"] = json.loads(line)
            reply = {"handle": "p-1", "sample": {
                "t_unix_s": 0.0, "az_deg": 0.0, "el_deg": 0.0, "az_rate_dps": 0.0,
                "el_rate_dps": 0.0, "range_m": 0.0, "range_rate_mps": 100.0}, "clamped": False}
            writer.write((json.dumps(reply) + "\n").encode())
            await writer.drain()

        server, port = await _serve(handle)
        try:
            src = OrbitdDopplerSource("127.0.0.1", port, "p-1", 4e8)  # NO now_fn -> omit t
            await src.read_offset_hz()
            await src.aclose()
            return seen["req"]
        finally:
            await _shutdown(server)

    req = asyncio.run(run())
    assert req["op"] == "ephem_at" and req["handle"] == "p-1"
    assert "t_unix_s" not in req  # omitted -> daemon uses its disciplined clock


def test_orbitd_source_returns_none_when_unreachable() -> None:
    async def run() -> float | None:
        # nothing listening on this port
        src = OrbitdDopplerSource("127.0.0.1", 1, "p-1", 4e8)
        out = await src.read_offset_hz()
        await src.aclose()
        return out

    assert asyncio.run(run()) is None


def test_orbitd_source_reconnects_after_server_drops() -> None:
    async def run() -> tuple[float | None, float | None]:
        # server that closes the connection after the FIRST reply
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            line = await reader.readline()
            if line:
                reply = {"handle": "p-1", "sample": {
                    "t_unix_s": 0.0, "az_deg": 0.0, "el_deg": 0.0, "az_rate_dps": 0.0,
                    "el_rate_dps": 0.0, "range_m": 0.0, "range_rate_mps": 500.0}, "clamped": False}
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
            writer.close()  # drop after one reply

        server, port = await _serve(handle)
        try:
            src = OrbitdDopplerSource("127.0.0.1", port, "p-1", 4e8, now_fn=lambda: 0.0)
            first = await src.read_offset_hz()   # ok
            second = await src.read_offset_hz()  # server dropped -> None, reconnect
            assert second is None
            third = await src.read_offset_hz()   # fresh connection -> ok again
            await src.aclose()
            return first, third
        finally:
            await _shutdown(server)

    first, third = asyncio.run(run())
    assert first == pytest.approx(-4e8 * 500.0 / _C)
    assert third == pytest.approx(-4e8 * 500.0 / _C)  # recovered by reconnect


# ── the poll loop ─────────────────────────────────────────────────────────────
class _FakeSource:
    def __init__(self, values: list[float | None]) -> None:
        self._values = list(values)
        self._i = 0
        self.closed = False

    async def read_offset_hz(self) -> float | None:
        if self._i < len(self._values):
            v = self._values[self._i]
            self._i += 1
            return v
        return None

    async def aclose(self) -> None:
        self.closed = True


def test_poll_loop_applies_only_past_threshold_and_closes_source() -> None:
    async def run() -> tuple[list[float], bool]:
        applied: list[float] = []
        stop = asyncio.Event()
        # 100 applies (first); 105 within 10 -> skip; 200 applies; None skip; 200 within 0 skip
        src = _FakeSource([100.0, 105.0, 200.0, None, 200.0])
        task = asyncio.create_task(
            run_doppler_poll(src, applied.append, stop, period_s=0.001, resend_threshold_hz=10.0))
        await asyncio.sleep(0.08)
        stop.set()
        await task
        return applied, src.closed

    applied, closed = asyncio.run(run())
    assert applied[:2] == [100.0, 200.0]  # 105 skipped (within threshold), 200 applied
    assert 105.0 not in applied
    assert closed is True  # the loop closes the source on exit


def test_poll_loop_survives_a_raising_apply() -> None:
    async def run() -> list[float]:
        seen: list[float] = []
        stop = asyncio.Event()

        def apply(hz: float) -> None:
            seen.append(hz)
            if len(seen) == 1:
                msg = "rotator glitch"
                raise RuntimeError(msg)  # first apply raises; loop must keep going

        src = _FakeSource([50.0, 500.0])
        task = asyncio.create_task(
            run_doppler_poll(src, apply, stop, period_s=0.001, resend_threshold_hz=10.0))
        await asyncio.sleep(0.05)
        stop.set()
        await task
        return seen

    seen = asyncio.run(run())
    assert 50.0 in seen and 500.0 in seen  # the raise on the first didn't kill the loop


class _SteadySource:
    """Always delivers the same offset (never None) — a poll source that stays healthy."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.closed = False

    async def read_offset_hz(self) -> float | None:
        return self.value

    async def aclose(self) -> None:
        self.closed = True


def test_fallback_ignored_while_source_healthy() -> None:
    # A live source owns Doppler; the fallback (pushed offset) is NEVER applied while reads succeed.
    async def run() -> list[float]:
        applied: list[float] = []
        stop = asyncio.Event()
        src = _SteadySource(100.0)
        task = asyncio.create_task(
            run_doppler_poll(src, applied.append, stop, period_s=0.001, resend_threshold_hz=10.0,
                             fallback_offset=lambda: 9999.0, fallback_grace_s=0.005))
        await asyncio.sleep(0.05)
        stop.set()
        await task
        return applied

    applied = asyncio.run(run())
    assert applied == [100.0]  # only the source value; the 9999 fallback never fires
    assert 9999.0 not in applied


def test_fallback_takes_over_when_source_dies_midpass() -> None:
    # Source resolves (100), then dies (None forever). After the grace, the orchestrator's pushed
    # offset takes over instead of freezing at 100 — the mid-pass-death fix.
    async def run() -> list[float]:
        applied: list[float] = []
        stop = asyncio.Event()
        pushed = {"hz": 2500.0}  # the orchestrator has been pushing since ARM (kept fresh offline)
        src = _FakeSource([100.0])  # one good read, then None forever
        # grace = 2 ticks; generous sleep so the loop actually runs enough ticks despite the
        # coarse (~15 ms) OS timer that stretches a nominal 1 ms period on some platforms.
        task = asyncio.create_task(
            run_doppler_poll(src, applied.append, stop, period_s=0.001, resend_threshold_hz=10.0,
                             fallback_offset=lambda: pushed["hz"], fallback_grace_s=0.002))
        await asyncio.sleep(0.2)
        stop.set()
        await task
        return applied

    applied = asyncio.run(run())
    assert applied[0] == 100.0  # the live source value applied first
    assert 2500.0 in applied  # after the source died past the grace, the pushed offset took over


def test_no_fallback_freezes_last_offset_when_source_dies() -> None:
    # Without a fallback (default), a dead source keeps the last offset — the pre-fix behavior,
    # preserved for callers that pass no fallback_offset (e.g. a record-only / no-push pass).
    async def run() -> list[float]:
        applied: list[float] = []
        stop = asyncio.Event()
        src = _FakeSource([100.0])  # one read, then None forever; no fallback provided
        task = asyncio.create_task(
            run_doppler_poll(src, applied.append, stop, period_s=0.001, resend_threshold_hz=10.0))
        await asyncio.sleep(0.05)
        stop.set()
        await task
        return applied

    applied = asyncio.run(run())
    assert applied == [100.0]  # frozen at the last offset; nothing else applied


def test_poll_loop_skips_nonfinite_offset() -> None:
    # A NaN/inf offset (decayed-TLE range-rate, or a bad pushed value) must NOT reach the rotator —
    # even on the first apply, where the resend-threshold guard short-circuits on last is None.
    async def run() -> list[float]:
        applied: list[float] = []
        stop = asyncio.Event()
        src = _FakeSource([float("nan"), float("inf"), 300.0])
        task = asyncio.create_task(
            run_doppler_poll(src, applied.append, stop, period_s=0.001, resend_threshold_hz=10.0))
        await asyncio.sleep(0.15)
        stop.set()
        await task
        return applied

    applied = asyncio.run(run())
    assert applied == [300.0]  # nan and inf skipped; only the finite value applied

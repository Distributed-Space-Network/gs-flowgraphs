"""Flowgraph-owned Doppler correction (docs/12 — Doppler v2).

The RX flowgraph OWNS its Doppler by POLLING a frequency source at a fixed cadence and applying
the shift to its own software rotator — mirroring how SatNOGS's flowgraph polls rigctld, instead
of the orchestrator PUSHING ``set_doppler`` over the control socket. That push path coupled
Doppler to orchestrator event-loop liveness + control-socket health and broke repeatedly; a
stalled writer silently froze Doppler. Owning the poll makes the correction survive any
orchestrator hiccup (a failed read just keeps the last offset).

Two backends, selected by availability:

* ``orbitd``  — poll gs-orbitd ``ephem_at{handle,t}`` for ``range_rate_mps`` and convert
                ``offset = -f0*v_r/c``. The primary (our SGP4 + GPS-time daemon). Localhost
                NDJSON, so gs-orbitd stays an OPTIONAL, non-GPL dependency (socket IPC, not
                linking — the flowgraph is GPLv3, gs-orbitd is not forced to be).
* ``rigctld`` — poll Hamlib ``rigctld`` ``f`` (get_freq) and convert ``offset = f_rig - f0``,
                the SatNOGS-style fallback for a station that runs rigctld instead of gs-orbitd.
* ``none``    — no Doppler (record-only).

Import-safe: stdlib ``asyncio``/``json`` only, NO GNU Radio, so the conversion + source selection
+ poll loop are unit-tested against fake servers. The one bench-only touch is the ``apply_offset``
callback the caller passes (``ctx.set_doppler`` → the rotator).

License: GPLv3 (see ../COPYING).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

_log = logging.getLogger("gs_flowgraphs._doppler")

_C_MPS = 299_792_458.0  # speed of light

DEFAULT_ORBITD_PORT = 45400
DEFAULT_RIGCTLD_PORT = 4532
DEFAULT_POLL_PERIOD_S = 0.04  # 25 Hz — smooth near TCA; tune per box (OS scheduler tick)
DEFAULT_RESEND_THRESHOLD_HZ = 10.0  # only retune the rotator once the offset moves this far


def doppler_shift_hz(center_freq_hz: float, range_rate_mps: float) -> float:
    """The RX Doppler shift ``f_rx - f0`` of a downlink: ``-f0 * v_r / c``. ``range_rate`` > 0
    (satellite RECEDING) → negative shift (received LOWER); approaching → positive. This matches
    gs-client's ``doppler_shift_hz`` and the flowgraph ``set_doppler(offset_hz)`` convention
    (positive offset = carrier higher = approaching)."""
    return -float(center_freq_hz) * float(range_rate_mps) / _C_MPS


class DopplerSource(Protocol):
    async def read_offset_hz(self) -> float | None: ...
    async def aclose(self) -> None: ...


class NullDopplerSource:
    """No Doppler source — every poll returns None (record-only / no correction)."""

    async def read_offset_hz(self) -> float | None:
        return None

    async def aclose(self) -> None:
        return None


class _LineClient:
    """A persistent line (NDJSON / plain-text) client with reconnect-on-failure. NEVER raises to
    the caller: a dropped or timed-out connection returns None and transparently reconnects on the
    next call — Doppler must never crash the recorder. gs-orbitd's server is persistent (it loops
    reading request lines), so one connection serves the whole pass."""

    def __init__(
        self, host: str, port: int, *, connect_timeout_s: float = 2.0, rpc_timeout_s: float = 1.0,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._connect_timeout_s = connect_timeout_s
        self._rpc_timeout_s = rpc_timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def _ensure(self) -> bool:
        if self._reader is not None:
            return True
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._connect_timeout_s,
            )
            return True
        except (OSError, TimeoutError) as e:
            _log.debug("doppler: connect %s:%s failed (%s)", self._host, self._port, e)
            await self._reset()
            return False

    async def request_line(self, line: str) -> str | None:
        """Send one request line, return one reply line (without the newline), or None on any
        failure (after resetting the connection so the next call reconnects)."""
        if not await self._ensure():
            return None
        assert self._writer is not None and self._reader is not None
        try:
            self._writer.write((line + "\n").encode("utf-8"))
            await asyncio.wait_for(self._writer.drain(), timeout=self._rpc_timeout_s)
            raw = await asyncio.wait_for(self._reader.readline(), timeout=self._rpc_timeout_s)
            if not raw:  # peer closed the connection
                await self._reset()
                return None
            return raw.decode("utf-8", errors="replace").rstrip("\n")
        except (OSError, TimeoutError, asyncio.IncompleteReadError) as e:
            _log.debug("doppler: rpc to %s:%s failed (%s); reconnecting", self._host, self._port, e)
            await self._reset()
            return None

    async def _reset(self) -> None:
        w = self._writer
        self._reader = self._writer = None
        if w is not None:
            with contextlib.suppress(Exception):
                w.close()
                await asyncio.wait_for(w.wait_closed(), timeout=1.0)

    async def aclose(self) -> None:
        await self._reset()


class OrbitdDopplerSource:
    """Poll gs-orbitd ``ephem_at{handle, now}`` → ``range_rate_mps`` → Doppler Hz. The plan
    ``handle`` is the one gs-client materialized at ARM (valid until LOS+margin)."""

    def __init__(
        self, host: str, port: int, handle: str, center_freq_hz: float,
        *, now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._client = _LineClient(host, port)
        self._handle = handle
        self._f0 = float(center_freq_hz)
        self._now_fn = now_fn

    async def read_offset_hz(self) -> float | None:
        req = json.dumps({"op": "ephem_at", "handle": self._handle, "t_unix_s": self._now_fn()})
        reply = await self._client.request_line(req)
        if reply is None:
            return None
        try:
            obj: Any = json.loads(reply)
            rr = float(obj["sample"]["range_rate_mps"])
        except (ValueError, KeyError, TypeError):
            _log.debug("doppler: bad ephem_at reply %r", reply[:120])
            return None
        return doppler_shift_hz(self._f0, rr)

    async def aclose(self) -> None:
        await self._client.aclose()


class RigctldDopplerSource:
    """Poll Hamlib ``rigctld`` ``f`` (get_freq) → ``offset = f_rig - f0`` (SatNOGS-style). A
    separate tracker sets rigctld's frequency to the Doppler-shifted downlink; we read it and
    shift the DSP to bring that carrier back to baseband centre, keeping the SDR fixed."""

    def __init__(self, host: str, port: int, center_freq_hz: float) -> None:
        self._client = _LineClient(host, port)
        self._f0 = float(center_freq_hz)

    async def read_offset_hz(self) -> float | None:
        reply = await self._client.request_line("f")  # rigctld: get_freq → a bare Hz line
        if reply is None:
            return None
        try:
            f_rig = float(reply.strip())
        except ValueError:
            _log.debug("doppler: bad rigctld freq reply %r", reply[:120])
            return None
        if f_rig <= 0.0:  # rigctld returns 0 / RPRT error before it's been set
            return None
        return f_rig - self._f0

    async def aclose(self) -> None:
        await self._client.aclose()


def make_doppler_source(
    *, source: str, center_freq_hz: float,
    orbitd_host: str = "127.0.0.1", orbitd_port: int = DEFAULT_ORBITD_PORT, orbitd_handle: str = "",
    rigctl_host: str = "", rigctl_port: int = DEFAULT_RIGCTLD_PORT,
    now_fn: Callable[[], float] = time.time,
) -> DopplerSource:
    """Pick the Doppler source (pure — no I/O; a source that can't actually connect just returns
    None each poll). ``source``: ``auto`` prefers orbitd (needs a handle) then rigctld (needs a
    host); ``orbitd`` / ``rigctld`` force one; ``none`` → :class:`NullDopplerSource`."""
    s = (source or "auto").strip().lower()
    if s == "none":
        return NullDopplerSource()
    if s in ("orbitd", "auto") and orbitd_handle:
        _log.info("doppler: source=orbitd handle=%s @ %s:%s f0=%.0f",
                  orbitd_handle, orbitd_host, orbitd_port, center_freq_hz)
        return OrbitdDopplerSource(orbitd_host, orbitd_port, orbitd_handle, center_freq_hz,
                                   now_fn=now_fn)
    if s in ("rigctld", "rigctl", "auto") and rigctl_host:
        _log.info("doppler: source=rigctld @ %s:%s f0=%.0f",
                  rigctl_host, rigctl_port, center_freq_hz)
        return RigctldDopplerSource(rigctl_host, rigctl_port, center_freq_hz)
    _log.info("doppler: no source (source=%r, orbitd_handle=%r, rigctl_host=%r); record-only",
              s, orbitd_handle, rigctl_host)
    return NullDopplerSource()


async def run_doppler_poll(
    source: DopplerSource, apply_offset: Callable[[float], None], stop: asyncio.Event,
    *, period_s: float = DEFAULT_POLL_PERIOD_S,
    resend_threshold_hz: float = DEFAULT_RESEND_THRESHOLD_HZ,
) -> None:
    """Poll ``source`` every ``period_s`` and call ``apply_offset(hz)`` (→ the rotator) whenever
    the offset moves past ``resend_threshold_hz``. Runs until ``stop`` is set. NEVER propagates a
    source error — a None read just skips the tick and keeps the last offset, so a Doppler-source
    outage can never wedge the recorder. The flowgraph OWNS this loop (SatNOGS-style)."""
    last: float | None = None
    try:
        while not stop.is_set():
            offset = await source.read_offset_hz()
            if offset is not None and (last is None or abs(offset - last) >= resend_threshold_hz):
                try:
                    apply_offset(offset)
                    last = offset
                except Exception:  # noqa: BLE001 — a rotator glitch must not kill the poll/recorder
                    _log.exception("doppler: apply_offset(%.1f) raised", offset)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=period_s)
    finally:
        await source.aclose()

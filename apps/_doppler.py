"""Flowgraph-owned Doppler correction (docs/12 — Doppler v2).

The RX flowgraph OWNS its Doppler by POLLING gs-orbitd at a fixed cadence and applying the shift
to its own software rotator — mirroring how SatNOGS's flowgraph polls a frequency source, instead
of the orchestrator PUSHING ``set_doppler`` over the control socket. That push path coupled
Doppler to orchestrator event-loop liveness + control-socket health and broke repeatedly; a
stalled writer silently froze Doppler. Owning the poll makes the correction survive any
orchestrator hiccup (a failed read just keeps the last offset).

Source:

* ``orbitd`` — poll gs-orbitd ``ephem_at{handle,t}`` for ``range_rate_mps`` and convert
               ``offset = -f0*v_r/c``. gs-orbitd is our SGP4 + GPS-time daemon and the single
               source of truth for orbit + time, so pointing and Doppler share one ephemeris and
               cannot diverge. Localhost NDJSON, so gs-orbitd stays an OPTIONAL, non-GPL dependency
               (socket IPC, not linking — the flowgraph is GPLv3, gs-orbitd is not forced to be).
* ``none``   — no Doppler (record-only).

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
import math
from collections.abc import Callable
from typing import Any, Protocol

_log = logging.getLogger("gs_flowgraphs._doppler")

_C_MPS = 299_792_458.0  # speed of light

DEFAULT_ORBITD_PORT = 45400
DEFAULT_POLL_PERIOD_S = 0.04  # 25 Hz — smooth near TCA; tune per box (OS scheduler tick)
DEFAULT_RESEND_THRESHOLD_HZ = 10.0  # only retune the rotator once the offset moves this far
DEFAULT_FALLBACK_GRACE_S = 1.0  # source dead this long → the pushed offset takes over


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
        except (OSError, asyncio.TimeoutError) as e:  # noqa: UP041 — asyncio.TimeoutError is a
            # DISTINCT class from builtin TimeoutError on Python 3.10 (our declared floor); they
            # were unified only in 3.11. asyncio.wait_for raises the asyncio class, so we MUST name
            # it here or a connect timeout escapes uncaught on 3.10.
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
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError,  # noqa: UP041 (3.10)
                asyncio.LimitOverrunError, ValueError) as e:
            # asyncio.TimeoutError: distinct from builtin TimeoutError on 3.10 (see _ensure).
            # LimitOverrunError/ValueError: a too-long or undecodable reply line must reconnect,
            # not propagate — _LineClient must NEVER raise to the poll loop.
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
    """Poll gs-orbitd ``ephem_at{handle}`` → ``range_rate_mps`` → Doppler Hz. The plan ``handle``
    is the one gs-client materialized at ARM (valid until LOS+margin).

    We OMIT ``t_unix_s`` so gs-orbitd interpolates at ITS OWN GPS-disciplined clock — the plan and
    the query instant then share one time frame by construction. Indexing with the flowgraph's raw
    host clock (a possibly-skewed ``time.time()``) would query Doppler in the wrong frame; near TCA
    a second of skew is hundreds of Hz. ``now_fn`` stays as an OPTIONAL override for tests/replay
    (when set, its value is sent as ``t_unix_s``); production leaves it ``None`` (omit)."""

    def __init__(
        self, host: str, port: int, handle: str, center_freq_hz: float,
        *, now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._client = _LineClient(host, port)
        self._host = host
        self._port = int(port)
        self._handle = handle
        self._f0 = float(center_freq_hz)
        self._now_fn = now_fn
        self._failed = False  # so a persistent failure is logged LOUD once, not silently at DEBUG

    def _note_failure(self, msg: str) -> None:
        # First failure → WARNING (visible in the INFO journal, so a dead Doppler source is never
        # invisible again); repeats → DEBUG to avoid spamming 25×/s.
        if self._failed:
            _log.debug("doppler: %s", msg)
        else:
            self._failed = True
            _log.warning("doppler: %s", msg)

    async def read_offset_hz(self) -> float | None:
        req: dict[str, Any] = {"op": "ephem_at", "handle": self._handle}
        if self._now_fn is not None:  # test/replay override; production omits t -> daemon's clock
            req["t_unix_s"] = self._now_fn()
        reply = await self._client.request_line(json.dumps(req))
        if reply is None:
            self._note_failure(
                f"no reply from gs-orbitd at {self._host}:{self._port} (op=ephem_at "
                f"handle={self._handle}) — is the daemon running? Doppler is NOT being applied.")
            return None
        try:
            obj: Any = json.loads(reply)
            rr = float(obj["sample"]["range_rate_mps"])
        except (ValueError, KeyError, TypeError):
            self._note_failure(
                f"gs-orbitd ephem_at returned no usable sample (reply={reply[:200]!r}). A "
                f"gs-orbitd older than the ephem_at op (Doppler v2) answers this with an error "
                f"— REDEPLOY gs-orbitd. Doppler is NOT being applied.")
            return None
        if self._failed:  # recovered after a warning — say so
            self._failed = False
            _log.info("doppler: ephem_at recovered @ %s:%s — Doppler applying again",
                      self._host, self._port)
        return doppler_shift_hz(self._f0, rr)

    async def aclose(self) -> None:
        await self._client.aclose()


def make_doppler_source(
    *, source: str, center_freq_hz: float,
    orbitd_host: str = "127.0.0.1", orbitd_port: int = DEFAULT_ORBITD_PORT, orbitd_handle: str = "",
    now_fn: Callable[[], float] | None = None,
) -> DopplerSource:
    """Pick the Doppler source (pure — no I/O; a source that can't actually connect just returns
    None each poll). ``orbitd`` (the default) polls gs-orbitd and needs a plan ``handle``; ``none``
    — or ``orbitd`` with no handle — → :class:`NullDopplerSource` (record-only)."""
    s = (source or "orbitd").strip().lower()
    if s in ("orbitd", "auto") and orbitd_handle:
        _log.info("doppler: source=orbitd handle=%s @ %s:%s f0=%.0f",
                  orbitd_handle, orbitd_host, orbitd_port, center_freq_hz)
        return OrbitdDopplerSource(orbitd_host, orbitd_port, orbitd_handle, center_freq_hz,
                                   now_fn=now_fn)
    _log.info("doppler: no source (source=%r, orbitd_handle=%r); record-only", s, orbitd_handle)
    return NullDopplerSource()


async def run_doppler_poll(
    source: DopplerSource, apply_offset: Callable[[float], None], stop: asyncio.Event,
    *, period_s: float = DEFAULT_POLL_PERIOD_S,
    resend_threshold_hz: float = DEFAULT_RESEND_THRESHOLD_HZ,
    fallback_offset: Callable[[], float | None] | None = None,
    fallback_grace_s: float = DEFAULT_FALLBACK_GRACE_S,
) -> None:
    """Poll ``source`` every ``period_s`` and call ``apply_offset(hz)`` (→ the rotator) whenever
    the offset moves past ``resend_threshold_hz``. Runs until ``stop`` is set. NEVER propagates a
    source error — a None read just skips the tick, so a Doppler-source outage can never wedge the
    recorder. The flowgraph OWNS this loop (SatNOGS-style).

    ``fallback_offset`` guards the failure mode where the source RESOLVES at spawn (so the caller
    latched into poll mode and gated off its own legacy push) then DIES mid-pass: instead of
    freezing at a stale offset for the rest of the window, once the source has gone
    ``fallback_grace_s`` of WALL-CLOCK time without a live read we apply ``fallback_offset()`` — the
    orchestrator's control-socket push, computed offline from the ARM-materialized ephemeris and
    valid even if gs-orbitd is gone. The grace is a monotonic-clock deadline, NOT a poll count, so a
    slow/hung read (which can block up to the source's connect/rpc timeout) can't stretch the
    hand-off. A single reader applies here, so poll and fallback never fight; a good source read
    resets the grace and reclaims ownership. ``None`` ⇒ no fallback (freeze, as before). Non-finite
    offsets (a NaN/inf from a decayed-TLE range-rate or a bad pushed value) are never applied."""
    last: float | None = None
    on_fallback = False  # so the poll→push handoff is logged once (visible in the INFO journal)
    last_log = 0.0  # throttle the "applying" INFO so the journal SHOWS the Doppler curve
    loop = asyncio.get_running_loop()
    last_ok = loop.time()  # monotonic time of the last live read — the wall-clock grace anchor
    try:
        while not stop.is_set():
            try:
                offset = await source.read_offset_hz()
            except Exception:  # noqa: BLE001 — contract: NEVER propagate a source error; a bug/edge
                # in a source must not kill the poll task and freeze Doppler for the whole pass.
                _log.exception("doppler: read_offset_hz raised; keeping last offset")
                offset = None
            if offset is not None:
                last_ok = loop.time()  # a live source read reclaims Doppler ownership
                if on_fallback:
                    on_fallback = False
                    _log.info("doppler: poll source recovered — poll owns Doppler again")
            elif fallback_offset is not None and loop.time() - last_ok >= fallback_grace_s:
                # source dead past the (wall-clock) grace → the pushed offset takes over.
                if not on_fallback:
                    on_fallback = True
                    _log.warning(
                        "doppler: poll source silent for >%.1fs — driving Doppler from the "
                        "orchestrator push instead (check gs-orbitd ephem_at)", fallback_grace_s)
                try:
                    offset = fallback_offset()
                except Exception:  # noqa: BLE001 — a bad fallback must not kill the poll/recorder
                    _log.exception("doppler: fallback_offset() raised")
                    offset = None
            if (offset is not None and math.isfinite(offset)
                    and (last is None or abs(offset - last) >= resend_threshold_hz)):
                try:
                    apply_offset(offset)
                    last = offset
                    now = loop.time()  # log the applied offset ~every 5 s: a visible tracking curve
                    if now - last_log >= 5.0:
                        last_log = now
                        _log.info("doppler: applying %.0f Hz to the rotator", offset)
                except Exception:  # noqa: BLE001 — a rotator glitch must not kill the poll/recorder
                    _log.exception("doppler: apply_offset(%.1f) raised", offset)
            # asyncio.TimeoutError (NOT builtin TimeoutError) is what wait_for raises on 3.10 — this
            # fires EVERY tick on the normal path, so getting the class wrong killed the whole poll.
            with contextlib.suppress(asyncio.TimeoutError):  # noqa: UP041 — 3.10 distinct class
                await asyncio.wait_for(stop.wait(), timeout=period_s)
    finally:
        await source.aclose()

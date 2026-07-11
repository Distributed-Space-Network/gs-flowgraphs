"""Shared production Soapy TX transport (F-03: P0-07, feeds R-16).

Ported from the PROVEN bench tool ``tools/tx_gfsk.py`` (``_write_burst``) and
promoted to the one transport every production TX path uses:

* stream-MTU query + chunking — writing more than the driver's packet size
  can segfault native drivers (XTRX/LMS);
* bounded zero/timeout/error handling — a stalled or erroring stream ends the
  burst with an explicit outcome instead of spinning forever;
* ``END_BURST`` on the LAST DATA chunk — a separate 0-length END_BURST write
  BLOCKS on XTRX/LMS drivers (never use one);
* a total per-burst deadline and cooperative cancellation — an async stop can
  end a blocked write loop;
* an ``on_first_accept`` hook so callers emit ``transmit_started`` only when
  the stream provably takes samples (R-16), never on command receipt.

Import-safe without SoapySDR: the constants are imported lazily inside
``write_burst`` so dev/CI hosts can unit-test with a fake device.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Consecutive no-progress writes (timeout OR zero-sample accepts) before the
# burst is declared stalled. Mirrors tools/tx_gfsk.py's bound.
DEFAULT_MAX_STALLS = 20
DEFAULT_WRITE_TIMEOUT_US = 1_000_000
FALLBACK_MTU = 4096


@dataclass(frozen=True)
class BurstResult:
    """Explicit, bounded outcome of one TX burst (R-16: completion must report
    accepted samples and a real outcome, never a fabricated success)."""

    accepted: int
    total: int
    outcome: str  # "complete" | "stalled" | "error" | "deadline" | "cancelled"
    detail: str = ""

    @property
    def complete(self) -> bool:
        return self.outcome == "complete" and self.accepted >= self.total


def query_tx_mtu(dev: object, stream: object, *, fallback: int = FALLBACK_MTU) -> int:
    """The driver's per-writeStream element budget; ``fallback`` when the call
    is unsupported/nonsensical. Every chunk MUST stay within it (P0-07)."""
    try:
        mtu = int(dev.getStreamMTU(stream))  # type: ignore[attr-defined]
    except Exception:
        log.info("tx: getStreamMTU unavailable; using fallback %d", fallback)
        return fallback
    if mtu <= 0:
        log.warning("tx: driver reported MTU %d; using fallback %d", mtu, fallback)
        return fallback
    return mtu


def write_burst(
    dev: object,
    stream: object,
    buf: object,
    *,
    mtu: int,
    timeout_us: int = DEFAULT_WRITE_TIMEOUT_US,
    deadline_s: float | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_first_accept: Callable[[], None] | None = None,
    max_stalls: int = DEFAULT_MAX_STALLS,
    copy_chunks: bool = False,
) -> BurstResult:
    """Write one IQ buffer as one burst, bounded on every axis.

    ``buf`` is a numpy complex64 array (or anything sliceable with ``len``).
    Returns a :class:`BurstResult`; NEVER raises for stream-level conditions —
    the caller decides what an incomplete burst means (P0-07: the transport
    itself can no longer spin, stall unboundedly, or oversize a write).
    """
    from SoapySDR import SOAPY_SDR_END_BURST, SOAPY_SDR_TIMEOUT  # noqa: PLC0415

    n = len(buf)  # type: ignore[arg-type]
    if n == 0:
        return BurstResult(accepted=0, total=0, outcome="complete")
    chunk = max(1, int(mtu))
    i = 0
    stalls = 0
    accepted_any = False
    call_shape_full = True  # try the 6-arg call first; fall back per binding
    t0 = time.monotonic()
    while i < n:
        if should_abort is not None and should_abort():
            return BurstResult(i, n, "cancelled", "aborted by caller")
        if deadline_s is not None and (time.monotonic() - t0) > deadline_s:
            return BurstResult(i, n, "deadline", f"total burst deadline {deadline_s:.1f}s")
        num = min(chunk, n - i)
        block = buf[i : i + num]  # type: ignore[index]
        if copy_chunks:
            block = block.copy()  # type: ignore[union-attr]
        flags = SOAPY_SDR_END_BURST if (i + num) >= n else 0
        if call_shape_full:
            try:
                sr = dev.writeStream(  # type: ignore[attr-defined]
                    stream, [block], num, flags, 0, timeout_us
                )
            except TypeError:
                # Some bindings only take (stream, buffs, numElems, flags).
                call_shape_full = False
                sr = dev.writeStream(stream, [block], num, flags)  # type: ignore[attr-defined]
        else:
            sr = dev.writeStream(stream, [block], num, flags)  # type: ignore[attr-defined]
        ret = int(sr.ret)
        if ret > 0:
            i += ret
            stalls = 0
            if not accepted_any:
                accepted_any = True
                log.info("tx: streaming (first %d samples accepted)", ret)
                if on_first_accept is not None:
                    try:
                        on_first_accept()
                    except Exception:
                        log.exception("tx: on_first_accept callback raised")
        elif ret == SOAPY_SDR_TIMEOUT or ret == 0:
            # ret==0 is "accepted nothing" — pre-fix loops treated it as
            # progressless success and spun forever (P0-07).
            stalls += 1
            if stalls > max_stalls:
                return BurstResult(
                    i, n, "stalled", f"no progress after {stalls} writes"
                )
        else:
            return BurstResult(i, n, "error", f"writeStream ret={ret}")
    return BurstResult(i, n, "complete")

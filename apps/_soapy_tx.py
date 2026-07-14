"""Shared production Soapy TX transport (F-03: P0-07, feeds R-16).

The one transport every production TX path uses. Its send behaviour is the
KNOWN-GOOD shape distilled from the XTRX bench probe
(``tools/probe_soapy_tx_write.py``) — the simplest call that reliably delivers a
buffer — with nothing the probe used only for diagnosis (case matrix, alternate
call signatures, per-write tracing):

* stream-MTU query + chunking at ``min(1024, MTU)`` — a REQUIRED positive MTU,
  never a guessed fallback: writing more than the driver's packet size (or a
  fabricated size) can segfault native drivers (XTRX/LMS);
* ONE call shape only — ``writeStream(stream, [chunk], num_elems)``, three
  arguments, no flags / timestamp / timeout overload and no ``END_BURST`` (a
  separate 0-length END_BURST write BLOCKS on XTRX/LMS drivers);
* ``num_elems`` counts COMPLEX samples; a flat CS16 buffer [I0,Q0,I1,Q1,...]
  is sliced ``buf[2*i : 2*(i+n)]`` (two int16 per complex sample), a complex64
  buffer one element per sample;
* bounded outcomes — advance only by the positive accepted count, reject a
  return greater than requested, bound repeated zero (no-progress) returns, and
  treat any negative Soapy result as an error;
* a total per-burst deadline and cooperative cancellation — an async stop ends
  the write loop and never writes after authority is revoked;
* an ``on_first_accept`` hook so callers emit ``transmit_started`` only when
  the stream provably takes a sample (R-16), never on command receipt;
* ONE bounded ``readStreamStatus`` outcome check (finding #22) so a burst the
  driver ACCEPTED then DISCARDED late is not reported as a clean transmission.

Import-safe without SoapySDR: ``write_burst``/``query_tx_mtu`` touch no SoapySDR
symbols at all, and ``poll_tx_status`` imports them lazily, so dev/CI hosts can
unit-test the whole path with a fake device.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Consecutive zero-sample writes (accepted nothing) before the burst is declared
# stalled. A NEGATIVE Soapy return is an error, not a stall (see write_burst);
# only ret==0 counts here.
DEFAULT_MAX_STALLS = 20

# Largest complex-sample chunk offered to one writeStream, capped further by the
# driver MTU. The bench-proven probe writes at min(1024, MTU); a bigger chunk
# buys nothing and risks oversizing a native driver.
CHUNK_COMPLEX_SAMPLES = 1024

# Post-burst readStreamStatus poll bounds (finding #22). writeStream returning
# ret>0 means the driver ACCEPTED the buffer into its DMA queue, NOT that valid
# RF left the antenna: the XTRX bench probe (tools/probe_soapy_tx_write.py)
# documents libxtrx accepting a buffer and then DISCARDING it late ("TX DMA ...
# skip due to TO buffers" / "delayed buffers"), surfaced via readStreamStatus as
# UNDERFLOW / TIME_ERROR / an END_ABRUPT flag. These bound that async check so it
# can NEVER hang the keyed window: at most _MAX_STATUS_POLLS calls, each capped at
# STATUS_POLL_TIMEOUT_US, and the whole poll capped at STATUS_POLL_DEADLINE_S.
STATUS_POLL_TIMEOUT_US = 100_000
STATUS_POLL_DEADLINE_S = 0.5
_MAX_STATUS_POLLS = 8


@dataclass(frozen=True)
class BurstResult:
    """Explicit, bounded outcome of one TX burst (R-16: completion must report
    accepted samples and a real outcome, never a fabricated success)."""

    accepted: int
    total: int
    # "discarded" = writeStream accepted every sample but the driver's own
    # readStreamStatus reported a late/underflow/abrupt discard afterwards, so
    # nothing valid (or only garbage) actually radiated — a dead uplink that must
    # NOT be reported as a good one (finding #22).
    outcome: str  # "complete" | "stalled" | "error" | "deadline" | "cancelled" | "discarded"
    detail: str = ""

    @property
    def complete(self) -> bool:
        return self.outcome == "complete" and self.accepted >= self.total


def query_tx_mtu(dev: object, stream: object) -> int:
    """The driver's per-writeStream element budget — REQUIRED to be a positive
    integer. A missing or non-positive MTU is a hard error: we never substitute a
    guessed fallback, because an oversized write can segfault a native driver
    (P0-07) and a fabricated MTU is exactly such a write. Every chunk must stay
    within this value; the caller fails closed (no burst) on a bad MTU."""
    mtu = int(dev.getStreamMTU(stream))  # type: ignore[attr-defined]  # raises -> fail closed
    if mtu <= 0:
        raise ValueError(f"driver reported non-positive stream MTU {mtu}")
    return mtu


def to_cs16(iq: object) -> np.ndarray:
    """Pack a complex baseband array into ONE contiguous flat CS16 buffer
    ``[I0,Q0,I1,Q1,...]`` int16 — the exact layout ``write_burst`` streams and the
    XTRX bench probe (``tools/probe_soapy_tx_write.py``) proved on hardware.

    Full-scale complex (``|value| <= 1.0``) maps to the int16 range; real/imag are
    clipped to ``[-1, 1]`` FIRST so an out-of-range sample cannot wrap to the
    opposite rail. This is the pre-key conversion both TX sinks do before keying,
    so nothing but a flat CS16 stream reaches the driver (P0-07 write shape)."""
    a = np.ascontiguousarray(np.asarray(iq, dtype=np.complex64))
    out = np.empty(a.size * 2, dtype=np.int16)
    out[0::2] = np.rint(np.clip(a.real, -1.0, 1.0) * 32767.0).astype(np.int16)
    out[1::2] = np.rint(np.clip(a.imag, -1.0, 1.0) * 32767.0).astype(np.int16)
    return out


class TxGainConfigError(ValueError):
    """No named per-element TX gain (e.g. PAD) is configured for a TX sink. The
    overall ``setGain`` overload is XTRX-unsafe (it aborts SoapyXTRX), so it is
    never used as a fallback — a TX with no named drive is a CONFIGURATION error,
    refused rather than transmitted deaf or keyed through the unsafe overload."""


def named_tx_gains(tx_settings: object) -> dict[str, float]:
    """The usable named per-element TX gains (e.g. ``{"PAD": 52.0}``) from resolved
    TX settings. REQUIRED: raises :class:`TxGainConfigError` when none is present.

    TX drive MUST be an explicit named gain — the overall ``setGain`` overload is
    never applied (it aborts SoapyXTRX), and a deaf TX (0 dB) radiates nothing, so
    "no named gain" is a hard configuration error, not something to paper over."""
    gains = tx_settings.get("sdr_gains") if isinstance(tx_settings, dict) else None
    named = {
        k: float(v)
        for k, v in (gains or {}).items()
        if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    # RE-AUDIT (P2): require PAD SPECIFICALLY, not merely "some named gain". PAD is the element that
    # sets the XTRX TX OUTPUT drive; any OTHER named gain alone (e.g. IAMP, a digital preamp) leaves
    # the output PAD at its default and radiates the wrong level (often deaf). The overall setGain
    # overload is XTRX-unsafe, so it is never a fallback — a TX without PAD is a config error.
    if not any(k.upper() == "PAD" for k in named):
        raise TxGainConfigError(
            "no PAD TX gain configured — refusing: PAD is the REQUIRED named TX drive element "
            f"(got {sorted(named) or 'none'}); the overall setGain overload is XTRX-unsafe and any "
            "other named gain alone does not set the TX output. Configure sdr_tx_gains / "
            "GS_SDR_TX_GAINS with PAD."
        )
    return named


def poll_tx_status(
    dev: object,
    stream: object,
    *,
    timeout_us: int = STATUS_POLL_TIMEOUT_US,
    deadline_s: float = STATUS_POLL_DEADLINE_S,
) -> tuple[bool, str]:
    """Drain ``readStreamStatus`` after a burst to catch a late/underflow/discard the
    driver reports ASYNCHRONOUSLY (finding #22).

    ``writeStream`` returning ret>0 only means the buffer entered the driver's DMA
    queue. An XTRX that then discards it late ("skip due to TO buffers") signals that
    through a stream-status event, never through the write return. This reads those
    events, bounded on every axis so it can't hang the keyed window.

    Returns ``(bad, detail)``:

    * ``(True, reason)`` on a DEFINITIVE underflow / time-error / stream-error /
      corruption / END_ABRUPT discard — the burst did not radiate cleanly.
    * ``(False, "")`` when the status is clean, the driver does not support status
      reads, the status read itself errors, or nothing definitive arrived within the
      bounded deadline. We NEVER fabricate a discard we cannot prove — a false failure
      would abort a good pass; the concrete, driver-reported discard is what we act on.
    """
    read = getattr(dev, "readStreamStatus", None)
    if not callable(read):
        return (False, "")
    import SoapySDR as _sd  # noqa: PLC0415

    timeout = int(getattr(_sd, "SOAPY_SDR_TIMEOUT", -1))
    not_supported = int(getattr(_sd, "SOAPY_SDR_NOT_SUPPORTED", -5))
    end_abrupt = int(getattr(_sd, "SOAPY_SDR_END_ABRUPT", 8))
    bad_ret = {
        int(getattr(_sd, "SOAPY_SDR_UNDERFLOW", -7)): "underflow (samples discarded as late)",
        int(getattr(_sd, "SOAPY_SDR_TIME_ERROR", -6)): "time-error (late buffers discarded)",
        int(getattr(_sd, "SOAPY_SDR_STREAM_ERROR", -2)): "stream-error",
        int(getattr(_sd, "SOAPY_SDR_CORRUPTION", -3)): "corruption",
    }
    t0 = time.monotonic()
    polls = 0
    while (time.monotonic() - t0) <= deadline_s and polls < _MAX_STATUS_POLLS:
        polls += 1
        try:
            try:
                sr = read(stream, timeoutUs=timeout_us)
            except TypeError:
                sr = read(stream, timeout_us)
        except Exception:
            # Can't read status → cannot prove a discard; do not fabricate one.
            log.debug("tx: readStreamStatus unavailable/raised; status not checked", exc_info=True)
            return (False, "")
        ret = int(getattr(sr, "ret", 0) or 0)
        flags = int(getattr(sr, "flags", 0) or 0)
        if ret in bad_ret:
            return (True, f"readStreamStatus ret={ret} ({bad_ret[ret]})")
        if flags & end_abrupt:
            return (True, f"readStreamStatus flags={flags} (END_ABRUPT — burst discarded)")
        if ret in (timeout, not_supported):
            # No (further) status pending, or the driver has no status queue: clean.
            return (False, "")
        if ret < 0:
            # An UNEXPECTED negative status (not TIMEOUT/NOT_SUPPORTED, not one of the
            # named discards above) is treated as a discard, not silently swallowed: an
            # unknown negative from the driver is a failure we cannot prove benign.
            return (True, f"readStreamStatus unexpected negative ret={ret}")
        # ret==0 (a benign status event): keep draining, bounded by the deadline/poll
        # cap, in case a discard event is queued behind it.
    return (False, "")


def _finish(accepted: int, total: int, outcome: str, detail: str = "") -> BurstResult:
    """Build the burst result and log exactly ONE terminal line for it — a final
    info line for a clean burst, or a single error/warning line otherwise. No
    per-chunk or per-phase logging (that was diagnostic-probe behaviour)."""
    result = BurstResult(accepted=accepted, total=total, outcome=outcome, detail=detail)
    if result.complete:
        log.info("tx: burst complete — %d/%d samples accepted", accepted, total)
    else:
        suffix = f" ({detail})" if detail else ""
        emit = log.error if outcome in ("error", "discarded") else log.warning
        emit("tx: burst %s — %d/%d accepted%s", outcome, accepted, total, suffix)
    return result


def write_burst(
    dev: object,
    stream: object,
    buf: object,
    *,
    mtu: int,
    deadline_s: float | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_first_accept: Callable[[], None] | None = None,
    max_stalls: int = DEFAULT_MAX_STALLS,
    poll_status: bool = True,
) -> BurstResult:
    """Write one already-built waveform as one burst, bounded on every axis.

    ``buf`` is the FINAL hardware-rate waveform. A flat CS16 int16 buffer
    ([I0,Q0,I1,Q1,...]) is the bench-proven layout — ``num_elems`` counts COMPLEX
    samples, so complex offset ``i`` / count ``n`` is ``buf[2*i : 2*(i+n)]``; a
    complex64 buffer (the CF32 sink path) carries one element per complex sample.

    Returns a :class:`BurstResult`; NEVER raises for stream-level conditions — the
    caller decides what an incomplete burst means, and (P0-07) the transport can no
    longer spin, stall unboundedly, oversize a write, or add flags/timeouts/an
    END_BURST the XTRX/LMS drivers block on.

    WRITE-TIMEOUT HONESTY (3h): the 3-arg ``writeStream`` carries NO timeout argument,
    and even one could not interrupt a HUNG native ``writeStream`` — a driver that never
    returns hangs this thread regardless. The per-burst ``deadline_s`` is checked at the
    TOP of the loop, BETWEEN writes, so it bounds the total time across writes and stops
    a slow/stalling burst; it likewise cannot unblock a single call that is already stuck
    inside the driver. The real backstop for a genuinely hung driver call is the
    orchestrator's keyed-window timeout (gs-client forcing the PA off), not anything here.
    """
    # int16 flat CS16 → two elements per complex sample; anything else (complex64)
    # is one element per complex sample. This is the only place the layout matters.
    dtype = getattr(buf, "dtype", None)
    elems_per = 2 if (dtype is not None and getattr(dtype, "kind", "") == "i") else 1
    n = len(buf) // elems_per  # type: ignore[arg-type]  # COMPLEX-sample count
    if n == 0:
        # (3g) an empty burst is an ERROR, never a silent "complete" success — a caller
        # that reached the write path with nothing to send has already failed.
        return _finish(0, 0, "error", "empty buffer — nothing to transmit")

    chunk = min(CHUNK_COMPLEX_SAMPLES, int(mtu))
    if chunk <= 0:
        return _finish(0, n, "error", f"non-positive stream MTU {mtu!r}")

    i = 0
    stalls = 0
    accepted_any = False
    t0 = time.monotonic()
    while i < n:
        if should_abort is not None and should_abort():
            return _finish(i, n, "cancelled", "aborted by caller")
        if deadline_s is not None and (time.monotonic() - t0) > deadline_s:
            return _finish(i, n, "deadline", f"total burst deadline {deadline_s:.1f}s")
        num = min(chunk, n - i)
        block = buf[elems_per * i : elems_per * (i + num)]  # type: ignore[index]
        # The ONE known-good call shape: three arguments, no flags/timestamp/timeout
        # overload and no END_BURST. num counts COMPLEX samples regardless of layout.
        sr = dev.writeStream(stream, [block], num)  # type: ignore[attr-defined]
        ret = int(sr.ret)
        if ret > num:
            # A driver cannot accept more than it was offered; a return past the
            # request is corrupt, not progress — refuse it rather than over-advance.
            return _finish(i, n, "error", f"writeStream ret={ret} exceeds requested {num}")
        if ret > 0:
            i += ret
            stalls = 0
            if not accepted_any:
                accepted_any = True
                log.info("tx: transmit start — first %d samples accepted", ret)
                if on_first_accept is not None:
                    try:
                        on_first_accept()
                    except Exception:
                        log.exception("tx: on_first_accept callback raised")
        elif ret == 0:
            # "accepted nothing" — pre-fix loops treated it as progressless success
            # and spun forever (P0-07). Bound the repeats.
            stalls += 1
            if stalls > max_stalls:
                return _finish(i, n, "stalled", f"no progress after {stalls} writes")
        else:
            # Any negative Soapy result (TIMEOUT/UNDERFLOW/STREAM_ERROR/...) is an
            # error, matching the bench probe's ret<=0 = failure rule.
            return _finish(i, n, "error", f"writeStream ret={ret}")
    # Finding #22: every sample was ACCEPTED into the driver, but acceptance is not
    # radiation. Before reporting a clean burst, drain the driver's stream-status for a
    # late/underflow/END_ABRUPT discard it only surfaces asynchronously — otherwise an
    # XTRX that threw the whole burst away reports as a fully successful transmission
    # (dead uplink indistinguishable from a good one). Bounded so it cannot hang the
    # keyed window; a status we cannot read leaves the burst reported "complete".
    if poll_status and accepted_any:
        bad, status_detail = poll_tx_status(dev, stream)
        if bad:
            return _finish(i, n, "discarded", status_detail)
    return _finish(i, n, "complete")

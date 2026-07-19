"""Channel-rate sizing shared by the RX engines (pure Python, no GNU Radio).

Historical note: this module used to also pick a brute-force fallback-demod bank
(``fallback_modes`` / ``GS_FALLBACK_DEMODS``). Decode is fully backend-driven now — the engine
builds the ONE ``(modulation, symbol_rate)`` the backend specified (see
``gnuradio_satellites._backend_mode``) — so the bank and its env override were dead code and
have been removed. ``GS_FALLBACK_DEMODS`` in a station environment is harmless and ignored.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import math
from typing import Any

from native_framing.sample_clock import select_channel_rate

# Samples/symbol the channel must give the demods. symbol_sync needs sps>1; ~4 is a
# comfortable margin for GFSK/PSK timing recovery.
CHANNEL_OVERSAMPLE = 4.0

# Maximum live scheduler-handoff drain interval. The symbol queues also have an
# item bound, so this is intentionally much shorter than a frame/report cadence.
LIVE_DECODE_DRAIN_PERIOD_S = 0.05

# The symbol rate reaches a flowgraph under different key names depending on the source:
# gs-client's codec renames the SatNOGS ``baud`` field to ``symbol_rate_hz``, but a raw rfLink, a
# hand-written params.json, or a future/other backend may carry ``baud``/``baudrate`` verbatim.
# Baud and symbol rate are the SAME physical quantity (symbols per second), so the demod MUST treat
# them as interchangeable — a rate under a different key is NOT a missing rate, and the demod chain
# must not go dark because of the key name. This tuple is that single source of truth, in priority
# order (the canonical ``symbol_rate_hz`` first). Keep it in sync wherever a rate is read.
SYMBOL_RATE_KEYS = ("symbol_rate_hz", "baud", "baudrate", "baud_rate", "symbol_rate")


def symbol_rate_hz_of(params: dict[str, Any] | None, default: float = 0.0) -> float:
    """The symbol rate (== baud) from ``params`` under ANY of its alias keys
    (:data:`SYMBOL_RATE_KEYS`), or ``default`` when none is present/usable. Baud and
    ``symbol_rate_hz`` are interchangeable, so the demod never fails just because the rate arrived
    under a different key. A present-but-invalid or non-positive value is skipped (0 baud is not a
    rate) so a later alias — or ``default`` — still applies. Pure: no GNU Radio, unit-testable.

    ROUND 10 — NON-FINITE IS NOT "USABLE". The test was ``v > 0``, and ``inf > 0`` is True, so an
    infinite baud was returned as a perfectly good symbol rate. (NaN happened to fall through to the
    default only because ``nan > 0`` is False — right answer, wrong reason.) This helper feeds EVERY
    app, RX and TX: an infinite rate divides into a zero-sample IQ, or a zero-sps demod, far from
    here and with nothing to point at. The rate comes from the backend's transmitter catalogue,
    which is not ours and has already offered baud=10, so it gets checked here."""
    p = params or {}
    for key in SYMBOL_RATE_KEYS:
        if key in p:
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                return v
    return float(default)


def channel_rate_for(sample_rate: float, symbol_rate_hz: float, sdr_rate: float) -> float:
    """The decimation-target channel rate: at least the requested ``sample_rate``, and
    wide enough for ~CHANNEL_OVERSAMPLE samples/symbol on the bird (so a high-baud bird —
    e.g. 50 kBd at a 48 kHz default — doesn't give symbol_sync sps<1), capped at the SDR
    capture rate (can't decimate to more than we sampled).

    When the channel must WIDEN past the requested rate (a high-baud bird), snap it UP to a rate
    that divides the capture rate, so ``make_decimator`` builds a light interp=1 decimator instead
    of a heavy interp-N polyphase resampler. Without this a 25 kBd bird's 100 kHz channel decimates
    2.048M→100k as ``25/512`` (an interp-25 filter) — needless CPU on the RZ/V2H and a fragile,
    rarely-exercised code path; snapping gives 102400 = 2.048M/20 = a clean ``1/20`` decimation.
    The LOW-baud path (channel == requested ``sample_rate``) is left untouched — it is the proven
    default and its mild resampler already records reliably."""
    return select_channel_rate(
        sample_rate,
        symbol_rate_hz,
        sdr_rate,
        minimum_samples_per_symbol=CHANNEL_OVERSAMPLE,
    )


def should_build_demod(
    *,
    mode: tuple[str, float] | None,
    local_deframer_enabled: bool,
    grsat_live: bool,
) -> bool:
    """Return whether a live demodulator has an enabled downstream decoder.

    Recorder-only passes must not construct a terminal symbol queue. GNU Radio
    would continue filling that queue even though nothing can turn its symbols
    into frames, eventually overflowing it and terminating the recorder.
    """

    return mode is not None and (local_deframer_enabled or grsat_live)


def no_decode_reason(
    *,
    has_decode_consumer: bool,
    mode: tuple[str, float] | None,
    grsat_live: bool,
    framing: str | None = None,
    deframer_available: bool = True,
    native_deframer_available: bool = False,
    native_live: bool = False,
) -> str:
    """R2-02: why (if at all) this graph ended up with NO decoder.

    A graph with no decode consumer is RECORDER-ONLY: it captures IQ and produces exactly
    zero frames. That is a legitimate outcome — the .cf32 is decodable offline — but it must
    never be reported as an ordinary successful decode pass. A green pass with no frames and
    no explanation is indistinguishable from a bird that was simply silent.

    Returns "" when a decoder exists. Pure (no GNU Radio) so it is testable off-bench; the
    engine puts the result on its ``ready`` event and gs-client carries it into the terminal
    PassResult.
    """
    if has_decode_consumer:
        if deframer_available:
            return ""
        # The demod built and the graph LOOKS healthy — it just cannot deframe what it
        # demodulates. A backend framing outside our local vocabulary (AX.100, USP,
        # Mobitex, NGHam, CCSDS Concatenated…) is decodable ONLY by gr-satellites, so
        # with GS_GRSAT_LIVE unset every drain returns nothing, forever, and the engine
        # still logs a success-shaped "our demod fsk@9600 …". Zero frames, no error.
        return (
            f"no deframer: the demod for {mode[0]}@{mode[1]:.0f} was built, but framing "
            f"{framing!r} has no local deframer and gr-satellites is gated off "
            f"(GS_GRSAT_LIVE unset) — NOTHING can turn these symbols into frames"
            if mode else
            f"no deframer: framing {framing!r} has no local deframer and gr-satellites "
            f"is gated off (GS_GRSAT_LIVE unset)"
        )
    if not mode:
        why = (
            "no decoder built: the backend sent no usable demod params (transmitter has no "
            "modulation + symbol rate — a null/zero baud yields none)"
        )
        if not grsat_live:
            why += (
                " and GS_GRSAT_LIVE is unset, so gr-satellites could not supply one either"
            )
        return why
    if not deframer_available:
        if native_deframer_available and not native_live:
            return (
                f"no enabled deframer: native framing {framing!r} exists but "
                "GS_NATIVE_FRAMING_LIVE is unset, and gr-satellites is gated off "
                "(GS_GRSAT_LIVE unset); recorder-only mode remains active"
            )
        return (
            f"no enabled deframer for framing {framing!r}; gr-satellites is gated off "
            "(GS_GRSAT_LIVE unset), so recorder-only mode remains active"
        )
    return (
        f"no decoder built: the demod chain for {mode[0]}@{mode[1]:.0f} "
        f"(framing={framing or 'auto'}) failed to construct"
    )

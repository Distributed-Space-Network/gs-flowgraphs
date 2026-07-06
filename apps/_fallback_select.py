"""Channel-rate sizing shared by the RX engines (pure Python, no GNU Radio).

Historical note: this module used to also pick a brute-force fallback-demod bank
(``fallback_modes`` / ``GS_FALLBACK_DEMODS``). Decode is fully backend-driven now — the engine
builds the ONE ``(modulation, symbol_rate)`` the backend specified (see
``gnuradio_satellites._backend_mode``) — so the bank and its env override were dead code and
have been removed. ``GS_FALLBACK_DEMODS`` in a station environment is harmless and ignored.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

from typing import Any

# Samples/symbol the channel must give the demods. symbol_sync needs sps>1; ~4 is a
# comfortable margin for GFSK/PSK timing recovery.
CHANNEL_OVERSAMPLE = 4.0

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
    rate) so a later alias — or ``default`` — still applies. Pure: no GNU Radio, unit-testable."""
    p = params or {}
    for key in SYMBOL_RATE_KEYS:
        if key in p:
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                continue
            if v > 0:
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
    sr = float(sample_rate)
    want = max(sr, CHANNEL_OVERSAMPLE * float(symbol_rate_hz or 0.0))
    sdr = float(sdr_rate)
    if want >= sdr:
        return sdr                       # can't decimate to more than we sampled
    if want <= sr:
        return sr                        # low-baud: keep the requested rate (proven path)
    # High-baud widening: largest integer decim that divides the capture rate and still leaves
    # channel >= want (so the demod keeps >= CHANNEL_OVERSAMPLE sps). Falls back to the exact
    # ``want`` (rational resampler, old behavior) only if no clean divisor fits.
    sdr_i = int(round(sdr))
    decim = int(sdr // want)             # floor => channel = sdr/decim >= want
    while decim > 1 and sdr_i % decim != 0:
        decim -= 1
    return sdr / decim if decim >= 1 else want

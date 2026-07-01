"""Channel-rate sizing shared by the RX engines (pure Python, no GNU Radio).

Historical note: this module used to also pick a brute-force fallback-demod bank
(``fallback_modes`` / ``GS_FALLBACK_DEMODS``). Decode is fully backend-driven now — the engine
builds the ONE ``(modulation, symbol_rate)`` the backend specified (see
``gnuradio_satellites._backend_mode``) — so the bank and its env override were dead code and
have been removed. ``GS_FALLBACK_DEMODS`` in a station environment is harmless and ignored.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

# Samples/symbol the channel must give the demods. symbol_sync needs sps>1; ~4 is a
# comfortable margin for GFSK/PSK timing recovery.
CHANNEL_OVERSAMPLE = 4.0


def channel_rate_for(sample_rate: float, symbol_rate_hz: float, sdr_rate: float) -> float:
    """The decimation-target channel rate: at least the requested ``sample_rate``, and
    wide enough for ~CHANNEL_OVERSAMPLE samples/symbol on the bird (so a high-baud bird —
    e.g. 50 kBd at a 48 kHz default — doesn't give symbol_sync sps<1), capped at the SDR
    capture rate (can't decimate to more than we sampled)."""
    want = max(float(sample_rate), CHANNEL_OVERSAMPLE * float(symbol_rate_hz or 0.0))
    return min(want, float(sdr_rate))

"""Pick which fallback demods to run for a bird gr-satellites can't decode.

Pure Python (no GNU Radio), so it's importable + unit-tested on any box; the GR engine
(``gnuradio_satellites``) consumes it. The point: the backend usually tells us the
bird's symbol rate (and sometimes modulation) in the pass ``params`` — the SatNOGS
"mode from the catalog" idea — so we target the demod instead of brute-forcing the bank.

Priority (the backend's per-pass mode always wins; we only fall back when it can't give
us one):

1. backend ``params``: a ``modulation`` + ``symbol_rate_hz`` → that one exact demod;
   a ``symbol_rate_hz`` alone → the modulation families that occur at that rate.
2. ``GS_FALLBACK_DEMODS`` env — the operator's chosen fallback set, used when the
   backend gave no usable mode.
3. the full default bank — brute force, when there's neither a backend mode nor an env.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import os
import re

# Full brute-force bank, the last-resort fallback. Override per station with
# GS_FALLBACK_DEMODS (comma list of "<kind><rate>").
DEFAULT_FALLBACK_DEMODS = "gfsk9600,gfsk4800,gmsk9600,gmsk4800,bpsk1200,bpsk9600,qpsk9600,afsk1200"

# Modulation kinds the demod builders understand (see gnuradio_satellites._build_fallbacks).
DEMOD_KINDS = ("gfsk", "fsk", "gmsk", "msk", "bpsk", "qpsk", "psk", "afsk")

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


def modes_from_params(params: dict | None) -> list[str]:
    """The targeted demod(s) from the backend's per-pass mode, or ``[]`` if the params
    carry no usable ``symbol_rate_hz`` (absent / non-numeric / ≤ 0)."""
    p = params or {}
    rate = p.get("symbol_rate_hz")
    if not rate:
        return []
    try:
        r = int(float(rate))
    except (TypeError, ValueError):
        return []
    if r <= 0:
        return []
    kind = re.sub(r"[^a-z]", "", str(p.get("modulation", "")).lower())
    if kind in DEMOD_KINDS:  # backend named the modulation → the exact demod
        return [f"{kind}{r}"]
    # symbol rate known, modulation not: the modulation families seen at that rate.
    cands = [f"gfsk{r}", f"gmsk{r}", f"bpsk{r}"]
    if r <= 1200:
        cands.append(f"afsk{r}")
    return cands


def fallback_modes(params: dict | None) -> list[str]:
    """Return the ``"<kind><rate>"`` demod specs to run. The backend's per-pass mode wins;
    only when it gives no usable mode do we fall back to GS_FALLBACK_DEMODS, then the bank."""
    targeted = modes_from_params(params)
    if targeted:  # 1. backend per-pass mode wins, always
        return targeted
    override = os.environ.get("GS_FALLBACK_DEMODS", "").strip()
    if override:  # 2. operator's fallback set
        return [m.strip() for m in override.split(",") if m.strip()]
    return DEFAULT_FALLBACK_DEMODS.split(",")  # 3. brute-force bank

"""Modem registry — modulation family → GNU Radio demod (and, later, mod) chain.

Phase 0 of docs/08 (universal modem + framing). The modulation→demod dispatch that used to
live inline in ``gnuradio_satellites._build_fallbacks`` lives here now, so new modulations
(QAM / APSK / OFDM / DVB-S2 per docs/08) plug in as isolated registry entries instead of
surgery on the engine. The demod DSP chains themselves live in ``gnuradio_gfsk.py``; this
module is a thin dispatcher that imports GNU Radio **lazily**, so the numpy ``dsp`` engine and
the test suite can import it freely.
"""
from __future__ import annotations

# Modulation family → PSK constellation order (2 = BPSK, 4 = QPSK).
_PSK_ORDER = {"bpsk": 2, "psk": 2, "qpsk": 4}
# 2-FSK family; (G)MSK ≈ h=0.5 CPFSK — all share the FSK demod (deviation from mod_index).
_FSK_KINDS = ("gfsk", "fsk", "gmsk", "msk")


def demod_families() -> set[str]:
    """The modulation families the fallback modem can currently demodulate."""
    return set(_FSK_KINDS) | set(_PSK_ORDER) | {"afsk"}


def build_demod(kind: str, tb, src, sample_rate: float, symbol_rate: float):
    """Build the GNU Radio demod chain for ``kind`` tapping ``src`` (already at the channel
    rate) and return its bit sink (``drain()`` → hard bits), or ``None`` if ``kind`` is not
    supported. Raises like the underlying builder if the chain can't be constructed for this
    channel (the caller guards + skips)."""
    from gnuradio_gfsk import (  # noqa: PLC0415 — GNU Radio only; keeps this module import-safe
        connect_afsk_demod,
        connect_gfsk_demod,
        connect_psk_demod,
    )

    from gfsk_ax25 import endurosat  # noqa: PLC0415

    kind = (kind or "").strip().lower()
    if kind in _FSK_KINDS:
        mod_index = 0.5 if kind in ("gmsk", "msk") else endurosat.LinkProfile().mod_index
        profile = endurosat.LinkProfile(
            symbol_rate_hz=symbol_rate or 9600.0, mod_index=mod_index)
        return connect_gfsk_demod(
            tb, src, sample_rate, profile, decimate=False, sdr_rate=sample_rate)
    if kind in _PSK_ORDER:
        return connect_psk_demod(
            tb, src, sample_rate, symbol_rate or 1200.0, order=_PSK_ORDER[kind])
    if kind == "afsk":
        return connect_afsk_demod(tb, src, sample_rate, baud=symbol_rate or 1200.0)
    return None

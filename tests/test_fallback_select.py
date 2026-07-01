"""Channel-rate sizing (_fallback_select) — the surviving pure helper.

The fallback-demod bank (``fallback_modes`` / ``GS_FALLBACK_DEMODS``) was removed as dead code
(decode is fully backend-driven; docs/10 P2): the engine builds the ONE backend-specified
``(modulation, symbol_rate)``. Only the channel-rate sizing lives here now.
"""
from __future__ import annotations

from _fallback_select import CHANNEL_OVERSAMPLE, channel_rate_for


def test_channel_rate_keeps_requested_rate_for_low_baud():
    # 9k6 bird at a 48 kHz requested channel: 4x oversample needs 38.4k < 48k -> keep 48k.
    assert channel_rate_for(48_000, 9_600, 2_048_000) == 48_000


def test_channel_rate_widens_for_high_baud():
    # 50 kBd needs CHANNEL_OVERSAMPLE * 50k = 200k, wider than the 48k default.
    assert channel_rate_for(48_000, 50_000, 2_048_000) == CHANNEL_OVERSAMPLE * 50_000


def test_channel_rate_capped_at_sdr_rate():
    # Can't decimate to more than we sampled.
    assert channel_rate_for(48_000, 1_000_000, 2_048_000) == 2_048_000


def test_channel_rate_handles_missing_symbol_rate():
    assert channel_rate_for(48_000, 0.0, 2_048_000) == 48_000
    assert channel_rate_for(48_000, None, 2_048_000) == 48_000


def test_bank_api_is_gone():
    # The brute-force bank must not silently return: its API was removed on purpose.
    import _fallback_select

    assert not hasattr(_fallback_select, "fallback_modes")
    assert not hasattr(_fallback_select, "DEFAULT_FALLBACK_DEMODS")

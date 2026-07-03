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


def test_channel_rate_widens_for_high_baud_and_snaps_to_integer_decimation():
    # 50 kBd needs CHANNEL_OVERSAMPLE * 50k = 200k, wider than the 48k default. It snaps UP to
    # 204800 = 2.048M/10 (a clean 1/10 decimation) instead of 200k (a heavy 25/256 resampler).
    from _soapy import resample_ratio

    ch = channel_rate_for(48_000, 50_000, 2_048_000)
    assert ch == 204_800  # >= CHANNEL_OVERSAMPLE*50k, and 2.048M/10
    assert resample_ratio(2_048_000, ch) == (1, 10)  # light integer decimator, interp=1


def test_channel_rate_25kbd_snaps_off_the_heavy_resampler():
    # cmd_71's bird: 25 kBd -> want 100k. OLD gave 100000 -> 2.048M/100k = 25/512 (interp-25
    # polyphase, the fragile heavy path). NEW snaps to 102400 = 2.048M/20 = a clean 1/20.
    from _soapy import resample_ratio

    ch = channel_rate_for(48_000, 25_000, 2_048_000)
    assert ch == 102_400
    assert resample_ratio(2_048_000, ch) == (1, 20)
    assert ch / 25_000 >= CHANNEL_OVERSAMPLE  # still >= 4 samples/symbol for the demod


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

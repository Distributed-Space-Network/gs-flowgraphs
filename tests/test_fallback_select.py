"""Channel-rate sizing (_fallback_select) — the surviving pure helper.

The fallback-demod bank (``fallback_modes`` / ``GS_FALLBACK_DEMODS``) was removed as dead code
(decode is fully backend-driven; docs/10 P2): the engine builds the ONE backend-specified
``(modulation, symbol_rate)``. Only the channel-rate sizing lives here now.
"""
from __future__ import annotations

from _fallback_select import (
    CHANNEL_OVERSAMPLE,
    MAX_ADDITIVE_FRAMINGS,
    channel_rate_for,
    requested_framings,
    symbol_rate_hz_of,
)


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
    # Four samples/symbol at 1 MBd would require 4 MHz from a 2.048 MHz capture. Silently
    # capping this used to construct a graph that could not satisfy its timing contract.
    import pytest

    with pytest.raises(ValueError, match="cannot provide"):
        channel_rate_for(48_000, 1_000_000, 2_048_000)


def test_channel_rate_handles_missing_symbol_rate():
    assert channel_rate_for(48_000, 0.0, 2_048_000) == 48_000
    assert channel_rate_for(48_000, None, 2_048_000) == 48_000


def test_bank_api_is_gone():
    # The brute-force bank must not silently return: its API was removed on purpose.
    import _fallback_select

    assert not hasattr(_fallback_select, "fallback_modes")
    assert not hasattr(_fallback_select, "DEFAULT_FALLBACK_DEMODS")


def test_additive_framings_keep_primary_dedupe_and_order() -> None:
    assert requested_framings(
        {"framing": "AX.25", "framings": ["EnduroSat", "AX.25", "EnduroSat"]}
    ) == ("AX.25", "EnduroSat")
    assert requested_framings({"framing": " USP "}) == ("USP",)
    assert requested_framings({}) == ()


def test_additive_framings_fail_closed_on_malformed_or_oversized_input() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be a list"):
        requested_framings({"framings": "AX.25,EnduroSat"})
    with pytest.raises(ValueError, match="entries must be strings"):
        requested_framings({"framings": ["AX.25", 7]})
    with pytest.raises(ValueError, match="profile limit"):
        requested_framings(
            {"framings": [f"profile-{index}" for index in range(MAX_ADDITIVE_FRAMINGS + 1)]}
        )


# ── symbol_rate_hz_of: baud / symbol_rate_hz are the SAME quantity, interchangeable ───────────
def test_symbol_rate_canonical_key():
    assert symbol_rate_hz_of({"symbol_rate_hz": 2400.0}) == 2400.0


def test_symbol_rate_accepts_baud_alias():
    # The demod must not go dark when the rate arrives as `baud` (SatNOGS field name) instead
    # of `symbol_rate_hz` — they are the same quantity.
    assert symbol_rate_hz_of({"baud": 9600}) == 9600.0


def test_symbol_rate_accepts_baudrate_and_underscore_and_symbol_rate_aliases():
    assert symbol_rate_hz_of({"baudrate": 1200}) == 1200.0
    assert symbol_rate_hz_of({"baud_rate": 4800}) == 4800.0
    assert symbol_rate_hz_of({"symbol_rate": 19200}) == 19200.0


def test_symbol_rate_canonical_wins_over_baud_when_both_present():
    # symbol_rate_hz is first in priority order; a stray/conflicting baud never overrides it.
    assert symbol_rate_hz_of({"symbol_rate_hz": 2400, "baud": 9600}) == 2400.0


def test_symbol_rate_falls_through_invalid_or_nonpositive_to_next_alias():
    # 0 baud is not a rate; a garbage string is not a rate — keep looking, then use the default.
    assert symbol_rate_hz_of({"baud": 0, "baudrate": 9600}) == 9600.0
    assert symbol_rate_hz_of({"baud": "fast", "symbol_rate": 1200}) == 1200.0


def test_symbol_rate_missing_returns_default():
    assert symbol_rate_hz_of({}) == 0.0
    assert symbol_rate_hz_of(None) == 0.0
    assert symbol_rate_hz_of({"framing": "ax25"}, default=9600.0) == 9600.0


def test_symbol_rate_string_number_is_accepted():
    # params.json values are sometimes strings (MessageToDict of a Struct) — coerce.
    assert symbol_rate_hz_of({"baud": "2400"}) == 2400.0

"""Fallback demod selection — backend mode (params) targets the demod, else brute force.

Pure logic (no GNU Radio), so it runs on any box."""

from __future__ import annotations

import pytest
from _fallback_select import (
    CHANNEL_OVERSAMPLE,
    DEFAULT_FALLBACK_DEMODS,
    channel_rate_for,
    fallback_modes,
)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("GS_FALLBACK_DEMODS", raising=False)


def test_no_params_uses_full_bank() -> None:
    assert fallback_modes(None) == DEFAULT_FALLBACK_DEMODS.split(",")
    assert fallback_modes({}) == DEFAULT_FALLBACK_DEMODS.split(",")


def test_symbol_rate_targets_families_at_that_rate() -> None:
    # The cmd_43 case: backend gives symbol_rate_hz=9600 (framing ax25), no modulation.
    assert fallback_modes({"symbol_rate_hz": 9600, "framing": "ax25"}) == [
        "gfsk9600",
        "gmsk9600",
        "bpsk9600",
    ]


def test_low_rate_adds_afsk() -> None:
    assert fallback_modes({"symbol_rate_hz": 1200}) == [
        "gfsk1200",
        "gmsk1200",
        "bpsk1200",
        "afsk1200",
    ]


def test_explicit_modulation_targets_one_demod() -> None:
    assert fallback_modes({"symbol_rate_hz": 9600, "modulation": "gfsk"}) == ["gfsk9600"]
    assert fallback_modes({"symbol_rate_hz": 1200, "modulation": "BPSK"}) == ["bpsk1200"]
    # a scrambler name (not a modulation) is ignored → fall back to the families
    assert fallback_modes({"symbol_rate_hz": 9600, "modulation": "G3RUH"}) == [
        "gfsk9600",
        "gmsk9600",
        "bpsk9600",
    ]


def test_backend_mode_wins_over_env(monkeypatch) -> None:
    # The backend's per-pass mode always wins — even over an operator GS_FALLBACK_DEMODS.
    monkeypatch.setenv("GS_FALLBACK_DEMODS", "bpsk9600,qpsk9600")
    assert fallback_modes({"symbol_rate_hz": 9600, "modulation": "gfsk"}) == ["gfsk9600"]
    assert fallback_modes({"symbol_rate_hz": 9600}) == ["gfsk9600", "gmsk9600", "bpsk9600"]


def test_env_used_only_when_no_backend_mode(monkeypatch) -> None:
    # No usable backend mode → fall back to the operator's GS_FALLBACK_DEMODS.
    monkeypatch.setenv("GS_FALLBACK_DEMODS", "bpsk9600,qpsk9600")
    assert fallback_modes(None) == ["bpsk9600", "qpsk9600"]
    assert fallback_modes({"framing": "ax25"}) == ["bpsk9600", "qpsk9600"]


def test_channel_rate_scales_with_symbol_rate() -> None:
    sdr = 2_048_000.0
    # Low-baud bird: channel stays at the requested --sample-rate (no needless widening).
    assert channel_rate_for(48_000.0, 9_600.0, sdr) == 48_000.0
    assert channel_rate_for(48_000.0, 0.0, sdr) == 48_000.0  # no symbol rate → default
    # High-baud bird (the cmd_48 case): 50 kBd needs ~CHANNEL_OVERSAMPLE×, not 48 kHz.
    assert channel_rate_for(48_000.0, 50_000.0, sdr) == CHANNEL_OVERSAMPLE * 50_000.0
    # Never exceeds the SDR capture rate.
    assert channel_rate_for(48_000.0, 1_000_000.0, sdr) == sdr


def test_bad_or_zero_rate_falls_back_to_bank() -> None:
    # No env, unusable / absent rate → the full brute-force bank.
    bank = DEFAULT_FALLBACK_DEMODS.split(",")
    assert fallback_modes({"symbol_rate_hz": "junk"}) == bank
    assert fallback_modes({"symbol_rate_hz": 0}) == bank
    assert fallback_modes({"symbol_rate_hz": -9600}) == bank
    assert fallback_modes({"symbol_rate_hz": None}) == bank
    assert fallback_modes({"symbol_rate_hz": ""}) == bank


def test_float_string_rate_is_coerced() -> None:
    # The backend often sends symbol_rate_hz as a JSON number → float; a "9600.0" string
    # must coerce to the integer rate (no "gfsk9600.0").
    assert fallback_modes({"symbol_rate_hz": "9600.0"}) == ["gfsk9600", "gmsk9600", "bpsk9600"]
    assert fallback_modes({"symbol_rate_hz": 9600.0}) == ["gfsk9600", "gmsk9600", "bpsk9600"]


def test_explicit_psk_and_fsk_kinds() -> None:
    assert fallback_modes({"symbol_rate_hz": 1200, "modulation": "qpsk"}) == ["qpsk1200"]
    assert fallback_modes({"symbol_rate_hz": 9600, "modulation": "fsk"}) == ["fsk9600"]
    assert fallback_modes({"symbol_rate_hz": 1200, "modulation": "afsk"}) == ["afsk1200"]

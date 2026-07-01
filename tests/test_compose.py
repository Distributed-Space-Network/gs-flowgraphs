"""docs/08 Phase 4 — the decode composer (rfLink → plan). Pure decision layer."""
from __future__ import annotations

import compose


def test_our_engine_only_local_framing():
    # GFSK + our local AX.25 token, not catalogued, gr-sat framing string absent → our engine.
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "ax25"}, catalogued=False)
    assert plan.our_modem and plan.our_framing and plan.our_engine
    assert not plan.grsat_catalogued
    # gr-sat synthesizable too (gfsk + framing + baud) → this is actually a race
    assert plan.grsat_synthesizable and plan.race


def test_grsatellites_synthetic_when_framing_is_grsat_vocab():
    # USP is a gr-satellites framing (not local) → only the gr-satellites synthetic path.
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 4800, "framing": "USP"}, catalogued=False)
    assert not plan.our_framing and not plan.our_engine
    assert plan.grsat_synthesizable and plan.grsatellites and not plan.race


def test_catalogued_bird_uses_grsatellites_even_without_params():
    plan = compose.plan_decode({}, catalogued=True)
    assert plan.grsat_catalogued and plan.grsatellites and plan.decodable
    assert not plan.our_engine


def test_tier2_modulation_recognized_but_needs_local_framing_for_our_engine():
    # QAM is recognized by the modem (tier 2) but its framing here isn't local → not our-engine;
    # QAM isn't gr-sat-synthesizable either → record-only.
    plan = compose.plan_decode(
        {"modulation": "qam16", "symbol_rate_hz": 200000, "framing": "CCSDS Concatenated"})
    assert plan.our_modem and plan.tier == 2
    assert not plan.our_framing and not plan.grsat_synthesizable
    assert not plan.decodable  # record-only (raw IQ product)


def test_unrecognized_modulation_is_not_decodable():
    plan = compose.plan_decode({"modulation": "smoke", "symbol_rate_hz": 1200, "framing": "ax25"})
    assert not plan.our_modem and plan.tier is None
    assert not plan.decodable


def test_local_ccsds_tm_framing_is_our_engine():
    plan = compose.plan_decode(
        {"modulation": "bpsk", "symbol_rate_hz": 2000000, "framing": "ccsds_tm"})
    assert plan.our_framing and plan.our_engine


def test_describe_is_readable():
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "USP"}, catalogued=False)
    text = plan.describe()
    assert "gfsk" in text and "USP" in text and "gr-satellites" in text

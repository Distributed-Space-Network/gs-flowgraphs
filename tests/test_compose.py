"""docs/08 Phase 4 — the decode composer (rfLink → plan). Pure decision layer."""
from __future__ import annotations

import compose


def test_local_token_ax25_races_grsatellites():
    # The local token "ax25" translates OUTBOUND to the gr-satellites label "AX.25"
    # (framings.to_grsatellites_framing), so BOTH engines decode it — our AX.25 deframer AND
    # gr-satellites' AX.25 via synthetic SatYAML — and race (first valid frame wins). Restores the
    # gr-satellites redundancy the verbatim-token bug had killed for the whole ax25-default class.
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "ax25"}, catalogued=False)
    assert plan.our_modem and plan.our_framing and plan.our_engine
    assert not plan.grsat_catalogued
    assert plan.grsat_synthesizable and plan.race


def test_local_only_framing_is_our_engine_only():
    # A genuinely local-only framing (endurosat/AirMAC) has NO gr-satellites equivalent
    # (to_grsatellites_framing → None) → our engine only, no synthetic racer.
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "endurosat"}, catalogued=False)
    assert plan.our_modem and plan.our_framing and plan.our_engine
    assert not plan.grsat_synthesizable and not plan.race


def test_verbatim_satyaml_label_races_both_engines():
    # The backend passes SatYAML labels VERBATIM (docs/10 P0-2): "AX.25 G3RUH" normalizes to the
    # local ax25 deframer AND is valid gr-satellites vocabulary → both paths → race.
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "AX.25 G3RUH"},
        catalogued=False)
    assert plan.our_framing and plan.our_engine
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

"""docs/08 Tier 1 — modem taxonomy (pure classifier, no GNU Radio).

The DSP chains are built + tuned on the bench; what's testable in CI is the classification that
SELECTS the chain: family, constellation order, differential/offset, and Tier. If this is right,
``build_demod``/``build_mod`` dispatch to the correct GNU Radio block.
"""
from __future__ import annotations

import sys
from types import ModuleType

import modem
import pytest


@pytest.mark.parametrize(
    ("kind", "family", "order"),
    [
        ("gfsk", "fsk", 2), ("2FSK", "fsk", 2), ("GMSK", "fsk", 2), ("msk", "fsk", 2),
        ("cpfsk", "fsk", 2), ("4fsk", "mfsk", 4), ("8fsk", "mfsk", 8),
        ("bpsk", "psk", 2), ("dbpsk", "psk", 2), ("qpsk", "psk", 4), ("oqpsk", "psk", 4),
        ("8psk", "psk", 8), ("afsk", "afsk", 2),
    ],
)
def test_family_and_order(kind, family, order):
    spec = modem.modulation_spec(kind)
    assert spec is not None
    assert spec.family == family
    assert spec.order == order
    assert spec.tier == 1


def test_differential_and_offset_flags():
    assert modem.modulation_spec("dbpsk").differential is True
    assert modem.modulation_spec("dqpsk").differential is True
    assert modem.modulation_spec("bpsk").differential is False
    assert modem.modulation_spec("qpsk").differential is False
    assert modem.modulation_spec("oqpsk").offset is True
    assert modem.modulation_spec("qpsk").offset is False
    assert modem.modulation_spec("BPSK Manchester").manchester is True
    dbpsk_manchester = modem.modulation_spec("DBPSK Manchester")
    assert dbpsk_manchester is not None
    assert dbpsk_manchester.differential is True
    assert dbpsk_manchester.manchester is True


@pytest.mark.parametrize(
    ("kind", "family", "order"),
    [("qam16", "qam", 16), ("qam256", "qam", 256), ("apsk32", "apsk", 32),
     ("ofdm", "ofdm", 0), ("dvbs2", "dvbs2", 0), ("dvb-s2x", "dvbs2", 0)],
)
def test_tier2_classifies_into_distinct_families(kind, family, order):
    spec = modem.modulation_spec(kind)
    assert spec is not None
    assert spec.tier == 2
    assert spec.family == family
    assert spec.order == order
    assert spec.supported is False  # not a Tier-1 in-process chain


@pytest.mark.parametrize(
    ("kind", "family"),
    [("ook", "ook"), ("ask", "ook"), ("cw", "cw"), ("morse", "cw"),
     ("nbfm", "nbfm"), ("fm", "nbfm"), ("wfm", "wfm"), ("am", "am")],
)
def test_tier3_families(kind, family):
    spec = modem.modulation_spec(kind)
    assert spec is not None and spec.tier == 3 and spec.family == family


def test_normalization_is_case_space_underscore_insensitive():
    assert modem.modulation_spec("G F_S K").family == "fsk"
    assert modem.modulation_spec("  BPSK ").family == "psk"


def test_unknown_modulation_is_none():
    assert modem.modulation_spec("") is None
    assert modem.modulation_spec("smoke-signals") is None


def test_build_demod_unrecognized_returns_none_tuple():
    # build_demod returns a 2-tuple (bit_sink, soft_tap) on EVERY path (docs/12 Phase 3). An
    # unrecognized modulation short-circuits BEFORE importing GNU Radio, so this pins the tuple
    # contract off-bench — a regression to a bare `return None` would break `sink, soft = ...`.
    assert modem.build_demod("smoke-signals", None, None, 48_000.0, 1_200.0) == (None, None)
    assert modem.build_demod("", None, None, 48_000.0, 1_200.0) == (None, None)
    assert modem.modulation_spec(None) is None


def test_soft_only_non_fsk_build_does_not_construct_an_unused_hard_queue():
    assert modem.build_demod(
        "bpsk", None, None, 48_000.0, 1_200.0, collect_hard=False
    ) == (None, None)


@pytest.mark.parametrize("value", [0.0, -0.5, 10.01, float("inf"), float("nan")])
def test_explicit_fsk_modulation_index_fails_closed_before_graph_construction(value):
    with pytest.raises(ValueError, match="mod_index"):
        modem.build_demod(
            "gmsk", None, None, 48_000.0, 2_400.0, mod_index=value
        )


def test_explicit_fsk_modulation_index_reaches_shared_demod_profile(monkeypatch):
    captured = {}
    frontend = ModuleType("gnuradio_gfsk")

    def connect_gfsk_demod(_tb, _src, _sample_rate, profile, **_kwargs):
        captured["profile"] = profile
        return object(), object()

    frontend.connect_gfsk_demod = connect_gfsk_demod
    frontend.connect_afsk_demod = lambda *_args, **_kwargs: None
    frontend.connect_psk_demod = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "gnuradio_gfsk", frontend)

    modem.build_demod("gmsk", object(), object(), 48_000.0, 2_400.0, mod_index=0.75)

    assert captured["profile"].mod_index == 0.75


@pytest.mark.parametrize("kind", ["fsk", "2fsk", "ffsk"])
def test_plain_1k2_fsk_selects_satnogs_adaptive_frontend(monkeypatch, kind):
    captured = {}
    frontend = ModuleType("gnuradio_gfsk")

    def connect_gfsk_demod(_tb, _src, _sample_rate, _profile, **kwargs):
        captured.update(kwargs)
        return object(), object()

    frontend.connect_gfsk_demod = connect_gfsk_demod
    frontend.connect_afsk_demod = lambda *_args, **_kwargs: None
    frontend.connect_psk_demod = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "gnuradio_gfsk", frontend)

    modem.build_demod(kind, object(), object(), 48_000.0, 1_200.0)

    assert captured["adaptive_centering"] is True


@pytest.mark.parametrize(
    ("kind", "symbol_rate"),
    [("gmsk", 1_200.0), ("msk", 1_200.0), ("gfsk", 1_200.0), ("fsk", 9_600.0)],
)
def test_continuous_phase_and_odd_ratio_fsk_retain_grsat_frontend(
    monkeypatch, kind, symbol_rate
):
    captured = {}
    frontend = ModuleType("gnuradio_gfsk")

    def connect_gfsk_demod(_tb, _src, _sample_rate, _profile, **kwargs):
        captured.update(kwargs)
        return object(), object()

    frontend.connect_gfsk_demod = connect_gfsk_demod
    frontend.connect_afsk_demod = lambda *_args, **_kwargs: None
    frontend.connect_psk_demod = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "gnuradio_gfsk", frontend)

    modem.build_demod(kind, object(), object(), 48_000.0, symbol_rate)

    assert captured["adaptive_centering"] is False

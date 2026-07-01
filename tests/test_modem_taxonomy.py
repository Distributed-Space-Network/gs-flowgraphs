"""docs/08 Tier 1 — modem taxonomy (pure classifier, no GNU Radio).

The DSP chains are built + tuned on the bench; what's testable in CI is the classification that
SELECTS the chain: family, constellation order, differential/offset, and Tier. If this is right,
``build_demod``/``build_mod`` dispatch to the correct GNU Radio block.
"""
from __future__ import annotations

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


@pytest.mark.parametrize("kind", ["qam16", "qam256", "apsk32", "ofdm", "dvbs2", "dvb-s2x"])
def test_tier2_classifies_but_is_not_tier1(kind):
    spec = modem.modulation_spec(kind)
    assert spec is not None
    assert spec.tier == 2
    assert spec.family == "tier2"
    assert spec.supported is False


def test_normalization_is_case_space_underscore_insensitive():
    assert modem.modulation_spec("G F_S K").family == "fsk"
    assert modem.modulation_spec("  BPSK ").family == "psk"


def test_unknown_modulation_is_none():
    assert modem.modulation_spec("") is None
    assert modem.modulation_spec("smoke-signals") is None
    assert modem.modulation_spec(None) is None

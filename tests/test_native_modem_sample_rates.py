"""NF-MODEM-003 exact sample-rate and symbol-clock contracts."""

from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path

import pytest
from native_framing.sample_clock import (
    SampleClock,
    convert_offset,
    legacy_satnogs_decimation,
    legacy_satnogs_sample_rate,
    select_channel_rate,
)

_PINNED_SATNOGS = (
    Path(__file__).resolve().parents[2]
    / "related-projects"
    / "satnogs-client"
    / "satnogsclient"
    / "radio"
    / "grsat.py"
)


def _pinned_satnogs_rate_oracle():
    """Load only the two pure rate helpers from the pinned comparison source."""

    tree = ast.parse(_PINNED_SATNOGS.read_text(encoding="utf-8"), filename=str(_PINNED_SATNOGS))
    wanted = {"find_decimation", "find_sample_rate"}
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    code = compile(ast.Module(body=definitions, type_ignores=[]), str(_PINNED_SATNOGS), "exec")
    exec(code, namespace)  # noqa: S102 - pinned pure functions
    return namespace["find_sample_rate"]


@pytest.mark.parametrize(
    ("baud", "script", "expected"),
    [
        (1_200, "satnogs_fsk.py", 48_000.0),
        (9_600, "satnogs_fsk.py", 57_600.0),
        (19_200, "satnogs_fsk.py", 76_800.0),
        (1_200, "satnogs_bpsk.py", 48_000.0),
        (9_600, "satnogs_bpsk.py", 76_800.0),
        (19_200, "satnogs_ssb.py", 76_800.0),
        (9_600, "satnogs_sstv.py", 66_560.0),
        (9_600, "satnogs_apt.py", 66_560.0),
        (9_600, "satnogs_afsk.py", 48_000.0),
    ],
)
def test_legacy_satnogs_sample_rate_matrix(baud: int, script: str, expected: float) -> None:
    assert legacy_satnogs_sample_rate(baud, script=script) == expected


@pytest.mark.parametrize(
    ("baud", "script"),
    [
        (1_200, "satnogs_fsk.py"),
        (9_600, "satnogs_fsk.py"),
        (19_200, "satnogs_fsk.py"),
        (1_200, "satnogs_bpsk.py"),
        (9_600, "satnogs_bpsk.py"),
        (19_200, "satnogs_ssb.py"),
        (9_600, "satnogs_sstv.py"),
        (9_600, "satnogs_apt.py"),
        (9_600, "satnogs_afsk.py"),
    ],
)
def test_legacy_matrix_matches_pinned_satnogs_helper(baud: int, script: str) -> None:
    oracle = _pinned_satnogs_rate_oracle()
    assert callable(oracle)
    assert legacy_satnogs_sample_rate(baud, script=script) == oracle(baud, script)


def test_legacy_decimation_alignment_and_invalid_baud_fallback() -> None:
    assert legacy_satnogs_decimation(1_200, minimum=2, multiple=4) == 40
    assert legacy_satnogs_decimation(9_600, minimum=2, multiple=4) == 8
    assert legacy_satnogs_sample_rate("bad", script="satnogs_fsk.py") == 57_600.0
    assert legacy_satnogs_sample_rate(float("inf"), script="satnogs_fsk.py") == 57_600.0


def test_fractional_sps_clock_maps_absolute_offsets_without_drift() -> None:
    clock = SampleClock(44_100.0, 1_200.0)
    assert clock.samples_per_symbol == 36.75
    assert clock.sample_offset_for_symbol(1) == 37
    assert clock.sample_offset_for_symbol(4) == 147
    assert clock.sample_offset_for_symbol(1_000_000) == 36_750_000
    assert clock.sample_offset_for_symbol(4, origin_sample=12_345) == 12_492
    assert clock.elapsed_seconds_for_sample(44_100) == Fraction(1, 1)


def test_clock_domain_conversion_is_absolute_and_round_trip_bounded() -> None:
    clock = SampleClock(48_000.0, 9_600.0)
    for symbol in (0, 1, 17, 100_001):
        sample = clock.sample_offset_for_symbol(symbol)
        assert clock.symbol_offset_for_sample(sample) == symbol
    assert convert_offset(44_100, from_rate_hz=44_100, to_rate_hz=48_000) == 48_000


def test_channel_rate_selection_preserves_fractional_sps_and_integer_decimation() -> None:
    assert select_channel_rate(44_100, 1_200, 2_048_000) == 44_100
    assert select_channel_rate(48_000, 50_000, 2_048_000) == 204_800
    assert select_channel_rate(48_000, 25_000, 2_048_000) == 102_400


@pytest.mark.parametrize(
    ("sample_rate", "symbol_rate", "message"),
    [
        (0.0, 1_200.0, "positive"),
        (48_000.0, 0.0, "positive"),
        (float("nan"), 1_200.0, "finite"),
        (48_000.0, float("inf"), "finite"),
        (48_000.0, 30_000.0, "samples/symbol"),
    ],
)
def test_sample_clock_rejects_invalid_or_nyquist_limited_rates(
    sample_rate: float, symbol_rate: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SampleClock(sample_rate, symbol_rate)


def test_channel_plan_rejects_impossible_and_inverted_rate_hierarchies() -> None:
    with pytest.raises(ValueError, match="cannot provide"):
        select_channel_rate(48_000, 600_000, 2_048_000)
    with pytest.raises(ValueError, match="exceeds"):
        select_channel_rate(96_000, 1_200, 48_000)
    with pytest.raises(ValueError, match="not bool"):
        select_channel_rate(48_000, False, 2_048_000)

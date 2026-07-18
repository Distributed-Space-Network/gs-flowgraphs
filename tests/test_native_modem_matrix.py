"""NF-MODEM-001 executable native RX pairing matrix."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from native_framing.modem_matrix import (
    PairingStatus,
    RxExecution,
    plan_native_rx_pairing,
)
from native_framing.registry import REGISTRY
from native_framing.types import SymbolInput

_PINNED_FLOWGRAPHS = (
    Path(__file__).resolve().parents[2]
    / "related-projects"
    / "satnogs-client"
    / "satnogsclient"
    / "radio"
    / "flowgraphs.py"
)

_HISTORICAL_PAIRINGS = (
    ("BPSK", "bpsk"),
    ("FSK", "fsk"),
    ("FSK AX.100 Mode 5", "fsk"),
    ("FSK AX.100 Mode 6", "fsk"),
    ("GFSK", "gfsk"),
    ("GMSK", "gmsk"),
    ("MSK", "msk"),
    ("MSK AX.100 Mode 5", "msk"),
    ("MSK AX.100 Mode 6", "msk"),
)


def _pinned_framed_modes() -> dict[str, str]:
    tree = ast.parse(
        _PINNED_FLOWGRAPHS.read_text(encoding="utf-8"), filename=str(_PINNED_FLOWGRAPHS)
    )
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SATNOGS_FLOWGRAPH_MODES"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)
    output: dict[str, str] = {}
    for mode_node, config_node in zip(
        assignment.value.keys, assignment.value.values, strict=True
    ):
        if not isinstance(mode_node, ast.Constant) or not isinstance(config_node, ast.Dict):
            continue
        for key_node, value_node in zip(config_node.keys, config_node.values, strict=True):
            if (
                isinstance(key_node, ast.Constant)
                and key_node.value == "framing"
                and isinstance(value_node, ast.Constant)
            ):
                output[str(mode_node.value)] = str(value_node.value)
    return output


@pytest.mark.parametrize(("mode", "modulation"), _HISTORICAL_PAIRINGS)
def test_historical_satnogs_pairings_have_executable_native_live_plans(
    mode: str, modulation: str
) -> None:
    framing = _pinned_framed_modes()[mode]
    plan = plan_native_rx_pairing(
        framing,
        modulation,
        sample_rate_hz=48_000,
        symbol_rate_hz=9_600,
        capture_rate_hz=2_048_000,
        execution=RxExecution.LIVE,
        evaluation=True,
    )
    assert plan.accepted, (mode, plan.reason)
    assert plan.symbol_input is SymbolInput.HARD_BITS
    assert plan.implementation.startswith("gnuradio-")
    assert plan.channel_rate_hz == 48_000


def test_every_advertised_profile_has_truthful_live_and_postpass_evaluation_plan() -> None:
    assert len(REGISTRY.profiles) == 28
    for profile in REGISTRY.profiles:
        live = plan_native_rx_pairing(
            profile.advertised_label,
            "gfsk",
            sample_rate_hz=48_000,
            symbol_rate_hz=9_600,
            capture_rate_hz=2_048_000,
            execution=RxExecution.LIVE,
            evaluation=True,
        )
        assert live.accepted, (profile.canonical, live.reason)
        expected_suffix = "hard" if profile.symbol_input is SymbolInput.HARD_BITS else "soft"
        assert live.implementation == f"gnuradio-binary-fsk-{expected_suffix}"

        postpass = plan_native_rx_pairing(
            profile.advertised_label,
            "gfsk",
            sample_rate_hz=48_000,
            symbol_rate_hz=9_600,
            capture_rate_hz=48_000,
            execution=RxExecution.POST_PASS,
            evaluation=True,
        )
        assert postpass.accepted, (profile.canonical, postpass.reason)
        assert postpass.implementation == "numpy-binary-fsk-replay"


def test_production_flags_and_explicit_evaluation_are_not_conflated() -> None:
    native = plan_native_rx_pairing(
        "AX.25",
        "gfsk",
        sample_rate_hz=48_000,
        symbol_rate_hz=9_600,
        capture_rate_hz=48_000,
        execution=RxExecution.LIVE,
    )
    assert native.status is PairingStatus.READY and native.production_ready

    gated = plan_native_rx_pairing(
        "GEOSCAN",
        "gfsk",
        sample_rate_hz=48_000,
        symbol_rate_hz=9_600,
        capture_rate_hz=48_000,
        execution=RxExecution.LIVE,
    )
    assert gated.status is PairingStatus.REJECTED
    assert "production-enabled" in gated.reason

    evaluation = plan_native_rx_pairing(
        "GEOSCAN",
        "gfsk",
        sample_rate_hz=48_000,
        symbol_rate_hz=9_600,
        capture_rate_hz=48_000,
        execution=RxExecution.LIVE,
        evaluation=True,
    )
    assert evaluation.status is PairingStatus.EVALUATION_ONLY
    assert not evaluation.production_ready

    bpsk = plan_native_rx_pairing(
        "AX.25",
        "bpsk",
        sample_rate_hz=48_000,
        symbol_rate_hz=1_200,
        capture_rate_hz=48_000,
        execution=RxExecution.POST_PASS,
        evaluation=True,
    )
    assert bpsk.status is PairingStatus.EVALUATION_ONLY
    assert bpsk.implementation == "numpy-bpsk-replay"
    assert not bpsk.production_ready

    manchester = plan_native_rx_pairing(
        "AX.25",
        "DBPSK Manchester",
        sample_rate_hz=48_000,
        symbol_rate_hz=1_200,
        capture_rate_hz=48_000,
        execution=RxExecution.POST_PASS,
        evaluation=True,
    )
    assert manchester.status is PairingStatus.EVALUATION_ONLY
    assert manchester.implementation == "numpy-bpsk-manchester-replay"
    assert not manchester.production_ready


@pytest.mark.parametrize(
    ("framing", "modulation", "execution", "reason"),
    [
        ("unknown", "gfsk", RxExecution.LIVE, "unknown native framing"),
        ("AX.25", "unknown", RxExecution.LIVE, "unknown modulation"),
        ("AX.25", "4fsk", RxExecution.LIVE, "no validated native live"),
        ("AX.25", "qam16", RxExecution.LIVE, "no validated native live"),
        ("AX.25", "oqpsk", RxExecution.LIVE, "half-symbol"),
        ("AX.25", "8psk", RxExecution.LIVE, "8-PSK"),
        ("AX.25", "qpsk", RxExecution.POST_PASS, "no native post-pass"),
        ("Mobitex", "bpsk", RxExecution.LIVE, "no native soft-symbol tap"),
    ],
)
def test_unavailable_pairings_are_rejected_with_actionable_reasons(
    framing: str, modulation: str, execution: RxExecution, reason: str
) -> None:
    plan = plan_native_rx_pairing(
        framing,
        modulation,
        sample_rate_hz=48_000,
        symbol_rate_hz=9_600,
        capture_rate_hz=48_000,
        execution=execution,
        evaluation=True,
    )
    assert plan.status is PairingStatus.REJECTED
    assert reason in plan.reason


def test_impossible_clock_is_rejected_before_any_builder_can_run() -> None:
    plan = plan_native_rx_pairing(
        "AX.25",
        "gfsk",
        sample_rate_hz=48_000,
        symbol_rate_hz=600_000,
        capture_rate_hz=2_048_000,
        execution=RxExecution.LIVE,
        evaluation=True,
    )
    assert plan.status is PairingStatus.REJECTED
    assert "cannot provide" in plan.reason


def test_execution_argument_is_typed_and_fail_closed() -> None:
    with pytest.raises(ValueError, match="RxExecution"):
        plan_native_rx_pairing(
            "AX.25",
            "gfsk",
            sample_rate_hz=48_000,
            symbol_rate_hz=9_600,
            capture_rate_hz=48_000,
            execution="live",  # type: ignore[arg-type]
        )

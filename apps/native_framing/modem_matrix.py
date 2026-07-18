"""Executable native RX modem/profile pairing contract.

This module describes what the current code can actually execute.  Modulation
taxonomy alone is not capability: several recognized families still have no
working builder, and the live GNU Radio bridge currently exposes hard decisions
only to native decoders.  Every caller therefore receives an explicit ready,
evaluation-only, or rejected plan before constructing a graph or replay decoder.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import modem

from native_framing.registry import resolve_profile
from native_framing.sample_clock import select_channel_rate
from native_framing.types import SymbolInput


class RxExecution(str, Enum):
    LIVE = "live"
    POST_PASS = "post_pass"


class PairingStatus(str, Enum):
    READY = "ready"
    EVALUATION_ONLY = "evaluation_only"
    REJECTED = "rejected"


@dataclass(frozen=True)
class NativeRxPairing:
    status: PairingStatus
    execution: RxExecution
    framing: str
    modulation: str
    symbol_input: SymbolInput | None
    implementation: str
    channel_rate_hz: float | None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status is not PairingStatus.REJECTED

    @property
    def production_ready(self) -> bool:
        return self.status is PairingStatus.READY


def _rejected(
    execution: RxExecution,
    framing: object,
    modulation: object,
    reason: str,
    *,
    symbol_input: SymbolInput | None = None,
    channel_rate_hz: float | None = None,
) -> NativeRxPairing:
    return NativeRxPairing(
        PairingStatus.REJECTED,
        execution,
        str(framing or "").strip(),
        str(modulation or "").strip().lower(),
        symbol_input,
        "",
        channel_rate_hz,
        reason,
    )


def _live_implementation(
    spec: modem.ModSpec, symbol_input: SymbolInput
) -> tuple[str, str, bool]:
    if symbol_input is SymbolInput.SOFT_SYMBOLS:
        if spec.family == "fsk" and spec.order == 2:
            return "gnuradio-binary-fsk-soft", "", True
        return "", "this live modem exposes no native soft-symbol tap", False
    if spec.family == "fsk" and spec.order == 2:
        return "gnuradio-binary-fsk-hard", "", True
    if spec.family == "afsk":
        return "gnuradio-afsk-hard", "", False
    if spec.family == "psk":
        if spec.manchester:
            return "", "Manchester BPSK live sync is bench-pending", False
        if spec.offset:
            return "", "OQPSK half-symbol staggering is bench-pending", False
        if spec.order == 8:
            return "", "8-PSK loop and mapping parity are bench-pending", False
        if spec.differential and spec.order > 2:
            return "", "differential M-PSK phase mapping is bench-pending", False
        return "gnuradio-psk-hard", "", False
    return "", f"{spec.family} has no validated native live symbol builder", False


def _post_pass_implementation(spec: modem.ModSpec) -> tuple[str, str, bool]:
    if spec.family == "fsk" and spec.order == 2:
        return "numpy-binary-fsk-replay", "", True
    if spec.family == "afsk":
        return "numpy-afsk-replay", "", False
    if spec.family == "psk" and spec.order == 2 and not spec.offset:
        implementation = (
            "numpy-bpsk-manchester-replay"
            if spec.manchester
            else "numpy-bpsk-replay"
        )
        return implementation, "", False
    return "", f"{spec.family} has no native post-pass symbol replay", False


def plan_native_rx_pairing(
    framing: object,
    modulation: object,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    capture_rate_hz: float,
    execution: RxExecution,
    evaluation: bool = False,
) -> NativeRxPairing:
    """Return an executable native pairing or an explicit rejection reason."""

    if not isinstance(execution, RxExecution):
        raise ValueError("execution must be an RxExecution")
    profile = resolve_profile(framing)
    if profile is None:
        return _rejected(execution, framing, modulation, "unknown native framing profile")
    if not profile.decoder_available:
        return _rejected(
            execution,
            framing,
            modulation,
            "native framing profile has no decoder factory",
            symbol_input=profile.symbol_input,
        )
    spec = modem.modulation_spec(str(modulation or ""))
    if spec is None:
        return _rejected(
            execution,
            framing,
            modulation,
            "unknown modulation",
            symbol_input=profile.symbol_input,
        )
    try:
        channel_rate = select_channel_rate(
            sample_rate_hz,
            symbol_rate_hz,
            capture_rate_hz,
        )
    except ValueError as exc:
        return _rejected(
            execution,
            framing,
            modulation,
            str(exc),
            symbol_input=profile.symbol_input,
        )

    if execution is RxExecution.LIVE:
        implementation, reason, modem_ready = _live_implementation(
            spec, profile.symbol_input
        )
        enabled = profile.live_supported and modem_ready
    else:
        implementation, reason, modem_ready = _post_pass_implementation(spec)
        enabled = profile.post_pass_supported and modem_ready
    if not implementation:
        return _rejected(
            execution,
            profile.canonical,
            spec.kind,
            reason,
            symbol_input=profile.symbol_input,
            channel_rate_hz=channel_rate,
        )
    if not enabled and not evaluation:
        return _rejected(
            execution,
            profile.canonical,
            spec.kind,
            f"{execution.value} use is not production-enabled for this profile",
            symbol_input=profile.symbol_input,
            channel_rate_hz=channel_rate,
        )
    status = PairingStatus.READY if enabled else PairingStatus.EVALUATION_ONLY
    return NativeRxPairing(
        status,
        execution,
        profile.canonical,
        spec.kind,
        profile.symbol_input,
        implementation,
        channel_rate,
        "" if enabled else "explicit evaluation path; production gate remains closed",
    )


__all__ = [
    "NativeRxPairing",
    "PairingStatus",
    "RxExecution",
    "plan_native_rx_pairing",
]

"""Waveform rate-plan validation (R-19) — import-safe (no GNU Radio).

The FM apps' DSP chains only produce their ADVERTISED output when the channel
sample rate divides cleanly through the IF and audio stages. The shipped
configuration once declared 1 MHz for a chain whose audio stage then ran at
50 kHz while the ready event advertised 48 kHz — every consumer of the audio
product resampled garbage. Each app validates its spawn rate through these
pure functions and REFUSES loudly (fail-closed at spawn, R-11) instead of
producing mislabeled output.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

AUDIO_RATE_HZ = 48_000  # spec canonical audio product rate (§A.9.4)
IF_RATE_HZ = 192_000  # FM RX IF target: 4x audio for clean LPF + demod headroom


def fm_rx_plan(sample_rate_hz: float) -> tuple[int, int, int]:
    """Return ``(decim_to_if, if_rate_hz, audio_decim)`` for the FM RX chain,
    or raise ``ValueError`` when the chain cannot produce EXACTLY 48 kHz audio
    from ``sample_rate_hz`` (R-19: 1 MHz gave 50 kHz audio labeled 48 kHz)."""
    rate = int(sample_rate_hz)
    if rate <= 0:
        msg = f"FM RX sample rate must be positive, got {sample_rate_hz}"
        raise ValueError(msg)
    decim_to_if = max(1, rate // IF_RATE_HZ)
    if rate % decim_to_if:
        msg = (
            f"FM RX rate plan invalid: {rate} Hz does not divide by the IF "
            f"decimation {decim_to_if} — pick a multiple of {AUDIO_RATE_HZ} Hz"
        )
        raise ValueError(msg)
    if_rate = rate // decim_to_if
    audio_decim = max(1, if_rate // AUDIO_RATE_HZ)
    if if_rate != audio_decim * AUDIO_RATE_HZ:
        msg = (
            f"FM RX rate plan invalid: {rate} Hz yields a {if_rate / audio_decim:.0f} Hz "
            f"audio product but the app advertises {AUDIO_RATE_HZ} Hz — pick a "
            f"rate whose IF stage divides to exactly {AUDIO_RATE_HZ} Hz (e.g. 192000)"
        )
        raise ValueError(msg)
    return decim_to_if, if_rate, audio_decim


__all__ = ["AUDIO_RATE_HZ", "IF_RATE_HZ", "fm_rx_plan"]

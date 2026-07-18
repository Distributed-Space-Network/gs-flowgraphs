"""Out-of-bench CPU, allocation, and retained-state regression guardrails.

These generous process-local limits catch accidental algorithmic regressions.  They
do not replace the station-class Linux RSS and coexistence evidence required by
NF-BENCH-001.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from native_framing import build_decoder
from native_framing.codes.ra import ra_wire_soft
from native_framing.profiles.smogp_ra import SYNCWORD as RA_SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import SymbolInput

_NOISE_SYMBOLS_PER_PROFILE = 16_384
_ALL_PROFILE_CPU_BUDGET_SECONDS = 20.0
_ALL_PROFILE_TRACED_PEAK_BYTES = 8 * 1024 * 1024
_RA_256_CPU_BUDGET_SECONDS = 12.0
_RA_256_TRACED_PEAK_BYTES = 4 * 1024 * 1024


def _measure(operation):
    tracemalloc.start()
    started = time.process_time()
    try:
        result = operation()
        cpu_seconds = time.process_time() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, cpu_seconds, peak_bytes


def test_all_native_profiles_have_bounded_noise_cpu_memory_and_state() -> None:
    rng = np.random.default_rng(0x5A7E)

    def exercise() -> None:
        assert len(REGISTRY.profiles) == 28
        for profile in REGISTRY.profiles:
            decoder = build_decoder(profile.advertised_label)
            if profile.symbol_input is SymbolInput.HARD_BITS:
                symbols = rng.integers(
                    0, 2, _NOISE_SYMBOLS_PER_PROFILE, dtype=np.uint8
                )
            else:
                symbols = rng.choice(
                    np.asarray([-1.0, 1.0]), _NOISE_SYMBOLS_PER_PROFILE
                )
            decoder.push(symbols)
            assert decoder.retained_symbols <= decoder.max_retained_symbols
            assert decoder.max_retained_symbols <= profile.max_retained_symbols
            decoder.flush()
            assert decoder.retained_symbols == 0

    _, cpu_seconds, peak_bytes = _measure(exercise)
    assert cpu_seconds < _ALL_PROFILE_CPU_BUDGET_SECONDS
    assert peak_bytes < _ALL_PROFILE_TRACED_PEAK_BYTES


def test_worst_size_smogp_ra_success_path_has_cpu_and_memory_headroom() -> None:
    payload = bytes((index * 37 + 256) & 0xFF for index in range(256))
    sync = np.fromiter(
        (0.9 if character == "1" else -0.9 for character in RA_SYNCWORD),
        dtype=np.float64,
    )
    stream = np.concatenate((sync, ra_wire_soft(payload, magnitude=0.8)))

    def decode():
        return build_decoder("SMOG-P RA", {"frame_size": 256}).push(stream)

    frames, cpu_seconds, peak_bytes = _measure(decode)
    assert [frame.payload for frame in frames] == [payload]
    assert cpu_seconds < _RA_256_CPU_BUDGET_SECONDS
    assert peak_bytes < _RA_256_TRACED_PEAK_BYTES

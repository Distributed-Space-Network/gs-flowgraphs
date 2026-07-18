"""Bounded hard/soft sync correlation independent of GNU Radio scheduling.

Soft symbols use the project convention ``positive => bit 1``.  The reported
distance is a confidence-weighted bit distance: it is exactly Hamming distance
for symbols in ``{-1, +1}`` and is invariant to positive scale.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from native_framing.types import Polarity, SymbolInput


@dataclass(frozen=True)
class SyncMatch:
    source_start: int
    source_end: int
    distance: float
    polarity: Polarity


class StreamingSync:
    def __init__(
        self,
        pattern: Sequence[int] | str,
        *,
        threshold: float = 0,
        symbol_input: SymbolInput = SymbolInput.HARD_BITS,
        accept_inverted: bool = False,
    ) -> None:
        if isinstance(pattern, str):
            if not pattern or set(pattern) - {"0", "1"}:
                raise ValueError("sync pattern must be a non-empty binary string")
            expected = np.fromiter((char == "1" for char in pattern), dtype=np.uint8)
        else:
            expected = np.asarray(pattern)
            if expected.ndim != 1 or not expected.size:
                raise ValueError("sync pattern must be non-empty and one-dimensional")
            if not np.all((expected == 0) | (expected == 1)):
                raise ValueError("sync pattern may contain only 0 and 1")
            expected = expected.astype(np.uint8, copy=False)
        if threshold < 0 or threshold > expected.size:
            raise ValueError("sync threshold must be between zero and the pattern length")
        self._pattern = expected
        self._signs = expected.astype(np.float64) * 2.0 - 1.0
        self._threshold = float(threshold)
        self._symbol_input = symbol_input
        self._accept_inverted = bool(accept_inverted)
        dtype = np.uint8 if symbol_input is SymbolInput.HARD_BITS else np.float64
        self._tail = np.empty(0, dtype=dtype)
        self._total_seen = 0

    @property
    def retained_symbols(self) -> int:
        return int(self._tail.size)

    @property
    def max_retained_symbols(self) -> int:
        return int(self._pattern.size - 1)

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[SyncMatch]:
        chunk = np.asarray(symbols)
        if chunk.ndim != 1:
            raise ValueError("symbol chunks must be one-dimensional")
        if self._symbol_input is SymbolInput.HARD_BITS:
            if chunk.size and not np.all((chunk == 0) | (chunk == 1)):
                raise ValueError("hard symbols may contain only 0 and 1")
            normalized = chunk.astype(np.uint8, copy=False)
        else:
            normalized = chunk.astype(np.float64, copy=False)
            if normalized.size and not np.all(np.isfinite(normalized)):
                raise ValueError("soft symbols must be finite")

        combined = np.concatenate((self._tail, normalized))
        source_base = self._total_seen - self._tail.size
        matches: list[SyncMatch] = []
        width = self._pattern.size
        for index in range(max(0, combined.size - width + 1)):
            window = combined[index : index + width]
            distance = self._distance(window)
            polarity = Polarity.NORMAL
            if self._accept_inverted:
                inverted_distance = width - distance
                if inverted_distance < distance:
                    distance = inverted_distance
                    polarity = Polarity.INVERTED
                elif inverted_distance == distance and distance <= self._threshold:
                    polarity = Polarity.AMBIGUOUS
            if distance <= self._threshold:
                start = int(source_base + index)
                matches.append(SyncMatch(start, start + width, float(distance), polarity))

        self._total_seen += normalized.size
        keep = min(self.max_retained_symbols, combined.size)
        self._tail = combined[-keep:].copy() if keep else combined[:0].copy()
        return matches

    def _distance(self, window: np.ndarray) -> float:
        if self._symbol_input is SymbolInput.HARD_BITS:
            return float(np.count_nonzero(window != self._pattern))
        magnitude = float(np.abs(window).sum())
        if magnitude == 0:
            return float(self._pattern.size)
        correlation = float(np.dot(window, self._signs) / magnitude)
        correlation = max(-1.0, min(1.0, correlation))
        return (1.0 - correlation) * self._pattern.size / 2.0

    def flush(self) -> list[SyncMatch]:
        self._tail = self._tail[:0].copy()
        return []


__all__ = ["StreamingSync", "SyncMatch"]

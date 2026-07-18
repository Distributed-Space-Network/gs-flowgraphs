"""Reusable bounded fixed-length hard-bit and soft-symbol frame collectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from native_framing.sync import StreamingSync, SyncMatch
from native_framing.types import FrameResult, IntegrityStatus, Polarity, SymbolInput


@dataclass(frozen=True)
class DecodedFixedFrame:
    payload: bytes
    integrity: IntegrityStatus = IntegrityStatus.PASSED
    corrected_symbols: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingFrame:
    sync_start: int
    capture_start: int
    capture_end: int
    sync_distance: float
    polarity: Polarity


WireDecoder = Callable[[bytes], DecodedFixedFrame | None]
SoftDecoder = Callable[[np.ndarray], DecodedFixedFrame | None]


class FixedSyncFrameDecoder:
    """Collect a fixed number of packed MSB-first bytes after a hard sync."""

    def __init__(
        self,
        *,
        canonical: str,
        syncword: str,
        frame_size: int,
        sync_threshold: int,
        decode_wire: WireDecoder,
    ) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._canonical = canonical
        self._frame_bits = frame_size * 8
        self._decode_wire = decode_wire
        self._syncword_length = len(syncword)
        self._sync = StreamingSync(
            syncword,
            threshold=sync_threshold,
            symbol_input=SymbolInput.HARD_BITS,
            accept_inverted=True,
        )
        self._buffer = np.empty(0, dtype=np.uint8)
        self._buffer_base = 0
        self._total_seen = 0
        self._pending: list[_PendingFrame] = []

    @property
    def retained_symbols(self) -> int:
        return int(self._buffer.size + self._sync.retained_symbols)

    @property
    def max_retained_symbols(self) -> int:
        return self._frame_bits + self._syncword_length - 1

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        bits = np.asarray(symbols)
        if bits.ndim != 1:
            raise ValueError("hard-bit chunks must be one-dimensional")
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValueError("hard-bit chunks may contain only 0 and 1")
        hard = bits.astype(np.uint8, copy=False)
        if not self._buffer.size:
            self._buffer_base = self._total_seen
        self._buffer = np.concatenate((self._buffer, hard))
        matches = self._sync.push(hard)
        self._total_seen += hard.size
        self._pending.extend(self._pending_from_match(match) for match in matches)

        output: list[FrameResult] = []
        waiting: list[_PendingFrame] = []
        for pending in self._pending:
            if pending.capture_end > self._total_seen:
                waiting.append(pending)
                continue
            result = self._decode(pending)
            if result is not None:
                output.append(result)
        self._pending = waiting
        self._trim_buffer()
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError(f"{self._canonical} decoder retained-symbol bound violated")
        return output

    def _pending_from_match(self, match: SyncMatch) -> _PendingFrame:
        return _PendingFrame(
            sync_start=match.source_start,
            capture_start=match.source_end,
            capture_end=match.source_end + self._frame_bits,
            sync_distance=match.distance,
            polarity=match.polarity,
        )

    def _decode(self, pending: _PendingFrame) -> FrameResult | None:
        start = pending.capture_start - self._buffer_base
        end = pending.capture_end - self._buffer_base
        if start < 0 or end > self._buffer.size:
            raise RuntimeError(f"{self._canonical} capture fell outside the retained buffer")
        bits = self._buffer[start:end]
        if pending.polarity is Polarity.INVERTED:
            bits = 1 - bits
        elif pending.polarity is Polarity.AMBIGUOUS:
            return None
        decoded = self._decode_wire(bytes(np.packbits(bits, bitorder="big")))
        if decoded is None:
            return None
        return FrameResult(
            canonical_framing=self._canonical,
            payload=decoded.payload,
            integrity=decoded.integrity,
            source_start=pending.sync_start,
            source_end=pending.capture_end,
            polarity=pending.polarity,
            sync_distance=pending.sync_distance,
            corrected_symbols=decoded.corrected_symbols,
            metadata=decoded.metadata,
        )

    def _trim_buffer(self) -> None:
        if not self._pending:
            self._buffer_base = self._total_seen
            self._buffer = self._buffer[:0].copy()
            return
        keep_from = min(pending.capture_start for pending in self._pending)
        drop = keep_from - self._buffer_base
        if drop > 0:
            self._buffer = self._buffer[drop:].copy()
            self._buffer_base = keep_from

    def flush(self) -> list[FrameResult]:
        self._sync.flush()
        self._pending.clear()
        self._buffer = self._buffer[:0].copy()
        self._buffer_base = self._total_seen
        return []


class FixedSoftSyncFrameDecoder:
    """Collect a fixed number of floating-point symbols after a soft sync."""

    def __init__(
        self,
        *,
        canonical: str,
        syncword: str,
        capture_symbols: int,
        sync_threshold: float,
        decode_symbols: SoftDecoder,
    ) -> None:
        if capture_symbols <= 0:
            raise ValueError("capture_symbols must be positive")
        self._canonical = canonical
        self._capture_symbols = int(capture_symbols)
        self._decode_symbols = decode_symbols
        self._syncword_length = len(syncword)
        self._sync = StreamingSync(
            syncword,
            threshold=sync_threshold,
            symbol_input=SymbolInput.SOFT_SYMBOLS,
            accept_inverted=True,
        )
        self._buffer = np.empty(0, dtype=np.float64)
        self._buffer_base = 0
        self._total_seen = 0
        self._pending: list[_PendingFrame] = []

    @property
    def retained_symbols(self) -> int:
        return int(self._buffer.size + self._sync.retained_symbols)

    @property
    def max_retained_symbols(self) -> int:
        return self._capture_symbols + self._syncword_length - 1

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        chunk = np.asarray(symbols, dtype=np.float64)
        if chunk.ndim != 1:
            raise ValueError("soft-symbol chunks must be one-dimensional")
        if chunk.size and not np.all(np.isfinite(chunk)):
            raise ValueError("soft-symbol chunks must be finite")
        if not self._buffer.size:
            self._buffer_base = self._total_seen
        self._buffer = np.concatenate((self._buffer, chunk))
        matches = self._sync.push(chunk)
        self._total_seen += chunk.size
        self._pending.extend(self._pending_from_match(match) for match in matches)

        output: list[FrameResult] = []
        waiting: list[_PendingFrame] = []
        for pending in self._pending:
            if pending.capture_end > self._total_seen:
                waiting.append(pending)
                continue
            result = self._decode(pending)
            if result is not None:
                output.append(result)
        self._pending = waiting
        self._trim_buffer()
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError(f"{self._canonical} decoder retained-symbol bound violated")
        return output

    def _pending_from_match(self, match: SyncMatch) -> _PendingFrame:
        return _PendingFrame(
            sync_start=match.source_start,
            capture_start=match.source_end,
            capture_end=match.source_end + self._capture_symbols,
            sync_distance=match.distance,
            polarity=match.polarity,
        )

    def _decode(self, pending: _PendingFrame) -> FrameResult | None:
        start = pending.capture_start - self._buffer_base
        end = pending.capture_end - self._buffer_base
        if start < 0 or end > self._buffer.size:
            raise RuntimeError(f"{self._canonical} capture fell outside the retained buffer")
        capture = self._buffer[start:end]
        if pending.polarity is Polarity.INVERTED:
            capture = -capture
        elif pending.polarity is Polarity.AMBIGUOUS:
            return None
        decoded = self._decode_symbols(capture)
        if decoded is None:
            return None
        return FrameResult(
            canonical_framing=self._canonical,
            payload=decoded.payload,
            integrity=decoded.integrity,
            source_start=pending.sync_start,
            source_end=pending.capture_end,
            polarity=pending.polarity,
            sync_distance=pending.sync_distance,
            corrected_symbols=decoded.corrected_symbols,
            metadata=decoded.metadata,
        )

    def _trim_buffer(self) -> None:
        if not self._pending:
            self._buffer_base = self._total_seen
            self._buffer = self._buffer[:0].copy()
            return
        keep_from = min(pending.capture_start for pending in self._pending)
        drop = keep_from - self._buffer_base
        if drop > 0:
            self._buffer = self._buffer[drop:].copy()
            self._buffer_base = keep_from

    def flush(self) -> list[FrameResult]:
        self._sync.flush()
        self._pending.clear()
        self._buffer = self._buffer[:0].copy()
        self._buffer_base = self._total_seen
        return []


class DistributedSoftFrameDecoder:
    """Collect a matrix frame whose sync bits are separated by a fixed step."""

    def __init__(
        self,
        *,
        canonical: str,
        syncword: str,
        step: int,
        sync_threshold: int,
        decode_symbols: SoftDecoder,
    ) -> None:
        if not syncword or set(syncword) - {"0", "1"}:
            raise ValueError("distributed syncword must be a non-empty binary string")
        if step <= 0:
            raise ValueError("distributed sync step must be positive")
        if sync_threshold < 0 or sync_threshold * 2 >= len(syncword):
            raise ValueError("distributed sync threshold must be below half the sync length")
        self._canonical = canonical
        self._pattern = np.fromiter((char == "1" for char in syncword), dtype=np.uint8)
        self._step = int(step)
        self._span = len(syncword) * self._step
        self._threshold = int(sync_threshold)
        self._decode_symbols = decode_symbols
        self._tail = np.empty(0, dtype=np.float64)
        self._total_seen = 0

    @property
    def retained_symbols(self) -> int:
        return int(self._tail.size)

    @property
    def max_retained_symbols(self) -> int:
        return self._span - 1

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        chunk = np.asarray(symbols, dtype=np.float64)
        if chunk.ndim != 1:
            raise ValueError("soft-symbol chunks must be one-dimensional")
        if chunk.size and not np.all(np.isfinite(chunk)):
            raise ValueError("soft-symbol chunks must be finite")
        combined = np.concatenate((self._tail, chunk))
        source_base = self._total_seen - self._tail.size
        output: list[FrameResult] = []
        for index in range(max(0, combined.size - self._span + 1)):
            capture = combined[index : index + self._span]
            received = (capture[:: self._step] >= 0).astype(np.uint8)
            normal_distance = int(np.count_nonzero(received != self._pattern))
            inverted_distance = len(self._pattern) - normal_distance
            if normal_distance < inverted_distance:
                distance = normal_distance
                polarity = Polarity.NORMAL
                normalized = capture
            elif inverted_distance < normal_distance:
                distance = inverted_distance
                polarity = Polarity.INVERTED
                normalized = -capture
            else:
                continue
            if distance > self._threshold:
                continue
            decoded = self._decode_symbols(normalized)
            if decoded is None:
                continue
            start = int(source_base + index)
            output.append(
                FrameResult(
                    canonical_framing=self._canonical,
                    payload=decoded.payload,
                    integrity=decoded.integrity,
                    source_start=start,
                    source_end=start + self._span,
                    polarity=polarity,
                    sync_distance=float(distance),
                    corrected_symbols=decoded.corrected_symbols,
                    metadata=decoded.metadata,
                )
            )

        self._total_seen += chunk.size
        keep = min(self.max_retained_symbols, combined.size)
        self._tail = combined[-keep:].copy() if keep else combined[:0].copy()
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError(f"{self._canonical} decoder retained-symbol bound violated")
        return output

    def flush(self) -> list[FrameResult]:
        self._tail = self._tail[:0].copy()
        return []


__all__ = [
    "DecodedFixedFrame",
    "DistributedSoftFrameDecoder",
    "FixedSoftSyncFrameDecoder",
    "FixedSyncFrameDecoder",
    "SoftDecoder",
    "WireDecoder",
]

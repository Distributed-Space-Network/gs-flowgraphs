"""Bounded native NGHam RS and no-RS receive profiles.

The sync/crop stages are adapted from gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``. The variable codeword lengths,
RS16/RS32 parameters, zero-padding semantics, and official test vector follow
Jon Petter Skagmo's NGHam reference implementation at commit
``29c4fd393049ac3483d9ffa034e867361d0f1764``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from native_framing.codes.ngham import (
    PARITY_SIZES,
    NGHamSize,
    classify_ngham_size,
    decode_ngham_rs,
    remove_ngham_padding,
)
from native_framing.crc import CRC16_X25
from native_framing.linecode import ccsds_randomize
from native_framing.sync import StreamingSync, SyncMatch
from native_framing.types import FrameResult, IntegrityStatus, Polarity, SymbolInput

SYNCWORD = "01011101111001100010101001111110"
CAPTURE_SIZE = 258
DEFAULT_SYNC_THRESHOLD = 4
DEFAULT_TAG_THRESHOLD = 6
_TAG_BITS = 24


@dataclass
class _PendingFrame:
    sync_start: int
    capture_start: int
    sync_distance: float
    polarity: Polarity
    size: NGHamSize | None = None

    @property
    def capture_end(self) -> int | None:
        if self.size is None:
            return None
        return self.capture_start + (3 + self.size.rs_size) * 8


class NGHamFrameDecoder:
    """Collect exactly the codeword length selected by the protected size tag."""

    def __init__(
        self,
        *,
        canonical: str,
        decode_rs: bool,
        sync_threshold: int,
        tag_threshold: int,
    ) -> None:
        self._canonical = canonical
        self._decode_rs = bool(decode_rs)
        self._tag_threshold = int(tag_threshold)
        self._sync = StreamingSync(
            SYNCWORD,
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
        return CAPTURE_SIZE * 8 + len(SYNCWORD) - 1

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
            if pending.size is None and not self._classify_pending(pending):
                if pending.capture_start + _TAG_BITS > self._total_seen:
                    waiting.append(pending)
                continue
            capture_end = pending.capture_end
            if capture_end is None:
                continue
            if capture_end > self._total_seen:
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

    @staticmethod
    def _pending_from_match(match: SyncMatch) -> _PendingFrame:
        return _PendingFrame(
            sync_start=match.source_start,
            capture_start=match.source_end,
            sync_distance=match.distance,
            polarity=match.polarity,
        )

    def _classify_pending(self, pending: _PendingFrame) -> bool:
        if pending.capture_start + _TAG_BITS > self._total_seen:
            return False
        bits = self._slice(pending.capture_start, pending.capture_start + _TAG_BITS)
        normalized = self._normalize(bits, pending.polarity)
        if normalized is None:
            return False
        pending.size = classify_ngham_size(
            bytes(np.packbits(normalized, bitorder="big")),
            max_distance=self._tag_threshold,
        )
        return pending.size is not None

    def _decode(self, pending: _PendingFrame) -> FrameResult | None:
        capture_end = pending.capture_end
        if pending.size is None or capture_end is None:
            return None
        bits = self._slice(pending.capture_start, capture_end)
        normalized = self._normalize(bits, pending.polarity)
        if normalized is None:
            return None
        wire = bytes(np.packbits(normalized, bitorder="big"))
        decoded = self._decode_wire(wire, pending.size)
        if decoded is None:
            return None
        payload, corrected_symbols, metadata = decoded
        return FrameResult(
            canonical_framing=self._canonical,
            payload=payload,
            integrity=IntegrityStatus.PASSED,
            source_start=pending.sync_start,
            source_end=capture_end,
            polarity=pending.polarity,
            sync_distance=pending.sync_distance,
            corrected_symbols=corrected_symbols,
            metadata=metadata,
        )

    def _decode_wire(
        self, wire: bytes, size: NGHamSize
    ) -> tuple[bytes, int | None, Mapping[str, object]] | None:
        if len(wire) != 3 + size.rs_size:
            return None
        derandomized = ccsds_randomize(wire[3:])
        corrected_symbols: int | None = None
        packet = derandomized
        if self._decode_rs:
            rs_result = decode_ngham_rs(derandomized, size)
            if rs_result is None:
                return None
            packet = rs_result.payload
            corrected_symbols = rs_result.corrected_symbols
        unpadded = remove_ngham_padding(packet, size)
        if unpadded is None:
            return None
        packet, padding = unpadded
        payload = CRC16_X25.strip_if_valid(packet, byteorder="big")
        if payload is None:
            return None
        metadata = {
            "size_index": size.index,
            "size_tag_distance": size.tag_distance,
            "rs": self._decode_rs,
            "rs_parity_symbols": PARITY_SIZES[size.index] if self._decode_rs else 0,
            "rs_slot_size": size.rs_size,
            "non_rs_size": size.non_rs_size,
            "padding": padding,
            "randomizer": "CCSDS",
            "crc": CRC16_X25.name,
            "crc_byteorder": "big",
        }
        return payload, corrected_symbols, metadata

    def _slice(self, start: int, end: int) -> np.ndarray:
        local_start = start - self._buffer_base
        local_end = end - self._buffer_base
        if local_start < 0 or local_end > self._buffer.size:
            raise RuntimeError(f"{self._canonical} capture fell outside the retained buffer")
        return self._buffer[local_start:local_end]

    @staticmethod
    def _normalize(bits: np.ndarray, polarity: Polarity) -> np.ndarray | None:
        if polarity is Polarity.INVERTED:
            return 1 - bits
        if polarity is Polarity.AMBIGUOUS:
            return None
        return bits

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


def _build(
    canonical: str, parameters: Mapping[str, object], *, decode_rs: bool
) -> NGHamFrameDecoder:
    return NGHamFrameDecoder(
        canonical=canonical,
        decode_rs=decode_rs,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        tag_threshold=int(parameters.get("tag_threshold", DEFAULT_TAG_THRESHOLD)),
    )


def build_ngham(parameters: Mapping[str, object]) -> NGHamFrameDecoder:
    return _build("ngham", parameters, decode_rs=True)


def build_ngham_no_rs(parameters: Mapping[str, object]) -> NGHamFrameDecoder:
    return _build("ngham_no_rs", parameters, decode_rs=False)


__all__ = [
    "CAPTURE_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "DEFAULT_TAG_THRESHOLD",
    "NGHamFrameDecoder",
    "SYNCWORD",
    "build_ngham",
    "build_ngham_no_rs",
]

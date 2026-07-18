"""Native Mobitex and Mobitex-NX receive deframers.

Copyright 2020 Daniel Estevez <daniel@destevez.net>
Copyright 2025 Fabian P. Schmidt <kerel@mailbox.org>
Adapted for the gs-flowgraphs bounded native streaming API in 2026.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from native_framing.codes.mobitex import (
    MobitexFecStatus,
    decode_mobitex_fec,
    unpack_mobitex_pair,
)
from native_framing.crc import CRC16_X25, CrcSpec
from native_framing.fixed import DecodedFixedFrame
from native_framing.linecode import additive_randomize_bits
from native_framing.sync import StreamingSync, SyncMatch
from native_framing.types import (
    FrameResult,
    IntegrityStatus,
    Polarity,
    SymbolInput,
)

SYNCWORD = "0101011101100101"
NX_SYNCWORD = "0000111011110000"
MAX_BLOCKS = 32
BLOCK_WIRE_BYTES = 30
MAX_CAPTURE_BYTES = 11 + BLOCK_WIRE_BYTES * MAX_BLOCKS
DEFAULT_SYNC_THRESHOLD = 3
DEFAULT_CALLSIGN_THRESHOLD = 2
DEFAULT_KNOWN_CALLSIGN_THRESHOLD = 12
VALID_VARIANTS = frozenset({"default", "BEESAT-1", "BEESAT-9"})

_CALLSIGN_CRC = CrcSpec("Mobitex callsign CRC", 16, 0x1021, 0, 0, False, False)
_INTERLEAVE = np.arange(12 * 20).reshape(12, 20).T.ravel()


def _callsign_syndrome(value: bytes) -> int:
    return _CALLSIGN_CRC.compute(value[:6]) ^ int.from_bytes(value[6:], "big")


def _callsign_error_patterns() -> tuple[dict[int, tuple[int, ...] | None], ...]:
    """Build bounded CRC-syndrome tables, marking non-unique corrections ambiguous."""

    tables: list[dict[int, tuple[int, ...] | None]] = []
    patterns: list[tuple[int, ...]] = [()]
    zero = bytearray(8)
    for threshold in range(DEFAULT_CALLSIGN_THRESHOLD + 1):
        if threshold:
            patterns.extend(combinations(range(64), threshold))
        table: dict[int, tuple[int, ...] | None] = {}
        for positions in patterns:
            error = zero.copy()
            for position in positions:
                error[position // 8] ^= 1 << (position % 8)
            syndrome = _callsign_syndrome(bytes(error))
            if syndrome in table and table[syndrome] != positions:
                table[syndrome] = None
            else:
                table[syndrome] = positions
        tables.append(table)
    return tuple(tables)


_CALLSIGN_ERROR_PATTERNS = _callsign_error_patterns()


def _decode_control(control0: int, control1: int, fec: int) -> tuple[bytes, int] | None:
    first = decode_mobitex_fec((control0 << 4) | (fec >> 4))
    second = decode_mobitex_fec((control1 << 4) | (fec & 0x0F))
    if (
        first.status is MobitexFecStatus.ERROR_UNCORRECTABLE
        or second.status is MobitexFecStatus.ERROR_UNCORRECTABLE
    ):
        return None
    corrected = bytes([first.message, second.message, (first.fec << 4) | second.fec])
    count = int(first.status is MobitexFecStatus.ERROR_CORRECTED) + int(
        second.status is MobitexFecStatus.ERROR_CORRECTED
    )
    return corrected, count


def _hamming(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming operands must have equal length")
    return sum((a ^ b).bit_count() for a, b in zip(left, right, strict=True))


def _recover_callsign(value: bytes, threshold: int) -> tuple[bytes, bytes, int] | None:
    if len(value) != 8:
        raise ValueError("Mobitex callsign field must contain eight bytes")
    if not 0 <= threshold <= DEFAULT_CALLSIGN_THRESHOLD:
        raise ValueError("unknown Mobitex callsign recovery is bounded to two bit flips")
    positions = _CALLSIGN_ERROR_PATTERNS[threshold].get(_callsign_syndrome(value))
    if positions is None:
        return None
    candidate = bytearray(value)
    for position in positions:
        candidate[position // 8] ^= 1 << (position % 8)
    callsign = bytes(candidate[:6])
    crc = bytes(candidate[6:])
    if _CALLSIGN_CRC.compute(callsign).to_bytes(2, "big") != crc:
        raise RuntimeError("Mobitex callsign syndrome correction invariant failed")
    return callsign, crc, len(positions)


def _decode_data_block(block: bytes) -> tuple[bytes, int, int]:
    if len(block) != BLOCK_WIRE_BYTES:
        raise ValueError("Mobitex encoded block must contain 30 bytes")
    payload = bytearray()
    corrected = 0
    uncorrectable = 0
    for offset in range(0, len(block), 3):
        for codeword in unpack_mobitex_pair(block[offset : offset + 3]):
            result = decode_mobitex_fec(codeword)
            payload.append(result.message)
            corrected += int(result.status is MobitexFecStatus.ERROR_CORRECTED)
            uncorrectable += int(result.status is MobitexFecStatus.ERROR_UNCORRECTABLE)
    return bytes(payload), corrected, uncorrectable


def _decoder(
    *, nx: bool, variant: str, callsign: str, callsign_threshold: int
):
    if variant not in VALID_VARIANTS:
        raise ValueError(
            "Mobitex variant must be one of " + ", ".join(sorted(VALID_VARIANTS))
        )
    if not nx and variant != "default":
        raise ValueError("classic Mobitex supports only the default variant")
    header_length = 3 if not nx or variant == "BEESAT-1" else 11
    try:
        expected_callsign = callsign.encode("ascii") if callsign else None
    except UnicodeEncodeError as exc:
        raise ValueError("Mobitex callsign must contain only ASCII") from exc
    if not nx and expected_callsign is not None:
        raise ValueError("callsign applies only to Mobitex-NX")
    if expected_callsign is not None and len(expected_callsign) != 6:
        raise ValueError("Mobitex callsign must encode to exactly six ASCII bytes")
    if expected_callsign is None and callsign_threshold > DEFAULT_CALLSIGN_THRESHOLD:
        raise ValueError("unknown Mobitex callsign recovery is bounded to two bit flips")
    if callsign_threshold < 0 or callsign_threshold > 64:
        raise ValueError("Mobitex callsign threshold must be in 0..64")

    def decode(capture: np.ndarray) -> DecodedFixedFrame | None:
        if capture.ndim != 1 or capture.size < header_length * 8:
            return None
        wire = bytearray(np.packbits(capture > 0, bitorder="big"))
        control = _decode_control(wire[0], wire[1], wire[2])
        if control is None:
            return None
        wire[:3], control_errors = control
        num_blocks = (
            32
            if variant == "BEESAT-9"
            else wire[1]
            if not nx
            else (wire[0] & 0x1F) + 1
        )
        if not 1 <= num_blocks <= MAX_BLOCKS:
            return None
        expected_bytes = header_length + BLOCK_WIRE_BYTES * num_blocks
        if len(wire) != expected_bytes:
            return None

        callsign_errors: int | None = None
        if nx and variant != "BEESAT-1":
            received = bytes(wire[3:11])
            if expected_callsign is not None:
                expected_crc = _CALLSIGN_CRC.compute(expected_callsign).to_bytes(2, "big")
                callsign_errors = _hamming(received, expected_callsign + expected_crc)
                if callsign_errors > callsign_threshold:
                    return None
                wire[3:11] = expected_callsign + expected_crc
            else:
                recovered = _recover_callsign(received, callsign_threshold)
                if recovered is None:
                    return None
                recovered_callsign, recovered_crc, callsign_errors = recovered
                try:
                    recovered_callsign.decode("ascii")
                except UnicodeDecodeError:
                    return None
                wire[3:11] = recovered_callsign + recovered_crc

        end = header_length + BLOCK_WIRE_BYTES * num_blocks
        selected = bytes(wire[header_length:end])
        if len(selected) != BLOCK_WIRE_BYTES * num_blocks:
            return None
        bits = np.unpackbits(np.frombuffer(selected, dtype=np.uint8))
        randomized = additive_randomize_bits(
            bits, mask=0x22, seed=0x1FF, register_length=9
        )
        permuted = np.concatenate(
            [randomized[start : start + 240][_INTERLEAVE] for start in range(0, bits.size, 240)]
        )
        fec_wire = bytes(np.packbits(permuted, bitorder="big"))

        bodies: list[bytes] = []
        invalid_mask = 0
        corrected_total = control_errors
        uncorrectable_total = 0
        valid_blocks = 0
        for block_id in range(num_blocks):
            block = fec_wire[
                block_id * BLOCK_WIRE_BYTES : (block_id + 1) * BLOCK_WIRE_BYTES
            ]
            decoded, corrected, uncorrectable = _decode_data_block(block)
            corrected_total += corrected
            uncorrectable_total += uncorrectable
            body = CRC16_X25.strip_if_valid(decoded, byteorder="big")
            if body is None:
                invalid_mask |= 1 << block_id
                body = decoded[:-2]
            else:
                valid_blocks += 1
            bodies.append(body)
        if valid_blocks == 0:
            return None

        header = bytes(wire[:2])
        if nx and variant != "BEESAT-1":
            header += bytes(wire[3:11])
        payload = (
            header
            + b"".join(bodies)
            + b"\xAA"
            + invalid_mask.to_bytes(4, "little")
            + b"\xBB"
        )
        return DecodedFixedFrame(
            payload=payload,
            integrity=(
                IntegrityStatus.PASSED if invalid_mask == 0 else IntegrityStatus.FAILED
            ),
            corrected_symbols=corrected_total,
            metadata={
                "variant": variant,
                "num_blocks": num_blocks,
                "valid_blocks": valid_blocks,
                "invalid_block_mask": invalid_mask,
                "control_errors_corrected": control_errors,
                "callsign_bit_errors": callsign_errors,
                "fec_errors_uncorrectable": uncorrectable_total,
            },
        )

    return decode


@dataclass(frozen=True)
class _PendingMobitex:
    sync_start: int
    capture_start: int
    sync_distance: float
    polarity: Polarity


class _MobitexSoftDecoder:
    """Bounded collector whose control header selects the on-air frame length."""

    def __init__(
        self,
        *,
        canonical: str,
        syncword: str,
        sync_threshold: float,
        nx: bool,
        variant: str,
        decode_symbols,
    ) -> None:
        self._canonical = canonical
        self._nx = nx
        self._variant = variant
        self._header_bytes = 3 if not nx or variant == "BEESAT-1" else 11
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
        self._pending: list[_PendingMobitex] = []

    @property
    def retained_symbols(self) -> int:
        return int(self._buffer.size + self._sync.retained_symbols)

    @property
    def max_retained_symbols(self) -> int:
        return (
            self._header_bytes + BLOCK_WIRE_BYTES * MAX_BLOCKS
        ) * 8 + self._syncword_length - 1

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
        waiting: list[_PendingMobitex] = []
        for pending in self._pending:
            needed = self._capture_symbols(pending)
            if needed is None:
                if pending.capture_start + self._header_bytes * 8 > self._total_seen:
                    waiting.append(pending)
                continue
            capture_end = pending.capture_start + needed
            if capture_end > self._total_seen:
                waiting.append(pending)
                continue
            result = self._decode(pending, capture_end)
            if result is not None:
                output.append(result)
        self._pending = waiting
        self._trim_buffer()
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError(f"{self._canonical} decoder retained-symbol bound violated")
        return output

    @staticmethod
    def _pending_from_match(match: SyncMatch) -> _PendingMobitex:
        return _PendingMobitex(
            sync_start=match.source_start,
            capture_start=match.source_end,
            sync_distance=match.distance,
            polarity=match.polarity,
        )

    def _normalized(self, pending: _PendingMobitex, end: int) -> np.ndarray | None:
        start = pending.capture_start - self._buffer_base
        stop = end - self._buffer_base
        if start < 0 or stop > self._buffer.size:
            raise RuntimeError(f"{self._canonical} capture fell outside retained buffer")
        capture = self._buffer[start:stop]
        if pending.polarity is Polarity.INVERTED:
            return -capture
        if pending.polarity is Polarity.AMBIGUOUS:
            return None
        return capture

    def _capture_symbols(self, pending: _PendingMobitex) -> int | None:
        header_end = pending.capture_start + self._header_bytes * 8
        if header_end > self._total_seen:
            return None
        capture = self._normalized(pending, header_end)
        if capture is None:
            return None
        header = bytes(np.packbits(capture > 0, bitorder="big"))
        control = _decode_control(header[0], header[1], header[2])
        if control is None:
            return None
        corrected, _ = control
        num_blocks = (
            32
            if self._variant == "BEESAT-9"
            else corrected[1]
            if not self._nx
            else (corrected[0] & 0x1F) + 1
        )
        if not 1 <= num_blocks <= MAX_BLOCKS:
            return None
        return (self._header_bytes + BLOCK_WIRE_BYTES * num_blocks) * 8

    def _decode(
        self, pending: _PendingMobitex, capture_end: int
    ) -> FrameResult | None:
        capture = self._normalized(pending, capture_end)
        if capture is None:
            return None
        decoded = self._decode_symbols(capture)
        if decoded is None:
            return None
        return FrameResult(
            canonical_framing=self._canonical,
            payload=decoded.payload,
            integrity=decoded.integrity,
            source_start=pending.sync_start,
            source_end=capture_end,
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


def _build(parameters: Mapping[str, object], *, nx: bool) -> _MobitexSoftDecoder:
    variant = str(parameters.get("variant", "default"))
    raw_callsign = parameters.get("callsign", "")
    callsign = "" if raw_callsign is None else str(raw_callsign)
    if not nx and "callsign_threshold" in parameters:
        raise ValueError("callsign_threshold applies only to Mobitex-NX")
    raw_threshold = parameters.get("callsign_threshold")
    threshold = (
        DEFAULT_KNOWN_CALLSIGN_THRESHOLD
        if raw_threshold is None and callsign
        else DEFAULT_CALLSIGN_THRESHOLD
        if raw_threshold is None
        else int(raw_threshold)
    )
    decoder = _decoder(
        nx=nx,
        variant=variant,
        callsign=callsign,
        callsign_threshold=threshold,
    )
    return _MobitexSoftDecoder(
        canonical="mobitex_nx" if nx else "mobitex",
        syncword=NX_SYNCWORD if nx else SYNCWORD,
        sync_threshold=float(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        nx=nx,
        variant=variant,
        decode_symbols=decoder,
    )


def build_mobitex(parameters: Mapping[str, object]) -> _MobitexSoftDecoder:
    return _build(parameters, nx=False)


def build_mobitex_nx(parameters: Mapping[str, object]) -> _MobitexSoftDecoder:
    return _build(parameters, nx=True)


__all__ = [
    "BLOCK_WIRE_BYTES",
    "DEFAULT_CALLSIGN_THRESHOLD",
    "DEFAULT_KNOWN_CALLSIGN_THRESHOLD",
    "DEFAULT_SYNC_THRESHOLD",
    "MAX_BLOCKS",
    "MAX_CAPTURE_BYTES",
    "NX_SYNCWORD",
    "SYNCWORD",
    "VALID_VARIANTS",
    "build_mobitex",
    "build_mobitex_nx",
]

"""Bounded S-NET BCH and CRC primitives.

The bit ordering and the documented buggy CRC compatibility mode are adapted
from gr-satellites at commit ``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np

_EXP_TABLE = (8, 4, 2, 1, 12, 6, 3, 13, 10, 5, 14, 7, 15, 11, 9)
_SUPPORTED_DISTANCES = (3, 5, 7)


@dataclass(frozen=True)
class Bch15DecodeResult:
    bits: np.ndarray
    corrected_bits: int


def _syndromes(bits: np.ndarray, distance: int) -> tuple[int, ...]:
    word = 0
    for bit in bits:
        word = (word << 1) | int(bit)
    output: list[int] = []
    # The narrow-sense BCH roots are alpha**1 through alpha**(d-1).  The
    # pinned gr-satellites helper iterates from zero, but that yields only 16
    # zero-syndrome words for the advertised BCH(15,5,7) code and therefore
    # cannot carry arbitrary five-bit systematic data.  This one-based range
    # matches the protocol description and the stated (15,k,d) dimensions.
    for root in range(1, distance):
        syndrome = 0
        value = word
        for power in range(14, -1, -1):
            if value & 1:
                syndrome ^= _EXP_TABLE[(power * root) % len(_EXP_TABLE)]
            value >>= 1
        output.append(syndrome)
    return tuple(output)


@cache
def _error_table(distance: int) -> dict[tuple[int, ...], tuple[int, ...]]:
    if distance not in _SUPPORTED_DISTANCES:
        raise ValueError("S-NET BCH distance must be 3, 5, or 7")
    table: dict[tuple[int, ...], tuple[int, ...]] = {}
    correction_limit = (distance - 1) // 2
    for count in range(1, correction_limit + 1):
        for positions in combinations(range(15), count):
            error = np.zeros(15, dtype=np.uint8)
            error[list(positions)] = 1
            syndrome = _syndromes(error, distance)
            prior = table.setdefault(syndrome, positions)
            if prior != positions:
                raise RuntimeError("ambiguous S-NET BCH syndrome within correction radius")
    return table


def decode_bch15(
    bits: np.ndarray | Sequence[int], *, distance: int = 7
) -> Bch15DecodeResult | None:
    """Decode a systematic BCH(15, k, d) word and report corrected bits."""

    received = np.asarray(bits)
    if received.shape != (15,):
        raise ValueError("S-NET BCH codewords contain exactly 15 bits")
    if not np.all((received == 0) | (received == 1)):
        raise ValueError("S-NET BCH codewords may contain only 0 and 1")
    corrected = received.astype(np.uint8, copy=True)
    syndrome = _syndromes(corrected, distance)
    if not any(syndrome):
        return Bch15DecodeResult(corrected, 0)
    positions = _error_table(distance).get(syndrome)
    if positions is None:
        return None
    corrected[list(positions)] ^= 1
    if any(_syndromes(corrected, distance)):
        return None
    return Bch15DecodeResult(corrected, len(positions))


def snet_crc5(header_without_crc: np.ndarray, *, buggy: bool) -> int:
    """Compute the S-NET header CRC5, including the upstream compatibility bugs."""

    bits = np.asarray(header_without_crc)
    if bits.shape != (65,) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("S-NET CRC5 input must contain exactly 65 bits")
    padded = np.concatenate((bits.astype(np.uint8), [1, 0, 1, 1, 0, 1, 1]))
    rows = padded.reshape(9, 8)
    if buggy:
        rows = np.flipud(rows).copy()
        rows[4, :] = rows[3, :]
    crc = 0x1F
    for bit in rows.ravel():
        top = (crc >> 4) & 1
        crc = (crc << 1) & 0x1F
        if top != int(bit):
            crc ^= 0x15
    return crc


def snet_crc13(payload: bytes, *, buggy: bool) -> int:
    """Compute the S-NET payload CRC13 in normal or historical-bug mode."""

    rows = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape((-1, 8))
    if buggy:
        rows = np.flipud(rows)
    crc = 0x1FFF
    for bit in rows.ravel():
        top = (crc >> 12) & 1
        crc = (crc << 1) & 0x1FFF
        if (top or int(bit)) if buggy else (top != int(bit)):
            crc ^= 0x1CF5
    return crc


__all__ = ["Bch15DecodeResult", "decode_bch15", "snet_crc5", "snet_crc13"]

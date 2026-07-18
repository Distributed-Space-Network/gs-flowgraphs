"""Mobitex (12,8,3) systematic linear code.

Copyright 2025 Fabian P. Schmidt <kerel@mailbox.org>
Adapted for gs-flowgraphs in 2026 from gr-satellites at the pinned commit.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

_MATRIX_8 = (0b11101100, 0b11010011, 0b10111010, 0b01110101)
_MATRIX_12 = (
    0b111011001000,
    0b110100110100,
    0b101110100010,
    0b011101010001,
)


class MobitexFecStatus(IntEnum):
    NO_ERROR = 0
    ERROR_CORRECTED = 1
    ERROR_UNCORRECTABLE = 2


def _syndrome(word: int) -> int:
    return sum(
        (((word & mask).bit_count() & 1) << (3 - index))
        for index, mask in enumerate(_MATRIX_12)
    )


_SYNDROME_TABLE = {_syndrome(1 << position): position for position in range(12)}


@dataclass(frozen=True)
class MobitexFecResult:
    message: int
    fec: int
    status: MobitexFecStatus


def encode_mobitex_fec(message: int) -> int:
    if not 0 <= message <= 0xFF:
        raise ValueError("Mobitex FEC message must be an octet")
    fec = sum(
        (((message & mask).bit_count() & 1) << (3 - index))
        for index, mask in enumerate(_MATRIX_8)
    )
    return (message << 4) | fec


def decode_mobitex_fec(codeword: int) -> MobitexFecResult:
    if not 0 <= codeword <= 0xFFF:
        raise ValueError("Mobitex FEC codeword must be 12 bits")
    syndrome = _syndrome(codeword)
    status = MobitexFecStatus.NO_ERROR
    if syndrome:
        position = _SYNDROME_TABLE.get(syndrome)
        if position is None:
            return MobitexFecResult(
                codeword >> 4,
                codeword & 0x0F,
                MobitexFecStatus.ERROR_UNCORRECTABLE,
            )
        codeword ^= 1 << position
        status = MobitexFecStatus.ERROR_CORRECTED
    return MobitexFecResult(codeword >> 4, codeword & 0x0F, status)


def unpack_mobitex_pair(code: bytes) -> tuple[int, int]:
    if len(code) != 3:
        raise ValueError("two Mobitex codewords occupy exactly three bytes")
    return (code[0] << 4) | (code[1] >> 4), ((code[1] & 0x0F) << 8) | code[2]


__all__ = [
    "MobitexFecResult",
    "MobitexFecStatus",
    "decode_mobitex_fec",
    "encode_mobitex_fec",
    "unpack_mobitex_pair",
]

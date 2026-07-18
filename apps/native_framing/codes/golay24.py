"""Extended binary Golay (24,12,8) encoder and bounded-distance decoder.

This is a direct Python implementation of the algorithm and parity-check matrix in gr-satellites
``lib/golay24.c`` at commit ``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2017 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass

_N = 12
_H = (
    0x8008ED,
    0x4001DB,
    0x2003B5,
    0x100769,
    0x080ED1,
    0x040DA3,
    0x020B47,
    0x01068F,
    0x008D1D,
    0x004A3B,
    0x002477,
    0x001FFE,
)
_B = tuple(value & 0xFFF for value in _H)


@dataclass(frozen=True)
class Golay24Result:
    data: int
    codeword: int
    corrected_bits: int


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _syndrome(word: int) -> int:
    value = 0
    for row in _H:
        value = (value << 1) | _parity(row & word)
    return value


def encode_golay24(data: int) -> int:
    if isinstance(data, bool) or not isinstance(data, int) or not 0 <= data <= 0xFFF:
        raise ValueError("Golay data must be a 12-bit integer")
    return (_syndrome(data) << _N) | data


def decode_golay24(codeword: int) -> Golay24Result | None:
    if (
        isinstance(codeword, bool)
        or not isinstance(codeword, int)
        or not 0 <= codeword <= 0xFFFFFF
    ):
        raise ValueError("Golay codeword must be a 24-bit integer")

    syndrome = _syndrome(codeword)
    error: int | None = None
    if syndrome.bit_count() <= 3:
        error = syndrome << _N
    else:
        for index, row in enumerate(_B):
            candidate = syndrome ^ row
            if candidate.bit_count() <= 2:
                error = (candidate << _N) | (1 << (_N - index - 1))
                break

    if error is None:
        modified = 0
        for row in _B:
            modified = (modified << 1) | _parity(row & syndrome)
        if modified.bit_count() <= 3:
            error = modified
        else:
            for index, row in enumerate(_B):
                candidate = modified ^ row
                if candidate.bit_count() <= 2:
                    error = (1 << (2 * _N - index - 1)) | candidate
                    break

    if error is None:
        return None
    corrected = codeword ^ error
    return Golay24Result(
        data=corrected & 0xFFF,
        codeword=corrected,
        corrected_bits=error.bit_count(),
    )


__all__ = ["Golay24Result", "decode_golay24", "encode_golay24"]

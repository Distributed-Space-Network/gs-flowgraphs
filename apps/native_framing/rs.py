"""Conventional and CCSDS dual-basis RS(255,223) with shortening/interleaving.

The CCSDS field/code parameters and conventional/dual basis map come from the
pinned gr-satellites libfec sources at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``. The table itself originates
from Phil Karn's libfec implementation.

Copyright 2002-2004 Phil Karn, KA9Q
SPDX-License-Identifier: LGPL-2.1-or-later
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from gfsk_ax25.reedsolomon import RSCodec

# Conventional -> CCSDS dual basis (libfec Taltab). The inverse table is
# derived and checked at import, avoiding two independently editable literals.
_TALTAB = bytes.fromhex(
    "00 7b af d4 99 e2 36 4d fa 81 55 2e 63 18 cc b7"
    " 86 fd 29 52 1f 64 b0 cb 7c 07 d3 a8 e5 9e 4a 31"
    " ec 97 43 38 75 0e da a1 16 6d b9 c2 8f f4 20 5b"
    " 6a 11 c5 be f3 88 5c 27 90 eb 3f 44 09 72 a6 dd"
    " ef 94 40 3b 76 0d d9 a2 15 6e ba c1 8c f7 23 58"
    " 69 12 c6 bd f0 8b 5f 24 93 e8 3c 47 0a 71 a5 de"
    " 03 78 ac d7 9a e1 35 4e f9 82 56 2d 60 1b cf b4"
    " 85 fe 2a 51 1c 67 b3 c8 7f 04 d0 ab e6 9d 49 32"
    " 8d f6 22 59 14 6f bb c0 77 0c d8 a3 ee 95 41 3a"
    " 0b 70 a4 df 92 e9 3d 46 f1 8a 5e 25 68 13 c7 bc"
    " 61 1a ce b5 f8 83 57 2c 9b e0 34 4f 02 79 ad d6"
    " e7 9c 48 33 7e 05 d1 aa 1d 66 b2 c9 84 ff 2b 50"
    " 62 19 cd b6 fb 80 54 2f 98 e3 37 4c 01 7a ae d5"
    " e4 9f 4b 30 7d 06 d2 a9 1e 65 b1 ca 87 fc 28 53"
    " 8e f5 21 5a 17 6c b8 c3 74 0f db a0 ed 96 42 39"
    " 08 73 a7 dc 91 ea 3e 45 f2 89 5d 26 6b 10 c4 bf"
)


def _inverse_table(table: bytes) -> bytes:
    if len(table) != 256 or len(set(table)) != 256:
        raise ValueError("basis table must be a 256-byte permutation")
    inverse = bytearray(256)
    for source, target in enumerate(table):
        inverse[target] = source
    return bytes(inverse)


_TAL1TAB = _inverse_table(_TALTAB)

# libfec fixed.h: GF polynomial 0x187, FCR=112, PRIM=11. RSCodec accepts a
# field element as its generator; alpha^11 is 0xad in this field.
_CODEC = RSCodec(32, prim=0x187, fcr=112, generator=0xAD)


@dataclass(frozen=True)
class ReedSolomonResult:
    payload: bytes
    corrected_symbols: int


class CcsdsReedSolomon:
    def __init__(self, *, basis: str = "dual", interleaving: int = 1) -> None:
        if basis not in ("conventional", "dual"):
            raise ValueError("basis must be 'conventional' or 'dual'")
        if interleaving <= 0:
            raise ValueError("interleaving must be positive")
        self.basis = basis
        self.interleaving = int(interleaving)

    def encode(self, payload: bytes) -> bytes:
        if not payload or len(payload) % self.interleaving:
            raise ValueError("payload length must be non-zero and divisible by interleaving")
        path_size = len(payload) // self.interleaving
        if path_size > 223:
            raise ValueError("interleaved RS path exceeds 223 data symbols")
        encoded_paths: list[bytes] = []
        for path in range(self.interleaving):
            data = bytes(payload[path:: self.interleaving])
            conventional = self._to_conventional(data)
            codeword = _CODEC.encode(conventional)
            encoded_paths.append(self._from_conventional(codeword))
        return self._interleave(encoded_paths)

    def decode(
        self,
        codeword: bytes,
        *,
        erase_pos: Iterable[int] | None = None,
    ) -> ReedSolomonResult | None:
        """Decode one shortened/interleaved codeword.

        ``erase_pos`` contains zero-based positions in the complete wire-order
        codeword.  Positions are mapped to their individual RS paths before
        decoding, so every path independently enforces
        ``2 * unknown_errors + erasures <= 32``.
        """

        if not codeword or len(codeword) % self.interleaving:
            return None
        path_size = len(codeword) // self.interleaving
        if path_size <= 32 or path_size > 255:
            return None
        erasures = self._validate_erasure_positions(erase_pos, len(codeword))
        path_erasures: list[list[int]] = [[] for _ in range(self.interleaving)]
        for position in erasures:
            path_erasures[position % self.interleaving].append(
                position // self.interleaving
            )
        decoded_paths: list[bytes] = []
        corrected = 0
        for path in range(self.interleaving):
            wire_path = bytes(codeword[path:: self.interleaving])
            result = _CODEC.decode_with_count(
                self._to_conventional(wire_path),
                erase_pos=path_erasures[path],
            )
            if result is None:
                return None
            payload, path_corrected = result
            decoded_paths.append(self._from_conventional(payload))
            corrected += path_corrected
        return ReedSolomonResult(self._interleave(decoded_paths), corrected)

    @staticmethod
    def _validate_erasure_positions(
        erase_pos: Iterable[int] | None,
        codeword_size: int,
    ) -> tuple[int, ...]:
        if erase_pos is None:
            return ()
        try:
            supplied = tuple(erase_pos)
        except TypeError as exc:
            raise TypeError("erase_pos must be an iterable of integer positions") from exc
        positions: set[int] = set()
        for position in supplied:
            if isinstance(position, bool) or not isinstance(position, Integral):
                raise TypeError("erasure positions must be integers")
            value = int(position)
            if value < 0 or value >= codeword_size:
                raise ValueError("erasure position out of range")
            positions.add(value)
        return tuple(sorted(positions))

    def _to_conventional(self, data: bytes) -> bytes:
        return bytes(_TAL1TAB[value] for value in data) if self.basis == "dual" else data

    def _from_conventional(self, data: bytes) -> bytes:
        return bytes(_TALTAB[value] for value in data) if self.basis == "dual" else data

    def _interleave(self, paths: list[bytes]) -> bytes:
        if not paths or len({len(path) for path in paths}) != 1:
            raise ValueError("RS paths must have equal lengths")
        return bytes(value for column in zip(*paths, strict=True) for value in column)


def ccsds_generator_log_coefficients() -> tuple[int, ...]:
    """Return the generated polynomial in libfec's logarithmic representation."""

    return tuple(_CODEC._log[value] if value else 255 for value in _CODEC._gpoly)


__all__ = ["CcsdsReedSolomon", "ReedSolomonResult", "ccsds_generator_log_coefficients"]

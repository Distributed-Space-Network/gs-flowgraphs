"""Reusable NGHam size-tag, padding, and Reed-Solomon primitives.

Constants and receive semantics follow the pinned gr-satellites
``ngham_packet_crop.py`` and ``ngham_remove_padding.py`` blocks.
The RS parameters and official encoded vectors come from Jon Petter Skagmo's
NGHam reference implementation at commit
``29c4fd393049ac3483d9ffa034e867361d0f1764``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass

from gfsk_ax25.reedsolomon import RSCodec

SIZE_TAGS = (
    0b001110110100100111001101,
    0b010011011101101001010111,
    0b011101101001001110011010,
    0b100110111011010010101110,
    0b101000001111110101100011,
    0b110101100110111011111001,
    0b111011010010011100110100,
)
RS_SIZES = (47, 79, 111, 159, 191, 223, 255)
NON_RS_SIZES = (31, 63, 95, 127, 159, 191, 223)
PARITY_SIZES = (16, 16, 16, 32, 32, 32, 32)
TAG_CORRECTION_LIMIT = 6

# NGHam reference implementation: MM=8, GF polynomial 0x187, FCR=112,
# primitive element step 11, and either 16 or 32 roots. RSCodec takes the
# field element alpha**11 (0xad) rather than the exponent itself.
_RS_CODECS = {
    parity: RSCodec(parity, prim=0x187, fcr=112, generator=0xAD)
    for parity in set(PARITY_SIZES)
}


@dataclass(frozen=True)
class NGHamSize:
    index: int
    tag_distance: int
    rs_size: int
    non_rs_size: int


@dataclass(frozen=True)
class NGHamReedSolomonResult:
    payload: bytes
    corrected_symbols: int


def classify_ngham_size(
    tag: bytes, *, max_distance: int = TAG_CORRECTION_LIMIT
) -> NGHamSize | None:
    """Return the uniquely nearest NGHam size tag within its six-bit radius."""

    if len(tag) != 3:
        raise ValueError("NGHam size tags are exactly three bytes")
    if max_distance < 0 or max_distance > TAG_CORRECTION_LIMIT:
        raise ValueError("NGHam size-tag threshold must be between 0 and 6")
    received = int.from_bytes(tag, "big")
    distances = tuple((received ^ candidate).bit_count() for candidate in SIZE_TAGS)
    best_distance = min(distances)
    best = tuple(index for index, distance in enumerate(distances) if distance == best_distance)
    if best_distance > max_distance or len(best) != 1:
        return None
    index = best[0]
    return NGHamSize(index, best_distance, RS_SIZES[index], NON_RS_SIZES[index])


def remove_ngham_padding(packet: bytes, size: NGHamSize) -> tuple[bytes, int] | None:
    """Remove parity slots and the bounded padding declared in byte zero."""

    if len(packet) == size.rs_size:
        packet = packet[: size.non_rs_size]
    elif len(packet) != size.non_rs_size:
        return None
    if not packet:
        return None
    padding = packet[0] & 0x1F
    if padding >= len(packet):
        return None
    return (packet[:-padding] if padding else packet), padding


def encode_ngham_rs(packet: bytes, size: NGHamSize) -> bytes:
    """Encode one full, already-padded NGHam data block for test/TX parity."""

    if len(packet) != size.non_rs_size:
        raise ValueError("NGHam RS input must match the selected data-block size")
    parity = PARITY_SIZES[size.index]
    return _RS_CODECS[parity].encode(packet)


def decode_ngham_rs(
    codeword: bytes, size: NGHamSize
) -> NGHamReedSolomonResult | None:
    """Decode a shortened NGHam RS16/RS32 codeword."""

    if len(codeword) != size.rs_size:
        return None
    parity = PARITY_SIZES[size.index]
    result = _RS_CODECS[parity].decode_with_count(codeword)
    if result is None:
        return None
    payload, corrected_symbols = result
    if len(payload) != size.non_rs_size:
        return None
    return NGHamReedSolomonResult(payload, corrected_symbols)


__all__ = [
    "NGHamSize",
    "NGHamReedSolomonResult",
    "NON_RS_SIZES",
    "PARITY_SIZES",
    "RS_SIZES",
    "SIZE_TAGS",
    "TAG_CORRECTION_LIMIT",
    "classify_ngham_size",
    "decode_ngham_rs",
    "encode_ngham_rs",
    "remove_ngham_padding",
]

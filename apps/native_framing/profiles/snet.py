"""Bounded native S-NET receive deframer.

Adapted from gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from native_framing.codes.snet import decode_bch15, snet_crc5, snet_crc13
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.types import IntegrityStatus

SYNCWORD = "00000100110011110101111111001000"
CAPTURE_SIZE = 512
DEFAULT_SYNC_THRESHOLD = 4
_HEADER_BITS = 210
_CODEWORDS_PER_BLOCK = 16
_AI_MODES = {
    1: (11, 3),
    2: (7, 5),
    3: (5, 7),
}


def _integer(bits: np.ndarray, start: int, width: int) -> int:
    return int.from_bytes(np.packbits(bits[start : start + width]), "big") >> (
        8 - width % 8 if width % 8 else 0
    )


def _decode_header(bits: np.ndarray, *, buggy_crc: bool):
    codewords = bits[:_HEADER_BITS].reshape((15, 14)).transpose().copy()
    corrected = 0
    for index in range(codewords.shape[0]):
        result = decode_bch15(codewords[index], distance=7)
        if result is None:
            return None
        codewords[index] = result.bits
        corrected += result.corrected_bits
    header = np.fliplr(codewords[:, -5:]).ravel()
    if snet_crc5(header[:-5], buggy=buggy_crc) != _integer(header, 65, 5):
        return None
    return (
        {
            "src_id": _integer(header, 0, 7),
            "dst_id": _integer(header, 7, 7),
            "ai_type_src": _integer(header, 26, 4),
            "pdu_length": _integer(header, 42, 10),
            "crc13": _integer(header, 52, 13),
        },
        corrected,
    )


def _decode_payload(
    bits: np.ndarray, *, ai_type: int, pdu_length: int
) -> tuple[bytes, int] | None:
    if pdu_length <= 0:
        return None
    if ai_type == 0:
        end = _HEADER_BITS + pdu_length * 8
        if end > bits.size:
            return None
        rows = bits[_HEADER_BITS:end].reshape((pdu_length, 8))
        return bytes(np.packbits(np.fliplr(rows))), 0
    mode = _AI_MODES.get(ai_type)
    if mode is None:
        return None
    data_bits_per_codeword, distance = mode
    data_bytes_per_block = _CODEWORDS_PER_BLOCK * data_bits_per_codeword // 8
    blocks = (pdu_length + data_bytes_per_block - 1) // data_bytes_per_block
    end = _HEADER_BITS + blocks * _CODEWORDS_PER_BLOCK * 15
    if end > bits.size:
        return None

    decoded_blocks: list[np.ndarray] = []
    corrected = 0
    for block_index in range(blocks):
        start = _HEADER_BITS + block_index * _CODEWORDS_PER_BLOCK * 15
        codewords = bits[start : start + 240].reshape((15, 16)).transpose().copy()
        for index in range(codewords.shape[0]):
            result = decode_bch15(codewords[index], distance=distance)
            if result is None:
                return None
            codewords[index] = result.bits
            corrected += result.corrected_bits
        decoded_blocks.append(codewords[:, -data_bits_per_codeword:].ravel())
    data = np.concatenate(decoded_blocks).reshape((-1, 8))
    payload = bytes(np.packbits(np.fliplr(data)))[:pdu_length]
    return payload, corrected


def _decoder(*, buggy_crc: bool):
    def decode(wire: bytes) -> DecodedFixedFrame | None:
        if len(wire) != CAPTURE_SIZE:
            return None
        bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8))
        header_result = _decode_header(bits, buggy_crc=buggy_crc)
        if header_result is None:
            return None
        header, header_corrections = header_result
        payload_result = _decode_payload(
            bits,
            ai_type=int(header["ai_type_src"]),
            pdu_length=int(header["pdu_length"]),
        )
        if payload_result is None:
            return None
        payload, payload_corrections = payload_result
        if snet_crc13(payload, buggy=buggy_crc) != header["crc13"]:
            return None
        return DecodedFixedFrame(
            payload=payload,
            integrity=IntegrityStatus.PASSED,
            corrected_symbols=header_corrections + payload_corrections,
            metadata={
                **header,
                "buggy_crc": buggy_crc,
                "header_corrected_bits": header_corrections,
                "payload_corrected_bits": payload_corrections,
                "crc5_algorithm": "S-NET buggy" if buggy_crc else "S-NET CRC-5",
                "crc13_algorithm": "S-NET buggy" if buggy_crc else "S-NET CRC-13",
            },
        )

    return decode


def build_snet(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="snet",
        syncword=SYNCWORD,
        frame_size=CAPTURE_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decoder(buggy_crc=bool(parameters.get("buggy_crc", False))),
    )


__all__ = [
    "CAPTURE_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "SYNCWORD",
    "build_snet",
]

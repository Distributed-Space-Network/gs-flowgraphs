"""Native Unified SPUTNIX Protocol (USP) decoder.

The chain and PLS vectors follow the pinned gr-satellites USP blocks and QA at
commit ``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

# Copyright 2021 Daniel Estevez <daniel@destevez.net>
# Adapted for the gs-flowgraphs bounded native streaming API in 2026.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from native_framing.fixed import DecodedFixedFrame, FixedSoftSyncFrameDecoder
from native_framing.linecode import ccsds_randomize
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import IntegrityStatus
from native_framing.viterbi import ConvolutionalCode

SYNCWORD = "0101000001110010111101100100101100101101100100001011000111110101"
CAPTURE_SYMBOLS = 4144
DEFAULT_SYNC_THRESHOLD = 13

_SCRAMBLED_PLS_BITS = (
    "0111000110011101100000111100100101010011010000100010110111111010",
    "0010010011001000110101101001110000000110000101110111100010101111",
)


def _pls_vectors() -> np.ndarray:
    bits = np.asarray(
        [[int(bit) for bit in vector] for vector in _SCRAMBLED_PLS_BITS],
        dtype=np.uint8,
    ).T
    if bits.shape != (64, 2):
        raise AssertionError("USP must define two 64-bit scrambled PLS vectors")
    return 2.0 * bits.astype(np.float64) - 1.0


_SCRAMBLED_PLS = _pls_vectors()
_VITERBI = ConvolutionalCode("CCSDS")
_RS = CcsdsReedSolomon(basis="dual", interleaving=1)


def _decode_capture(capture: np.ndarray) -> DecodedFixedFrame | None:
    if capture.shape != (CAPTURE_SYMBOLS,) or not np.all(np.isfinite(capture)):
        return None
    correlations = capture[:64] @ _SCRAMBLED_PLS
    if not np.all(np.isfinite(correlations)) or correlations[0] == correlations[1]:
        return None
    pls_code = int(np.argmax(correlations))
    # The pinned test IQ establishes this mapping; rev. 1.04 lists it reversed.
    data_length = 48 if pls_code == 0 else 223
    coded_symbols = 2 * 8 * (data_length + 32)
    soft = capture[64 : 64 + coded_symbols]
    if soft.size != coded_symbols:
        return None
    try:
        viterbi = _VITERBI.decode_soft(soft, mode="truncated")
    except ValueError:
        return None
    decoded = bytes(
        np.packbits(np.asarray(viterbi.bits, dtype=np.uint8), bitorder="big")
    )
    derandomized = ccsds_randomize(decoded)
    rs = _RS.decode(derandomized)
    if rs is None or len(rs.payload) != data_length:
        return None
    if len(rs.payload) < 4:
        return None
    ax25_length = int.from_bytes(rs.payload[2:4], "little")
    if ax25_length > len(rs.payload) - 4:
        return None
    payload = rs.payload[4 : 4 + ax25_length]
    return DecodedFixedFrame(
        payload=payload,
        integrity=IntegrityStatus.PASSED,
        corrected_symbols=rs.corrected_symbols,
        metadata={
            "pls_code": pls_code,
            "data_length": data_length,
            "viterbi_convention": "CCSDS",
            "viterbi_mode": "truncated",
            "viterbi_metric": viterbi.metric,
            "randomizer": "CCSDS",
            "rs_basis": "dual",
            "rs_parity_symbols": 32,
            "ax25_length": ax25_length,
        },
    )


def build_usp(parameters: Mapping[str, object]) -> FixedSoftSyncFrameDecoder:
    return FixedSoftSyncFrameDecoder(
        canonical="usp",
        syncword=SYNCWORD,
        capture_symbols=CAPTURE_SYMBOLS,
        sync_threshold=float(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_symbols=_decode_capture,
    )


__all__ = [
    "CAPTURE_SYMBOLS",
    "DEFAULT_SYNC_THRESHOLD",
    "SYNCWORD",
    "build_usp",
]

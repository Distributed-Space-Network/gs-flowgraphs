"""Native GOMspace U482C decoder with header-controlled coding stages.

Stage order and length semantics follow gr-satellites ``u482c_deframer.py`` and
``u482c_decode_impl.cc`` at commit ``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from native_framing.codes.golay24 import decode_golay24
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import ccsds_randomize
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import IntegrityStatus
from native_framing.viterbi import ConvolutionalCode

SYNCWORD = "11000011101010100110011001010101"
CAPTURE_SIZE = 258
DEFAULT_SYNC_THRESHOLD = 4

_RS = CcsdsReedSolomon(basis="conventional", interleaving=1)
_VITERBI = ConvolutionalCode("NASA-DSN uninverted")


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    if len(wire) != CAPTURE_SIZE:
        return None
    header = decode_golay24(int.from_bytes(wire[:3], "big"))
    if header is None:
        return None
    frame_length = header.data & 0xFF
    viterbi = bool(header.data & 0x100)
    randomize = bool(header.data & 0x200)
    rs_enabled = bool(header.data & 0x400)
    if frame_length <= 0:
        return None
    packet = wire[3 : 3 + frame_length]
    if len(packet) != frame_length:
        return None

    viterbi_metric: float | None = None
    if viterbi:
        decoded_length = frame_length // 2 - 1
        if decoded_length <= 0:
            return None
        pair_count = decoded_length * 8 + 6
        encoded_bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))[: 2 * pair_count]
        if encoded_bits.size != 2 * pair_count:
            return None
        try:
            result = _VITERBI.decode_hard(encoded_bits, mode="terminated")
        except ValueError:
            return None
        if len(result.bits) != decoded_length * 8:
            return None
        packet = bytes(np.packbits(np.asarray(result.bits, dtype=np.uint8), bitorder="big"))
        viterbi_metric = result.metric

    if randomize:
        packet = ccsds_randomize(packet)

    corrected_symbols: int | None = None
    integrity = IntegrityStatus.NOT_PRESENT
    if rs_enabled:
        decoded = _RS.decode(packet)
        if decoded is None:
            return None
        packet = decoded.payload
        corrected_symbols = decoded.corrected_symbols
        integrity = IntegrityStatus.PASSED

    return DecodedFixedFrame(
        payload=packet,
        integrity=integrity,
        corrected_symbols=corrected_symbols,
        metadata={
            "golay_corrected_bits": header.corrected_bits,
            "declared_length": frame_length,
            "viterbi": viterbi,
            "viterbi_convention": "NASA-DSN uninverted" if viterbi else "none",
            "viterbi_metric": viterbi_metric,
            "randomizer": "CCSDS" if randomize else "none",
            "rs": rs_enabled,
            "rs_basis": "conventional" if rs_enabled else "none",
            "rs_parity_symbols": 32 if rs_enabled else 0,
            "false_positive_policy": (
                "RS integrity gate" if rs_enabled else "explicit-profile only; no integrity field"
            ),
        },
    )


def build_u482c(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return FixedSyncFrameDecoder(
        canonical="u482c",
        syncword=SYNCWORD,
        frame_size=CAPTURE_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=_decode_wire,
    )


__all__ = ["CAPTURE_SIZE", "DEFAULT_SYNC_THRESHOLD", "SYNCWORD", "build_u482c"]

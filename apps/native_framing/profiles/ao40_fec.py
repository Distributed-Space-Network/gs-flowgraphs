"""Native AO-40 distributed-sync convolutional/RS receive profiles.

The matrix dimensions, convolutional convention, randomizer, and RS layout
follow gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from native_framing.crc import CRC16_ARC
from native_framing.fixed import DecodedFixedFrame, DistributedSoftFrameDecoder
from native_framing.linecode import ccsds_randomize
from native_framing.rs import CcsdsReedSolomon
from native_framing.viterbi import ConvolutionalCode

SYNCWORD = "11111110000111011110010110010010000001000100110001011101011011000"
SHORT_SYNCWORD = "1111111000011101111001011001001000000100010011000101"
DEFAULT_SYNC_THRESHOLD = 8

_VITERBI = ConvolutionalCode("CCSDS")


def _decode_capture(*, short: bool, crc_enabled: bool):
    rows = 51 if short else 80
    columns = 52 if short else 65
    output_size = 2572 if short else 5132
    output_skip = 80 if short else 65
    interleaving = 1 if short else 2
    rs = CcsdsReedSolomon(basis="conventional", interleaving=interleaving)

    def decode(capture: np.ndarray) -> DecodedFixedFrame | None:
        if capture.shape != (rows * columns,) or not np.all(np.isfinite(capture)):
            return None
        deinterleaved = capture.reshape((columns, rows)).T.ravel()
        soft = deinterleaved[output_skip : output_skip + output_size]
        if soft.size != output_size:
            return None
        try:
            viterbi = _VITERBI.decode_soft(soft, mode="terminated")
        except ValueError:
            return None
        decoded = bytes(
            np.packbits(np.asarray(viterbi.bits, dtype=np.uint8), bitorder="big")
        )
        derandomized = ccsds_randomize(decoded)
        rs_result = rs.decode(derandomized)
        if rs_result is None:
            return None
        payload = rs_result.payload
        if crc_enabled and CRC16_ARC.strip_if_valid(payload, byteorder="little") is None:
            return None
        return DecodedFixedFrame(
            payload=payload,
            corrected_symbols=rs_result.corrected_symbols,
            metadata={
                "short_frames": short,
                "matrix_rows": rows,
                "matrix_columns": columns,
                "matrix_output_skip": output_skip,
                "viterbi_convention": "CCSDS",
                "viterbi_mode": "terminated",
                "viterbi_metric": viterbi.metric,
                "randomizer": "CCSDS",
                "rs_basis": "conventional",
                "rs_interleaving": interleaving,
                "crc": CRC16_ARC.name if crc_enabled else "none",
                "crc_preserved": crc_enabled,
            },
        )

    return decode


def _build(
    canonical: str, parameters: Mapping[str, object], *, short: bool
) -> DistributedSoftFrameDecoder:
    return DistributedSoftFrameDecoder(
        canonical=canonical,
        syncword=SHORT_SYNCWORD if short else SYNCWORD,
        step=51 if short else 80,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_symbols=_decode_capture(
            short=short, crc_enabled=bool(parameters.get("crc", False))
        ),
    )


def build_ao40_fec(parameters: Mapping[str, object]) -> DistributedSoftFrameDecoder:
    return _build("ao40_fec", parameters, short=False)


def build_ao40_fec_short(parameters: Mapping[str, object]) -> DistributedSoftFrameDecoder:
    return _build("ao40_fec_short", parameters, short=True)


__all__ = [
    "DEFAULT_SYNC_THRESHOLD",
    "SHORT_SYNCWORD",
    "SYNCWORD",
    "build_ao40_fec",
    "build_ao40_fec_short",
]

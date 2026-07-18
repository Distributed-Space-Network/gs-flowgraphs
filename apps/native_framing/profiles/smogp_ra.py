"""Bounded native SMOG-P repeat-accumulate receive profile.

Adapted from gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from native_framing.codes.ra import (
    DEFAULT_ERROR_THRESHOLD,
    decode_ra_soft,
    ra_config,
)
from native_framing.fixed import DecodedFixedFrame, FixedSoftSyncFrameDecoder
from native_framing.types import IntegrityStatus

SYNCWORD = "0010110111010100"
FRAME_SIZES = (128, 256)
DEFAULT_FRAME_SIZE = 128
DEFAULT_SYNC_THRESHOLD = 0


def _decoder(*, frame_size: int, error_threshold: float):
    def decode(capture: np.ndarray) -> DecodedFixedFrame | None:
        result = decode_ra_soft(
            capture,
            frame_size=frame_size,
            error_threshold=error_threshold,
        )
        if result is None:
            return None
        return DecodedFixedFrame(
            payload=result.payload,
            integrity=IntegrityStatus.NOT_PRESENT,
            metadata={
                "variant": "SMOG-P",
                "frame_size": frame_size,
                "ra_passes": 40,
                "recode_bit_errors": result.recode_bit_errors,
                "recode_error_fraction": result.recode_error_fraction,
                "recode_error_threshold": error_threshold,
            },
        )

    return decode


def build_smogp_ra(parameters: Mapping[str, object]) -> FixedSoftSyncFrameDecoder:
    frame_size = int(parameters.get("frame_size", DEFAULT_FRAME_SIZE))
    if frame_size not in FRAME_SIZES:
        raise ValueError("SMOG-P RA frame_size must be 128 or 256")
    config = ra_config(frame_size)
    error_threshold = float(
        parameters.get("error_threshold", DEFAULT_ERROR_THRESHOLD)
    )
    return FixedSoftSyncFrameDecoder(
        canonical="smogp_ra",
        syncword=SYNCWORD,
        capture_symbols=config.code_length * 16,
        sync_threshold=float(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_symbols=_decoder(
            frame_size=frame_size,
            error_threshold=error_threshold,
        ),
    )


__all__ = [
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "FRAME_SIZES",
    "SYNCWORD",
    "build_smogp_ra",
]

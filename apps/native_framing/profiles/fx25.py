"""Native Astrocast non-compliant FX.25 receive profile.

The chain follows gr-satellites at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.  It is deliberately named
Astrocast-compatible rather than claiming generic FX.25 interoperability.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from native_framing.crc import CRC16_X25
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import reflect_bytes
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import FrameResult, Polarity

SYNCWORD = "0111010111111010110000011010001101011000110100000110010001110110"
CAPTURE_SIZE = 255
DEFAULT_SYNC_THRESHOLD = 8

_RS = CcsdsReedSolomon(basis="dual", interleaving=1)


def _decode_wire(wire: bytes) -> DecodedFixedFrame | None:
    if len(wire) != CAPTURE_SIZE:
        return None
    rs = _RS.decode(reflect_bytes(wire))
    if rs is None or not rs.payload or rs.payload[0] != 0x7E:
        return None
    packet = rs.payload[1:]
    try:
        closing = packet.index(0x7E)
    except ValueError:
        return None
    if closing <= 2:
        return None
    payload = CRC16_X25.strip_if_valid(packet[:closing], byteorder="little")
    if payload is None:
        return None
    return DecodedFixedFrame(
        payload=payload,
        corrected_symbols=rs.corrected_symbols,
        metadata={
            "variant": "Astrocast non-compliant FX.25",
            "byte_reflection": True,
            "rs_basis": "dual",
            "rs_interleaving": 1,
            "opening_flag": "0x7e",
            "closing_flag_offset": closing + 1,
            "crc": CRC16_X25.name,
            "crc_byteorder": "little",
        },
    )


class Fx25StreamingDecoder:
    """Optional streaming NRZI adapter around the bounded FX.25 collector."""

    def __init__(self, *, nrzi: bool, sync_threshold: int) -> None:
        self._nrzi = bool(nrzi)
        self._previous_level = 1
        self._decoder = FixedSyncFrameDecoder(
            canonical="fx25_nrzi",
            syncword=SYNCWORD,
            frame_size=CAPTURE_SIZE,
            sync_threshold=sync_threshold,
            decode_wire=_decode_wire,
        )

    @property
    def retained_symbols(self) -> int:
        return self._decoder.retained_symbols

    @property
    def max_retained_symbols(self) -> int:
        return self._decoder.max_retained_symbols

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        bits = np.asarray(symbols)
        if bits.ndim != 1:
            raise ValueError("hard-bit chunks must be one-dimensional")
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValueError("hard-bit chunks may contain only 0 and 1")
        hard = bits.astype(np.uint8, copy=False)
        if not self._nrzi:
            return self._annotate(self._decoder.push(hard))
        decoded = np.empty_like(hard)
        previous = self._previous_level
        for index, level in enumerate(hard):
            current = int(level)
            decoded[index] = int(current == previous)
            previous = current
        self._previous_level = previous
        return self._annotate(self._decoder.push(decoded))

    def _annotate(self, frames: list[FrameResult]) -> list[FrameResult]:
        output = []
        for frame in frames:
            metadata = dict(frame.metadata)
            metadata["nrzi"] = self._nrzi
            if self._nrzi:
                metadata["line_polarity_unobservable"] = True
                frame = replace(frame, polarity=Polarity.AMBIGUOUS, metadata=metadata)
            else:
                metadata["line_polarity_unobservable"] = False
                frame = replace(frame, metadata=metadata)
            output.append(frame)
        return output

    def flush(self) -> list[FrameResult]:
        self._previous_level = 1
        return self._decoder.flush()


def build_fx25(parameters: Mapping[str, object]) -> Fx25StreamingDecoder:
    return Fx25StreamingDecoder(
        nrzi=bool(parameters.get("nrzi", True)),
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
    )


__all__ = [
    "CAPTURE_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "SYNCWORD",
    "Fx25StreamingDecoder",
    "build_fx25",
]

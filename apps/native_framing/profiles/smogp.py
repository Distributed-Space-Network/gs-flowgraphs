"""SMOG-P signalling receive profile.

Syncwords and fixed extraction behavior are adapted from gr-satellites
``smogp_signalling_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2019 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.types import FrameResult, IntegrityStatus

RX_SYNCWORD = "0010110111010100100101111111110111010011011110110000111100011111"
TX_SYNCWORD = "0010110111010100101000111001111000011010010101010110101111001011"
FRAME_SIZE = 64
DEFAULT_SYNC_THRESHOLD = 8


def _wire_decoder(variant: str):
    def decode(wire: bytes) -> DecodedFixedFrame:
        return DecodedFixedFrame(
            payload=wire,
            integrity=IntegrityStatus.NOT_PRESENT,
            metadata={
                "sync_variant": variant,
                "false_positive_policy": "explicit-profile only; never autodetect",
            },
        )

    return decode


class SmogpSignallingDecoder:
    def __init__(self, *, sync_threshold: int, new_protocol: bool) -> None:
        variants = [("rx", RX_SYNCWORD)]
        if new_protocol:
            variants.append(("tx-observation", TX_SYNCWORD))
        self._decoders = tuple(
            FixedSyncFrameDecoder(
                canonical="smogp_signalling",
                syncword=syncword,
                frame_size=FRAME_SIZE,
                sync_threshold=sync_threshold,
                decode_wire=_wire_decoder(name),
            )
            for name, syncword in variants
        )

    @property
    def retained_symbols(self) -> int:
        return sum(decoder.retained_symbols for decoder in self._decoders)

    @property
    def max_retained_symbols(self) -> int:
        return sum(decoder.max_retained_symbols for decoder in self._decoders)

    def push(self, symbols: Sequence[float]) -> list[FrameResult]:
        frames = [frame for decoder in self._decoders for frame in decoder.push(symbols)]
        unique: dict[tuple[int, int, bytes], FrameResult] = {}
        for frame in frames:
            unique.setdefault((frame.source_start, frame.source_end, frame.payload), frame)
        return sorted(unique.values(), key=lambda frame: (frame.source_start, frame.source_end))

    def flush(self) -> list[FrameResult]:
        return [frame for decoder in self._decoders for frame in decoder.flush()]


def build_smogp_signalling(parameters: Mapping[str, object]) -> SmogpSignallingDecoder:
    return SmogpSignallingDecoder(
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        new_protocol=bool(parameters.get("new_protocol", False)),
    )


__all__ = [
    "DEFAULT_SYNC_THRESHOLD",
    "FRAME_SIZE",
    "RX_SYNCWORD",
    "TX_SYNCWORD",
    "SmogpSignallingDecoder",
    "build_smogp_signalling",
]

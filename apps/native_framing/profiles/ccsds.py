"""Native CCSDS RS and explicitly uncoded hard-bit profiles.

Receive-chain parameters are adapted from gr-satellites
``ccsds_rs_deframer.py`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``.

Copyright 2019 Daniel Estévez <daniel@destevez.net>
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import ccsds_randomize, differential_decode
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import FrameResult, IntegrityStatus, Polarity
from native_framing.viterbi import CONVENTIONS, StreamingViterbiDecoder

SYNCWORD = "00011010110011111111110000011101"
DEFAULT_FRAME_SIZE = 223
DEFAULT_SYNC_THRESHOLD = 4


def _validated(parameters: Mapping[str, object], *, rs_enabled: bool):
    frame_size = int(parameters.get("frame_size", DEFAULT_FRAME_SIZE))
    interleaving = int(parameters.get("rs_interleaving", 1))
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    if interleaving <= 0 or interleaving > 8:
        raise ValueError("rs_interleaving must be between 1 and 8")
    if frame_size % interleaving:
        raise ValueError("rs_interleaving must divide frame_size")
    if rs_enabled and frame_size // interleaving > 223:
        raise ValueError("each RS path may contain at most 223 data symbols")
    return frame_size, interleaving


def _wire_decoder(
    *, randomize: bool, codec: CcsdsReedSolomon | None
):
    def decode(wire: bytes) -> DecodedFixedFrame | None:
        channel = ccsds_randomize(wire) if randomize else wire
        if codec is None:
            return DecodedFixedFrame(
                payload=channel,
                integrity=IntegrityStatus.NOT_PRESENT,
                metadata={
                    "randomizer": "CCSDS" if randomize else "none",
                    "false_positive_policy": "explicit-profile only; no integrity gate",
                },
            )
        result = codec.decode(channel)
        if result is None:
            return None
        return DecodedFixedFrame(
            payload=result.payload,
            corrected_symbols=result.corrected_symbols,
            metadata={
                "randomizer": "CCSDS" if randomize else "none",
                "rs_basis": codec.basis,
                "rs_interleaving": codec.interleaving,
            },
        )

    return decode


class CcsdsHardDecoder:
    """Optional GNU Radio-compatible differential stage before hard sync."""

    def __init__(self, decoder: FixedSyncFrameDecoder, *, differential: bool) -> None:
        self._decoder = decoder
        self._differential = bool(differential)
        self._previous = 0

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
        if self._differential:
            decoded = differential_decode(hard, initial=self._previous)
            if hard.size:
                self._previous = int(hard[-1])
        else:
            decoded = hard
        return self._annotate(self._decoder.push(decoded))

    def _annotate(self, frames: list[FrameResult]) -> list[FrameResult]:
        output = []
        for frame in frames:
            metadata = dict(frame.metadata)
            metadata["precoding"] = (
                "differential" if self._differential else "none"
            )
            metadata["line_polarity_unobservable"] = self._differential
            if self._differential:
                frame = replace(
                    frame,
                    polarity=Polarity.AMBIGUOUS,
                    metadata=metadata,
                )
            else:
                frame = replace(frame, metadata=metadata)
            output.append(frame)
        return output

    def flush(self) -> list[FrameResult]:
        self._previous = 0
        return self._annotate(self._decoder.flush())


def build_ccsds_rs(parameters: Mapping[str, object]) -> CcsdsHardDecoder:
    frame_size, interleaving = _validated(parameters, rs_enabled=True)
    basis = str(parameters.get("rs_basis", "dual"))
    scrambler = str(parameters.get("scrambler", "CCSDS"))
    codec = CcsdsReedSolomon(basis=basis, interleaving=interleaving)
    return CcsdsHardDecoder(
        FixedSyncFrameDecoder(
            canonical="ccsds_reed_solomon",
            syncword=SYNCWORD,
            frame_size=frame_size + 32 * interleaving,
            sync_threshold=int(
                parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)
            ),
            decode_wire=_wire_decoder(randomize=scrambler == "CCSDS", codec=codec),
        ),
        differential=parameters.get("precoding", "none") == "differential",
    )


def build_ccsds_uncoded(parameters: Mapping[str, object]) -> CcsdsHardDecoder:
    frame_size, _ = _validated(parameters, rs_enabled=False)
    scrambler = str(parameters.get("scrambler", "CCSDS"))
    return CcsdsHardDecoder(
        FixedSyncFrameDecoder(
            canonical="ccsds_uncoded",
            syncword=SYNCWORD,
            frame_size=frame_size,
            sync_threshold=int(
                parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)
            ),
            decode_wire=_wire_decoder(randomize=scrambler == "CCSDS", codec=None),
        ),
        differential=parameters.get("precoding", "none") == "differential",
    )


class CcsdsConcatenatedDecoder:
    """Bounded dual-pair-phase convolutional + ASM/RS streaming decoder."""

    def __init__(self, parameters: Mapping[str, object]) -> None:
        frame_size, interleaving = _validated(
            parameters, rs_enabled=bool(parameters.get("rs_enabled", True))
        )
        rs_enabled = bool(parameters.get("rs_enabled", True))
        common = dict(parameters)
        common["frame_size"] = frame_size
        common["rs_interleaving"] = interleaving
        convention = str(parameters.get("convolutional", "CCSDS"))
        traceback = int(parameters.get("viterbi_traceback", 80))
        self._viterbi = (
            StreamingViterbiDecoder(convention, traceback_depth=traceback),
            StreamingViterbiDecoder(convention, traceback_depth=traceback),
        )
        # Match upstream's one-soft-symbol delay hypothesis. The leading neutral
        # symbol produces one irrelevant decoded bit before the correctly paired
        # stream when the input has a one-symbol phase offset.
        self._viterbi[1].push((0.0,))
        self._frames = (
            build_ccsds_rs(common) if rs_enabled else build_ccsds_uncoded(common),
            build_ccsds_rs(common) if rs_enabled else build_ccsds_uncoded(common),
        )

    @property
    def retained_symbols(self) -> int:
        return sum(decoder.retained_symbol_count for decoder in self._viterbi) + sum(
            decoder.retained_symbols for decoder in self._frames
        )

    @property
    def max_retained_symbols(self) -> int:
        return sum(2 * decoder.traceback_depth + 1 for decoder in self._viterbi) + sum(
            decoder.max_retained_symbols for decoder in self._frames
        )

    def push(self, symbols) -> list:
        soft = np.asarray(symbols)
        if soft.ndim != 1:
            raise ValueError("soft-symbol chunks must be one-dimensional")
        if soft.size and not np.all(np.isfinite(soft)):
            raise ValueError("soft-symbol chunks must be finite")
        output = []
        for phase, (viterbi, frames) in enumerate(
            zip(self._viterbi, self._frames, strict=True)
        ):
            decoded = viterbi.push(soft)
            output.extend(self._map_frame(frame, phase) for frame in frames.push(decoded))
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError("CCSDS concatenated decoder retained-symbol bound violated")
        return output

    def flush(self) -> list:
        output = []
        for phase, (viterbi, frames) in enumerate(
            zip(self._viterbi, self._frames, strict=True)
        ):
            if viterbi.retained_symbol_count % 2:
                decoded = viterbi.push((0.0,))
                output.extend(self._map_frame(frame, phase) for frame in frames.push(decoded))
            decoded = viterbi.flush()
            output.extend(self._map_frame(frame, phase) for frame in frames.push(decoded))
            frames.flush()
        return output

    @staticmethod
    def _map_frame(frame, phase: int):
        decoded_lead = phase
        source_start = max(0, 2 * (frame.source_start - decoded_lead) + phase)
        source_end = max(source_start, 2 * (frame.source_end - decoded_lead) + phase)
        metadata = dict(frame.metadata)
        metadata.update(
            {
                "convolutional_phase": phase,
                "source_offset_domain": "input_soft_symbols",
            }
        )
        return replace(
            frame,
            canonical_framing="ccsds_concatenated",
            source_start=source_start,
            source_end=source_end,
            metadata=metadata,
        )


def build_ccsds_concatenated(
    parameters: Mapping[str, object],
) -> CcsdsConcatenatedDecoder:
    convention = str(parameters.get("convolutional", "CCSDS"))
    if convention not in CONVENTIONS:
        raise ValueError(f"convolutional must be one of {tuple(CONVENTIONS)!r}")
    return CcsdsConcatenatedDecoder(parameters)


__all__ = [
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "SYNCWORD",
    "CcsdsHardDecoder",
    "CcsdsConcatenatedDecoder",
    "build_ccsds_concatenated",
    "build_ccsds_rs",
    "build_ccsds_uncoded",
]

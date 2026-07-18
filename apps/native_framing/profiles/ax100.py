"""Native GOMspace AX100 Mode 6 (RS) decoder.

The chain and packet-length semantics follow gr-satellites ``ax100_deframer.py`` and
``ax100_decode_impl.cc`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``. The multiplicative descrambler behavior follows
the pinned GNU Radio ``digital.descrambler_bb`` implementation.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from native_framing.codes.golay24 import decode_golay24
from native_framing.fixed import DecodedFixedFrame, FixedSyncFrameDecoder
from native_framing.linecode import SelfSynchronizingDescrambler, ccsds_randomize
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import FrameResult

SYNCWORD = "10010011000010110101000111011110"
CAPTURE_SIZE = 256
ASM_CAPTURE_SIZE = 258
DEFAULT_SYNC_THRESHOLD = 4
DESCRAMBLER_MASK = 0x21
DESCRAMBLER_SEED = 0
DESCRAMBLER_LENGTH = 16

_RS = CcsdsReedSolomon(basis="conventional", interleaving=1)


def _decode_mode6_wire(wire: bytes) -> DecodedFixedFrame | None:
    if len(wire) != CAPTURE_SIZE:
        return None
    declared_length = wire[0]
    # libfec accepts pad <= 222. AX100 passes pad=256-declared_length, so the smallest
    # decodable declaration is 34; 255 is the largest representable byte value.
    if declared_length < 34:
        return None
    codeword = wire[1:declared_length]
    decoded = _RS.decode(codeword)
    if decoded is None or not decoded.payload:
        return None
    return DecodedFixedFrame(
        payload=decoded.payload,
        corrected_symbols=decoded.corrected_symbols,
        metadata={
            "ax100_mode": 6,
            "declared_length": declared_length,
            "rs_basis": "conventional",
            "rs_field_polynomial": "0x187",
            "rs_first_root": 112,
            "rs_primitive_step": 11,
            "rs_parity_symbols": 32,
            "self_synchronizing_descrambler": "mask=0x21,seed=0,length=16",
        },
    )


def _decode_asm_wire(
    wire: bytes, *, canonical: str, randomize: bool
) -> DecodedFixedFrame | None:
    if len(wire) != ASM_CAPTURE_SIZE:
        return None
    header = decode_golay24(int.from_bytes(wire[:3], "big"))
    if header is None:
        return None
    declared_length = header.data & 0xFF
    viterbi_flag = bool(header.data & 0x100)
    scrambler_flag = bool(header.data & 0x200)
    rs_flag = bool(header.data & 0x400)
    if declared_length < 33:
        return None
    packet = wire[3 : 3 + declared_length]
    if len(packet) != declared_length:
        return None
    channel = ccsds_randomize(packet) if randomize else packet
    decoded = _RS.decode(channel)
    if decoded is None or not decoded.payload:
        return None
    return DecodedFixedFrame(
        payload=decoded.payload,
        corrected_symbols=decoded.corrected_symbols,
        metadata={
            "ax100_mode": 5,
            "declared_length": declared_length,
            "golay_corrected_bits": header.corrected_bits,
            "header_viterbi_flag": viterbi_flag,
            "header_scrambler_flag": scrambler_flag,
            "header_rs_flag": rs_flag,
            "forced_viterbi": False,
            "forced_scrambler": "CCSDS" if randomize else "none",
            "forced_rs": True,
            "rs_basis": "conventional",
            "rs_field_polynomial": "0x187",
            "rs_first_root": 112,
            "rs_primitive_step": 11,
            "rs_parity_symbols": 32,
            "profile_equivalence": "AX100 Mode 5 and AX100 ASM+Golay share the pinned ASM path",
            "canonical_profile": canonical,
        },
    )


class Ax100Mode6Decoder:
    """Stream descrambler followed by the bounded fixed-sync Mode 6 collector."""

    def __init__(self, *, sync_threshold: int = DEFAULT_SYNC_THRESHOLD) -> None:
        self._descrambler = SelfSynchronizingDescrambler(
            DESCRAMBLER_MASK, DESCRAMBLER_SEED, DESCRAMBLER_LENGTH
        )
        self._inner = FixedSyncFrameDecoder(
            canonical="ax100_mode6",
            syncword=SYNCWORD,
            frame_size=CAPTURE_SIZE,
            sync_threshold=sync_threshold,
            decode_wire=_decode_mode6_wire,
        )

    @property
    def retained_symbols(self) -> int:
        return self._inner.retained_symbols

    @property
    def max_retained_symbols(self) -> int:
        return self._inner.max_retained_symbols

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        return self._inner.push(self._descrambler.push(np.asarray(symbols)))

    def flush(self) -> list[FrameResult]:
        output = self._inner.flush()
        self._descrambler.reset()
        return output


def build_ax100_mode6(parameters: Mapping[str, object]) -> Ax100Mode6Decoder:
    return Ax100Mode6Decoder(
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD))
    )


def _build_asm(parameters: Mapping[str, object], *, canonical: str) -> FixedSyncFrameDecoder:
    randomize = str(parameters.get("scrambler", "CCSDS")) == "CCSDS"
    return FixedSyncFrameDecoder(
        canonical=canonical,
        syncword=SYNCWORD,
        frame_size=ASM_CAPTURE_SIZE,
        sync_threshold=int(parameters.get("sync_threshold", DEFAULT_SYNC_THRESHOLD)),
        decode_wire=lambda wire: _decode_asm_wire(
            wire, canonical=canonical, randomize=randomize
        ),
    )


def build_ax100_mode5(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return _build_asm(parameters, canonical="ax100_mode5")


def build_ax100_asm_golay(parameters: Mapping[str, object]) -> FixedSyncFrameDecoder:
    return _build_asm(parameters, canonical="ax100_asm_golay")


__all__ = [
    "Ax100Mode6Decoder",
    "ASM_CAPTURE_SIZE",
    "CAPTURE_SIZE",
    "DEFAULT_SYNC_THRESHOLD",
    "DESCRAMBLER_LENGTH",
    "DESCRAMBLER_MASK",
    "DESCRAMBLER_SEED",
    "SYNCWORD",
    "build_ax100_asm_golay",
    "build_ax100_mode5",
    "build_ax100_mode6",
]

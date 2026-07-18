"""Repeat-accumulate codec used by the SMOG family.

This is a NumPy port of gr-satellites ``lib/radecoder`` at commit
``b8b227d456a6c7e65a590dfb8f00e80e89d86a3c``. The original code is by
Miklos Maroti and Daniel Estevez.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RA_BITCOUNT = 16
RA_PUNCTURE_RATE = 3
DEFAULT_PASSES = 40
DEFAULT_ERROR_THRESHOLD = 0.35

_MASKS = (
    (0x12, 0x17, 0x1B, 0x1E),
    (0x21, 0x2D, 0x30, 0x39),
    (0x41, 0x53, 0x69, 0x7B),
    (0x8E, 0xAF, 0xC3, 0xE7),
    (0x108, 0x13B, 0x168, 0x1DC),
    (0x204, 0x2E3, 0x369, 0x3AA),
    (0x415, 0x4BF, 0x553, 0x62B),
    (0x83E, 0x939, 0xAF5, 0xD70),
    (0x1013, 0x109D, 0x117D, 0x1271),
)


@dataclass(frozen=True)
class RaConfig:
    data_length: int
    check_length: int
    code_length: int
    highbit: int
    masks: tuple[int, int, int, int]


@dataclass(frozen=True)
class RaDecodeResult:
    payload: bytes
    recode_bit_errors: int
    recode_error_fraction: float


def ra_config(frame_size: int) -> RaConfig:
    if frame_size < 8 or frame_size > 4096 or frame_size % 2:
        raise ValueError("RA frame size must be an even number from 8 through 4096 bytes")
    data_length = frame_size // 2
    check_length = (data_length + RA_PUNCTURE_RATE - 1) // RA_PUNCTURE_RATE
    code_length = data_length + check_length * 3
    reduced = data_length
    highbit = 4
    while reduced >= 32:
        reduced //= 2
        highbit += 1
    if not 4 <= highbit <= 12:
        raise ValueError("RA frame size selects an unsupported LFSR order")
    return RaConfig(
        data_length=data_length,
        check_length=check_length,
        code_length=code_length,
        highbit=highbit,
        masks=_MASKS[highbit - 4],
    )


def _positions(config: RaConfig, sequence: int) -> np.ndarray:
    mask = config.masks[sequence]
    offset = config.data_length >> (1 + sequence)
    state = 1 + sequence + offset
    output = np.empty(config.data_length, dtype=np.intp)
    for index in range(config.data_length):
        while True:
            bit = state & 1
            state >>= 1
            state ^= (-bit) & mask
            if state <= config.data_length:
                break
        position = state - 1
        if position < offset:
            position += config.data_length
        output[index] = position - offset
    return output


def encode_ra(payload: bytes) -> np.ndarray:
    """Return the upstream RA wire words for an even-sized payload."""

    config = ra_config(len(payload))
    packet = np.frombuffer(payload, dtype="<u2").astype(np.uint16, copy=False)
    positions = tuple(_positions(config, sequence) for sequence in range(4))
    cursors = [0, 0, 0, 0]
    nextword = 0
    pass_number = 0
    output = np.empty(config.code_length, dtype=np.uint16)
    for output_index in range(config.code_length):
        word = nextword
        count = 1 if pass_number == 0 else RA_PUNCTURE_RATE
        while True:
            word = ((word >> 1) | (word << 15)) & 0xFFFF
            position = int(positions[pass_number][cursors[pass_number]])
            cursors[pass_number] += 1
            word ^= int(packet[position])
            if position == pass_number:
                break
            count -= 1
            if count == 0:
                break
        if count != 0:
            nextword = 0
            pass_number = (pass_number + 1) % 4
            cursors[pass_number] = 0
        else:
            nextword = word
        output[output_index] = word
    return output


def ra_wire_soft(payload: bytes, *, magnitude: float = 1.0) -> np.ndarray:
    """Construct upstream-ordered soft symbols for tests and differential checks."""

    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("RA soft-symbol magnitude must be finite and positive")
    wire = encode_ra(payload).astype("<u2", copy=False).view(np.uint8)
    bits = np.unpackbits(wire)
    return np.where(bits != 0, magnitude, -magnitude).astype(np.float64)


def _llr_min(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.copysign(np.minimum(np.abs(left), np.abs(right)), left * right)


def _improve(
    dataword: np.ndarray,
    codeword: np.ndarray,
    positions: np.ndarray,
    *,
    puncture: int,
    half: bool,
) -> None:
    data_length = dataword.shape[0]
    accumulator = np.full(RA_BITCOUNT, np.finfo(np.float32).max, dtype=np.float64)
    forward = np.empty_like(dataword)
    code_index = 0
    for index, position in enumerate(positions):
        data = dataword[position]
        forward[index] = accumulator
        accumulator = _llr_min(accumulator, data)
        if (index + 1) % puncture == 0:
            accumulator += codeword[code_index]
            code_index += 1
        accumulator = np.roll(accumulator, -1)

    if data_length % puncture:
        accumulator += 2.0 * np.roll(codeword[code_index], -1)

    for index in range(data_length - 1, -1, -1):
        accumulator = np.roll(accumulator, 1)
        if (index + 1) % puncture == 0:
            code_index -= 1
            accumulator += codeword[code_index]
        position = positions[index]
        left = _llr_min(forward[index], accumulator)
        data = dataword[position].copy()
        accumulator = _llr_min(accumulator, data)
        if half:
            data *= 0.5
        dataword[position] = left + data


def decode_ra_soft(
    soft_symbols: np.ndarray,
    *,
    frame_size: int,
    passes: int = DEFAULT_PASSES,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
) -> RaDecodeResult | None:
    """Decode upstream-ordered RA soft symbols and apply its recode-distance gate."""

    config = ra_config(frame_size)
    soft = np.asarray(soft_symbols, dtype=np.float64)
    expected = config.code_length * RA_BITCOUNT
    if soft.shape != (expected,) or not np.all(np.isfinite(soft)):
        return None
    if passes <= 0:
        raise ValueError("RA decoder passes must be positive")
    if not 0.0 <= error_threshold <= DEFAULT_ERROR_THRESHOLD:
        raise ValueError("RA recode error threshold must be between 0 and 0.35")

    ra_input = -soft.reshape((-1, 8))[:, ::-1].reshape((-1, 16))
    dataword = np.zeros((config.data_length, RA_BITCOUNT), dtype=np.float64)
    positions = tuple(_positions(config, sequence) for sequence in range(4))
    offset = 0
    for pass_index in range(passes):
        offset = 0
        for sequence in range(4):
            length = config.data_length if sequence == 0 else config.check_length
            _improve(
                dataword,
                ra_input[offset : offset + length],
                positions[sequence],
                puncture=1 if sequence == 0 else RA_PUNCTURE_RATE,
                half=pass_index > 0,
            )
            offset += length
    if offset != config.code_length:
        raise RuntimeError("RA codeword partition invariant violated")

    words = np.zeros(config.data_length, dtype=np.uint16)
    for bit in range(RA_BITCOUNT):
        words |= (dataword[:, bit] < 0).astype(np.uint16) << bit
    payload = words.astype("<u2", copy=False).view(np.uint8).tobytes()
    recoded = encode_ra(payload)
    recoded_bits = (
        recoded[:, None] & (1 << np.arange(RA_BITCOUNT, dtype=np.uint16))
    ) != 0
    errors = int(np.count_nonzero((ra_input >= 0) == recoded_bits))
    fraction = errors / expected
    if fraction >= error_threshold:
        return None
    return RaDecodeResult(payload, errors, fraction)


__all__ = [
    "DEFAULT_ERROR_THRESHOLD",
    "DEFAULT_PASSES",
    "RA_BITCOUNT",
    "RaConfig",
    "RaDecodeResult",
    "decode_ra_soft",
    "encode_ra",
    "ra_config",
    "ra_wire_soft",
]

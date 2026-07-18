"""Reusable line-code, reflection, and additive-randomizer primitives.

The PN9 convention matches GNU Radio ``additive_scrambler_bb(0x21, 0x1ff,
8, bits_per_byte=8)`` and the independently pinned TinyGS implementation:
the register output is packed least-significant bit first in each byte.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from gfsk_ax25.g3ruh import descramble as g3ruh_descramble
from gfsk_ax25.g3ruh import nrzi_decode, nrzi_encode
from gfsk_ax25.g3ruh import scramble as g3ruh_scramble


def reflect_bytes(data: bytes) -> bytes:
    """Reverse bit order independently inside every byte."""

    table = _reflection_table()
    return bytes(table[value] for value in data)


@lru_cache(maxsize=1)
def _reflection_table() -> tuple[int, ...]:
    return tuple(int(f"{value:08b}"[::-1], 2) for value in range(256))


def pn9_bytes(data: bytes, *, seed: int = 0x1FF) -> bytes:
    """XOR bytes with the x^9+x^5+1 PN9 sequence, reset per packet."""

    if not 0 < seed <= 0x1FF:
        raise ValueError("PN9 seed must be a non-zero 9-bit value")
    state = seed
    output = bytearray(len(data))
    for index, value in enumerate(data):
        sequence = 0
        for bit_index in range(8):
            output_bit = state & 1
            sequence |= output_bit << bit_index
            feedback = (state & 1) ^ ((state >> 5) & 1)
            state = (state >> 1) | (feedback << 8)
        output[index] = value ^ sequence
    return bytes(output)


def ccsds_randomize(data: bytes) -> bytes:
    """Apply the published CCSDS TM pseudo-randomizer (its own inverse)."""

    from fec import ccsds_randomize as apply  # noqa: PLC0415

    return apply(data)


def additive_randomize_bits(
    bits: np.ndarray, *, mask: int, seed: int, register_length: int
) -> np.ndarray:
    """Apply GNU Radio's additive LFSR sequence to unpacked bits."""

    source = _hard_bits(bits)
    if register_length <= 0 or register_length > 63:
        raise ValueError("register_length must be between 1 and 63")
    if mask <= 0 or mask >= 1 << (register_length + 1):
        raise ValueError("mask does not fit the configured register")
    if seed < 0 or seed >= 1 << (register_length + 1):
        raise ValueError("seed does not fit the configured register")
    state = int(seed)
    output = np.empty_like(source)
    for index, bit in enumerate(source):
        output[index] = int(bit) ^ (state & 1)
        feedback = (state & int(mask)).bit_count() & 1
        state = (state >> 1) | (feedback << register_length)
    return output


def differential_encode(bits: np.ndarray, *, initial: int = 0) -> np.ndarray:
    source = _hard_bits(bits)
    output = np.empty_like(source)
    previous = initial & 1
    for index, bit in enumerate(source):
        previous ^= int(bit)
        output[index] = previous
    return output


def differential_decode(bits: np.ndarray, *, initial: int = 0) -> np.ndarray:
    source = _hard_bits(bits)
    output = np.empty_like(source)
    previous = initial & 1
    for index, bit in enumerate(source):
        current = int(bit)
        output[index] = current ^ previous
        previous = current
    return output


class SelfSynchronizingDescrambler:
    """GNU Radio-compatible multiplicative bit descrambler with streaming state.

    ``register_length`` follows ``digital.lfsr``: the received bit is inserted at that bit
    position after the register shifts right. The transform becomes independent of its seed after
    that many received bits, which is why it is suitable before an access-code correlator.
    """

    def __init__(self, mask: int, seed: int, register_length: int) -> None:
        if register_length <= 0 or register_length > 63:
            raise ValueError("register_length must be between 1 and 63")
        if mask <= 0 or mask >= 1 << (register_length + 1):
            raise ValueError("mask does not fit the configured register")
        if seed < 0 or seed >= 1 << (register_length + 1):
            raise ValueError("seed does not fit the configured register")
        self.mask = int(mask)
        self.seed = int(seed)
        self.register_length = int(register_length)
        self._state = self.seed

    @property
    def state(self) -> int:
        return self._state

    def reset(self) -> None:
        self._state = self.seed

    def push(self, bits: np.ndarray) -> np.ndarray:
        source = _hard_bits(bits)
        output = np.empty_like(source)
        state = self._state
        for index, bit in enumerate(source):
            received = int(bit)
            output[index] = ((state & self.mask).bit_count() & 1) ^ received
            state = (state >> 1) | (received << self.register_length)
        self._state = state
        return output


def multiplicative_scramble(
    bits: np.ndarray, *, mask: int, seed: int, register_length: int
) -> np.ndarray:
    """Encode bits for :class:`SelfSynchronizingDescrambler` (one-to-one bit mapping)."""

    # Reuse constructor validation and its public configuration, but generate the transmitted bit
    # from the requested plain bit before shifting that transmitted bit into the register.
    config = SelfSynchronizingDescrambler(mask, seed, register_length)
    source = _hard_bits(bits)
    output = np.empty_like(source)
    state = config.state
    for index, bit in enumerate(source):
        transmitted = ((state & config.mask).bit_count() & 1) ^ int(bit)
        output[index] = transmitted
        state = (state >> 1) | (transmitted << config.register_length)
    return output


def _hard_bits(bits: np.ndarray) -> np.ndarray:
    source = np.asarray(bits)
    if source.ndim != 1 or (source.size and not np.all((source == 0) | (source == 1))):
        raise ValueError("bits must be a one-dimensional array containing only 0 and 1")
    return source.astype(np.uint8, copy=False)


__all__ = [
    "additive_randomize_bits",
    "differential_decode",
    "differential_encode",
    "ccsds_randomize",
    "g3ruh_descramble",
    "g3ruh_scramble",
    "nrzi_decode",
    "nrzi_encode",
    "pn9_bytes",
    "reflect_bytes",
    "multiplicative_scramble",
    "SelfSynchronizingDescrambler",
]

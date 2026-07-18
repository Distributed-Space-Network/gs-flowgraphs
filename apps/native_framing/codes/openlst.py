"""CC1110 DN504 convolutional coding and interleaving used by OpenLST.

The receive algorithm is an engine-independent extraction of the pinned
``gr-satellites`` OpenLST deframer.  The encoder is the inverse trellis used
for construction tests; exposing it does not constitute a transmit-path claim.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

_SOURCE_STATES = (
    (0, 4),
    (0, 4),
    (1, 5),
    (1, 5),
    (2, 6),
    (2, 6),
    (3, 7),
    (3, 7),
)
_TRANSITION_OUTPUTS = (
    (0, 3),
    (3, 0),
    (1, 2),
    (2, 1),
    (3, 0),
    (0, 3),
    (2, 1),
    (1, 2),
)
_TRANSITION_INPUTS = (0, 1, 0, 1, 0, 1, 0, 1)


def interleave_openlst_chunk(chunk: bytes) -> bytes:
    """Apply the involutive OpenLST 4-byte dibit matrix transpose."""

    if len(chunk) != 4:
        raise ValueError("OpenLST interleaving requires exactly four bytes")
    value = int.from_bytes(chunk, byteorder="little")
    grid: list[list[int]] = []
    for _ in range(4):
        row: list[int] = []
        for _ in range(4):
            row.append((value & 0xC0000000) >> 30)
            value <<= 2
        grid.append(row)

    flipped = 0
    for x in range(4):
        for y in range(4):
            flipped = (flipped << 2) | grid[y][x]
    return flipped.to_bytes(4, byteorder="little")


def decode_openlst_fec(encoded: bytes) -> bytes:
    """Decode complete 4-byte OpenLST FEC chunks using the pinned traceback."""

    if not encoded or len(encoded) % 4:
        raise ValueError("OpenLST encoded data must be a non-empty multiple of four bytes")

    path_bits = 0
    cost = [[100] * 8, [0] * 8]
    path = [[0] * 8, [0] * 8]
    last_buffer = 0
    current_buffer = 1
    output = bytearray()

    for offset in range(0, len(encoded), 4):
        chunk = interleave_openlst_chunk(encoded[offset : offset + 4])
        for value in chunk:
            for shift in (6, 4, 2, 0):
                symbol = (value >> shift) & 0x03
                minimum_cost = 0xFF
                for destination in range(8):
                    input_bit = _TRANSITION_INPUTS[destination]
                    source0, source1 = _SOURCE_STATES[destination]
                    cost0 = cost[last_buffer][source0] + (
                        symbol ^ _TRANSITION_OUTPUTS[destination][0]
                    ).bit_count()
                    cost1 = cost[last_buffer][source1] + (
                        symbol ^ _TRANSITION_OUTPUTS[destination][1]
                    ).bit_count()
                    if cost0 < cost1:
                        selected_cost = cost0
                        selected_path = path[last_buffer][source0]
                    else:
                        selected_cost = cost1
                        selected_path = path[last_buffer][source1]
                    cost[current_buffer][destination] = selected_cost
                    path[current_buffer][destination] = (selected_path << 1) | input_bit
                    minimum_cost = min(minimum_cost, selected_cost)

                path_bits += 1
                if path_bits >= 32:
                    output.append((path[current_buffer][0] >> 24) & 0xFF)
                    path_bits -= 8
                last_buffer, current_buffer = current_buffer, last_buffer
                for state in range(8):
                    cost[last_buffer][state] -= minimum_cost
    return bytes(output)


def encode_openlst_fec(data: bytes, *, encoded_size: int | None = None) -> bytes:
    """Encode bytes with zero termination and OpenLST 4-byte interleaving."""

    minimum_size = 2 * (len(data) + 3)
    if encoded_size is None:
        encoded_size = ((minimum_size + 3) // 4) * 4
    if encoded_size <= 0 or encoded_size % 4:
        raise ValueError("OpenLST encoded size must be a positive multiple of four")
    if encoded_size < minimum_size:
        raise ValueError("OpenLST encoded size leaves fewer than 24 termination bits")

    bits = [
        (value >> shift) & 1
        for value in data
        for shift in range(7, -1, -1)
    ]
    bits.extend([0] * (encoded_size * 4 - len(bits)))

    state = 0
    encoded_bits: list[int] = []
    for bit in bits:
        destination = ((state << 1) & 0x07) | bit
        branch = 0 if state < 4 else 1
        symbol = _TRANSITION_OUTPUTS[destination][branch]
        encoded_bits.extend((symbol >> 1, symbol & 1))
        state = destination

    packed = bytearray()
    for offset in range(0, len(encoded_bits), 8):
        value = 0
        for bit in encoded_bits[offset : offset + 8]:
            value = (value << 1) | bit
        packed.append(value)
    return b"".join(
        interleave_openlst_chunk(bytes(packed[offset : offset + 4]))
        for offset in range(0, len(packed), 4)
    )


__all__ = ["decode_openlst_fec", "encode_openlst_fec", "interleave_openlst_chunk"]

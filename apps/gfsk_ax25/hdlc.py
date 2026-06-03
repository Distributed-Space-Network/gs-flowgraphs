"""HDLC bit-level framing for AX.25 (flags, bit-stuffing, FCS).

AX.25 frames are delimited by the flag octet ``0x7E`` (``01111110``), the body
is protected by a 16-bit :mod:`.fcs`, and the serial bit stream is sent
least-significant-bit first with bit-stuffing (a ``0`` is inserted after any run
of five ``1``s so the flag pattern never appears inside a frame).

Bits are represented as 1-D ``numpy.uint8`` arrays of 0/1 in transmission order.

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import numpy as np

from . import fcs

FLAG_BITS: tuple[int, ...] = (0, 1, 1, 1, 1, 1, 1, 0)  # 0x7E, LSB-first on the wire

# Smallest sensible AX.25 UI frame: dest(7) + src(7) + control(1) + pid(1) +
# fcs(2) = 18 octets. Used to reject flag-noise fragments before the FCS check.
_MIN_FRAME_OCTETS = 18


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Expand ``data`` to a uint8 bit array, least-significant bit first."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="little")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a uint8 bit array (LSB first) back to bytes. Truncates a trailing
    partial octet."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    if n == 0:
        return b""
    return np.packbits(bits[:n], bitorder="little").tobytes()


def bit_stuff(bits: np.ndarray) -> list[int]:
    """Insert a 0 after every run of five consecutive 1s."""
    out: list[int] = []
    ones = 0
    for bit in bits.tolist():
        out.append(bit)
        if bit == 1:
            ones += 1
            if ones == 5:
                out.append(0)
                ones = 0
        else:
            ones = 0
    return out


def _destuff(bits: list[int]) -> list[int]:
    """Remove the 0 stuffed after each run of five 1s."""
    out: list[int] = []
    ones = 0
    for bit in bits:
        if ones == 5 and bit == 0:
            ones = 0
            continue  # drop the stuffed zero
        out.append(bit)
        ones = ones + 1 if bit == 1 else 0
    return out


def frame(body: bytes, *, preamble_flags: int = 1, postamble_flags: int = 1) -> np.ndarray:
    """Frame ``body`` (AX.25 address+control+pid+info) into an HDLC bit stream.

    Appends the FCS, bit-stuffs, and brackets the result with flag octets.
    ``preamble_flags`` extra leading flags give the receiver clock/sync runway.
    """
    body_fcs = body + fcs.fcs_bytes(body)
    stuffed = bit_stuff(bytes_to_bits(body_fcs))
    flags_pre = list(FLAG_BITS) * max(1, preamble_flags)
    flags_post = list(FLAG_BITS) * max(1, postamble_flags)
    return np.array(flags_pre + stuffed + flags_post, dtype=np.uint8)


def deframe(bits: np.ndarray) -> list[bytes]:
    """Recover valid AX.25 frame bodies (FCS stripped) from a bit stream.

    Splits on flag octets, de-stuffs the content between flags, and keeps only
    byte-aligned fragments whose FCS verifies. Robust to preamble flag runs,
    bit offset, and trailing noise.
    """
    bl = np.asarray(bits, dtype=np.uint8).tolist()
    n = len(bl)
    flag = list(FLAG_BITS)

    # Flag start indices. A valid flag cannot overlap another (0x7E does not
    # self-overlap within 8 bits), so skip a whole octet on a hit.
    starts: list[int] = []
    i = 0
    while i <= n - 8:
        if bl[i : i + 8] == flag:
            starts.append(i)
            i += 8
        else:
            i += 1

    frames: list[bytes] = []
    for k in range(len(starts) - 1):
        start = starts[k] + 8
        end = starts[k + 1]
        if end <= start:
            continue  # adjacent flags (preamble) — no content between them
        content = _destuff(bl[start:end])
        if len(content) % 8 != 0:
            continue
        body_fcs = bits_to_bytes(np.array(content, dtype=np.uint8))
        if len(body_fcs) >= _MIN_FRAME_OCTETS and fcs.check(body_fcs):
            frames.append(body_fcs[:-2])
    return frames


__all__ = [
    "FLAG_BITS",
    "bit_stuff",
    "bits_to_bytes",
    "bytes_to_bits",
    "deframe",
    "frame",
]

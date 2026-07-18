"""Parameterized CRC catalog with explicit wire byte order."""

from __future__ import annotations

from dataclasses import dataclass


def _reflect(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


@dataclass(frozen=True)
class CrcSpec:
    name: str
    width: int
    polynomial: int
    initial: int
    xor_output: int
    reflect_input: bool
    reflect_output: bool

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width % 8:
            raise ValueError("CRC width must be a positive multiple of eight")
        mask = (1 << self.width) - 1
        for name, value in (
            ("polynomial", self.polynomial),
            ("initial", self.initial),
            ("xor_output", self.xor_output),
        ):
            if value < 0 or value > mask:
                raise ValueError(f"CRC {name} does not fit width")

    @property
    def byte_width(self) -> int:
        return self.width // 8

    def compute(self, data: bytes) -> int:
        mask = (1 << self.width) - 1
        register = self.initial
        if self.reflect_input:
            polynomial = _reflect(self.polynomial, self.width)
            for value in data:
                register ^= value
                for _ in range(8):
                    register = (register >> 1) ^ (polynomial if register & 1 else 0)
        else:
            high_bit = 1 << (self.width - 1)
            for value in data:
                register ^= value << (self.width - 8)
                for _ in range(8):
                    register = ((register << 1) & mask) ^ (
                        self.polynomial if register & high_bit else 0
                    )
        # The right-shifting algorithm already returns reflected output when
        # refin==refout, which is the convention used by the catalog entries.
        if self.reflect_output != self.reflect_input:
            register = _reflect(register, self.width)
        return (register ^ self.xor_output) & mask

    def append(self, data: bytes, *, byteorder: str) -> bytes:
        return bytes(data) + self.compute(data).to_bytes(self.byte_width, byteorder)

    def strip_if_valid(self, frame: bytes, *, byteorder: str) -> bytes | None:
        if byteorder not in ("big", "little"):
            raise ValueError("CRC byteorder must be 'big' or 'little'")
        if len(frame) < self.byte_width:
            return None
        payload = frame[: -self.byte_width]
        received = int.from_bytes(frame[-self.byte_width :], byteorder)
        return payload if self.compute(payload) == received else None


CRC16_CCITT_FALSE = CrcSpec("CRC-16/CCITT-FALSE", 16, 0x1021, 0xFFFF, 0, False, False)
CRC16_X25 = CrcSpec("CRC-16/X-25", 16, 0x1021, 0xFFFF, 0xFFFF, True, True)
CRC16_ARC = CrcSpec("CRC-16/ARC", 16, 0x8005, 0, 0, True, True)
CRC16_CC11XX = CrcSpec("CRC-16/CC11XX", 16, 0x8005, 0xFFFF, 0, False, False)

CATALOG = {
    spec.name: spec
    for spec in (CRC16_CCITT_FALSE, CRC16_X25, CRC16_ARC, CRC16_CC11XX)
}

__all__ = [
    "CATALOG",
    "CRC16_ARC",
    "CRC16_CC11XX",
    "CRC16_CCITT_FALSE",
    "CRC16_X25",
    "CrcSpec",
]

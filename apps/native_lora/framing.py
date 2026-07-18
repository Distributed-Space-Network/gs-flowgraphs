"""Bounded LoRa PHY symbol-to-payload framing.

SPDX-License-Identifier: GPL-3.0-only

Adapted on 2026-07-18 from the receive and matching transmit primitives in
tapparelj/gr-lora_sdr at commit 862746dd1cf635c9c8a4bfbaa2c3a0ec3a5306c9
(GPL-3.0-only).  The native API, validation, diagnostics, and bounded whole-
frame orchestration are repository-specific modifications.

The input boundary is the hard-symbol output of gr-lora_sdr's ``fft_demod``:
the FFT-bin offset has already been corrected, and reduced-rate header/LDRO
symbols have already been divided by four.  Preamble acquisition, sync-word
validation, STO/CFO correction, and IQ inversion belong to NF-MODEM-004 and
are deliberately not inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MIN_SF: Final = 5
MAX_SF: Final = 12
MAX_PAYLOAD_BYTES: Final = 255
MAX_FRAME_SYMBOLS: Final = 4096
LDRO_MAX_DURATION_MS: Final = 16.0

# Literal from gr-lora_sdr/lib/tables.h at the pinned commit.  Keeping the
# literal makes accidental polynomial/bit-order substitutions visible.
WHITENING_SEQUENCE: Final[bytes] = bytes.fromhex(
    "ff fe fc f8 f0 e1 c2 85 0b 17 2f 5e bc 78 f1 e3 "
    "c6 8d 1a 34 68 d0 a0 40 80 01 02 04 08 11 23 47 "
    "8e 1c 38 71 e2 c4 89 12 25 4b 97 2e 5c b8 70 e0 "
    "c0 81 03 06 0c 19 32 64 c9 92 24 49 93 26 4d 9b "
    "37 6e dc b9 72 e4 c8 90 20 41 82 05 0a 15 2b 56 "
    "ad 5b b6 6d da b5 6b d6 ac 59 b2 65 cb 96 2c 58 "
    "b0 61 c3 87 0f 1f 3e 7d fb f6 ed db b7 6f de bd "
    "7a f5 eb d7 ae 5d ba 74 e8 d1 a2 44 88 10 21 43 "
    "86 0d 1b 36 6c d8 b1 63 c7 8f 1e 3c 79 f3 e7 ce "
    "9c 39 73 e6 cc 98 31 62 c5 8b 16 2d 5a b4 69 d2 "
    "a4 48 91 22 45 8a 14 29 52 a5 4a 95 2a 54 a9 53 "
    "a7 4e 9d 3b 77 ee dd bb 76 ec d9 b3 67 cf 9e 3d "
    "7b f7 ef df bf 7e fd fa f4 e9 d3 a6 4c 99 33 66 "
    "cd 9a 35 6a d4 a8 51 a3 46 8c 18 30 60 c1 83 07 "
    "0e 1d 3a 75 ea d5 aa 55 ab 57 af 5f be 7c f9 f2 "
    "e5 ca 94 28 50 a1 42 84 09 13 27 4f 9f 3f 7f"
)


class LoRaFrameError(ValueError):
    """The symbol sequence violates a bounded LoRa PHY contract."""


class LoRaIntegrityError(LoRaFrameError):
    """A complete LoRa frame has an invalid payload CRC."""


@dataclass(frozen=True)
class LoRaPhyConfig:
    """Parameters that are known before decoding a single PHY frame.

    ``cr``, ``payload_length``, and ``has_crc`` are authoritative only in
    implicit-header mode.  Explicit mode obtains those fields from its PHY
    header.  ``ldro=None`` applies gr-lora_sdr's 16 ms automatic threshold.
    """

    sf: int
    bandwidth: int
    explicit_header: bool = True
    cr: int | None = None
    payload_length: int | None = None
    has_crc: bool | None = None
    ldro: bool | None = None

    def __post_init__(self) -> None:
        if not MIN_SF <= self.sf <= MAX_SF:
            raise ValueError(f"sf must be in {MIN_SF}..{MAX_SF}")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.explicit_header:
            if self.sf < 7:
                raise ValueError("explicit-header LoRa requires sf >= 7")
        else:
            if self.cr not in range(1, 5):
                raise ValueError("implicit-header cr must be in 1..4")
            if self.payload_length is None or not 1 <= self.payload_length <= 255:
                raise ValueError("implicit-header payload_length must be in 1..255")
            if not isinstance(self.has_crc, bool):
                raise ValueError("implicit-header has_crc must be boolean")
        if self.cr is not None and self.cr not in range(1, 5):
            raise ValueError("cr must be in 1..4")
        if self.payload_length is not None and not 1 <= self.payload_length <= 255:
            raise ValueError("payload_length must be in 1..255")
        if self.has_crc is not None and not isinstance(self.has_crc, bool):
            raise ValueError("has_crc must be boolean or None")
        if self.ldro is not None and not isinstance(self.ldro, bool):
            raise ValueError("ldro must be boolean or None")

    @property
    def resolved_ldro(self) -> bool:
        if self.ldro is not None:
            return self.ldro
        duration_ms = (1 << self.sf) * 1000.0 / self.bandwidth
        return duration_ms > LDRO_MAX_DURATION_MS


@dataclass(frozen=True)
class LoRaHeader:
    payload_length: int
    cr: int
    has_crc: bool
    checksum: int | None
    explicit: bool


@dataclass(frozen=True)
class HammingDecode:
    nibble: int
    error_detected: bool
    data_bit_corrected: bool


@dataclass(frozen=True)
class LoRaDecodeResult:
    payload: bytes
    header: LoRaHeader
    crc_valid: bool | None
    ldro: bool
    consumed_symbols: int
    corrected_codewords: int
    detected_codewords: int


def gray_map_symbol(symbol: int, width: int) -> int:
    """Map a demodulated binary-index symbol to its Gray-coded value."""

    if not 1 <= width <= MAX_SF:
        raise ValueError("symbol width must be in 1..12")
    if not isinstance(symbol, int) or isinstance(symbol, bool):
        raise TypeError("symbol must be an integer")
    if not 0 <= symbol < (1 << width):
        raise LoRaFrameError(f"symbol {symbol} does not fit the {width}-bit boundary")
    return symbol ^ (symbol >> 1)


def _bits_msb(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def deinterleave_block(
    symbols: tuple[int, ...] | list[int], sf_app: int, cr: int
) -> tuple[int, ...]:
    """Undo one gr-lora_sdr diagonal-interleaver block.

    ``cr`` is 1..4 and therefore the block contains ``cr + 4`` symbols.
    The first/reduced-rate block is represented with ``cr=4`` and
    ``sf_app=sf-2``.
    """

    if not 3 <= sf_app <= MAX_SF:
        raise ValueError("sf_app must be in 3..12")
    if cr not in range(1, 5):
        raise ValueError("cr must be in 1..4")
    cw_len = cr + 4
    if len(symbols) != cw_len:
        raise LoRaFrameError(f"interleaver block requires {cw_len} symbols")
    interleaved = [_bits_msb(gray_map_symbol(symbol, sf_app), sf_app) for symbol in symbols]
    codewords = [[0] * cw_len for _ in range(sf_app)]
    for i in range(cw_len):
        for j in range(sf_app):
            codewords[(i - j - 1) % sf_app][i] = interleaved[i][j]
    return tuple(_bits_to_int(codeword) for codeword in codewords)


def hamming_decode_codeword(codeword: int, cr: int) -> HammingDecode:
    """Hard-decode the LoRa 4/(4+CR) codeword used by gr-lora_sdr."""

    if cr not in range(1, 5):
        raise ValueError("cr must be in 1..4")
    cw_len = cr + 4
    if not isinstance(codeword, int) or isinstance(codeword, bool):
        raise TypeError("codeword must be an integer")
    if not 0 <= codeword < (1 << cw_len):
        raise LoRaFrameError(f"codeword does not fit {cw_len} bits")

    bits = _bits_msb(codeword, cw_len)
    data = [bits[3], bits[2], bits[1], bits[0]]
    detected = False
    corrected = False

    if cr >= 3:
        s0 = bits[0] ^ bits[1] ^ bits[2] ^ bits[4]
        s1 = bits[1] ^ bits[2] ^ bits[3] ^ bits[5]
        s2 = bits[0] ^ bits[1] ^ bits[3] ^ bits[6]
        syndrome = s0 | (s1 << 1) | (s2 << 2)
        detected = syndrome != 0
        should_correct = cr == 3 or (sum(bits) % 2 == 1)
        if should_correct:
            data_index = {5: 3, 7: 2, 3: 1, 6: 0}.get(syndrome)
            if data_index is not None:
                data[data_index] ^= 1
                corrected = True
    elif cr == 2:
        s0 = bits[0] ^ bits[1] ^ bits[2] ^ bits[4]
        s1 = bits[1] ^ bits[2] ^ bits[3] ^ bits[5]
        detected = bool(s0 | s1)
    else:
        detected = bool(sum(bits) % 2)

    return HammingDecode(_bits_to_int(data), detected, corrected)


def _header_checksum(a: int, b: int, c: int) -> int:
    a3, a2, a1, a0 = ((a >> shift) & 1 for shift in (3, 2, 1, 0))
    b3, b2, b1, b0 = ((b >> shift) & 1 for shift in (3, 2, 1, 0))
    c3, c2, c1, c0 = ((c >> shift) & 1 for shift in (3, 2, 1, 0))
    check4 = a3 ^ a2 ^ a1 ^ a0
    check3 = a3 ^ b3 ^ b2 ^ b1 ^ c0
    check2 = a2 ^ b3 ^ b0 ^ c3 ^ c1
    check1 = a1 ^ b2 ^ b0 ^ c2 ^ c1 ^ c0
    check0 = a0 ^ b1 ^ c3 ^ c2 ^ c1 ^ c0
    return (check4 << 4) | (check3 << 3) | (check2 << 2) | (check1 << 1) | check0


def _decode_explicit_header(nibbles: list[int]) -> LoRaHeader:
    if len(nibbles) < 5:
        raise LoRaFrameError("explicit PHY header is incomplete")
    a, b, c, d, e = nibbles[:5]
    payload_length = (a << 4) | b
    cr = c >> 1
    has_crc = bool(c & 1)
    received_checksum = ((d & 1) << 4) | e
    expected_checksum = _header_checksum(a, b, c)
    if received_checksum != expected_checksum:
        raise LoRaFrameError("explicit PHY header checksum is invalid")
    if payload_length == 0:
        raise LoRaFrameError("LoRa payload cannot be empty")
    if cr not in range(1, 5):
        raise LoRaFrameError("explicit PHY header coding rate is invalid")
    return LoRaHeader(payload_length, cr, has_crc, received_checksum, True)


def lora_payload_crc(payload: bytes) -> int:
    """Return the pinned gr-lora_sdr LoRa payload CRC value."""

    if len(payload) < 2:
        raise ValueError("LoRa payload CRC requires at least two payload bytes")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("LoRa payload exceeds 255 bytes")
    crc = 0
    for byte in payload[:-2]:
        value = byte
        for _ in range(8):
            if ((crc & 0x8000) >> 8) ^ (value & 0x80):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
            value = (value << 1) & 0xFF
    return crc ^ payload[-1] ^ (payload[-2] << 8)


def _decode_codewords(codewords: tuple[int, ...], cr: int) -> tuple[list[int], int, int]:
    nibbles: list[int] = []
    corrected = 0
    detected = 0
    for codeword in codewords:
        decoded = hamming_decode_codeword(codeword, cr)
        nibbles.append(decoded.nibble)
        corrected += int(decoded.data_bit_corrected)
        detected += int(decoded.error_detected)
    return nibbles, corrected, detected


def _decode_payload_bytes(nibbles: list[int], payload_length: int) -> bytes:
    if len(nibbles) < payload_length * 2:
        raise LoRaFrameError("payload nibbles are truncated")
    payload = bytearray(payload_length)
    for index in range(payload_length):
        whitened = nibbles[2 * index] | (nibbles[2 * index + 1] << 4)
        payload[index] = whitened ^ WHITENING_SEQUENCE[index]
    return bytes(payload)


def decode_lora_symbols(
    symbols: tuple[int, ...] | list[int],
    config: LoRaPhyConfig,
    *,
    require_valid_crc: bool = True,
) -> LoRaDecodeResult:
    """Decode one exact, post-demodulation LoRa symbol frame.

    Trailing symbols are rejected because this function is a one-frame
    boundary; a streaming caller must split frames using NF-MODEM-004's source
    offsets.  CRC-bearing frames fail closed by default.
    """

    if len(symbols) > MAX_FRAME_SYMBOLS:
        raise LoRaFrameError("LoRa frame exceeds the symbol bound")
    if len(symbols) < 8:
        raise LoRaFrameError("LoRa frame is missing the first eight-symbol block")

    first_codewords = deinterleave_block(symbols[:8], config.sf - 2, 4)
    first_nibbles, corrected, detected = _decode_codewords(first_codewords, 4)

    if config.explicit_header:
        header = _decode_explicit_header(first_nibbles)
        payload_nibbles = first_nibbles[5:]
    else:
        assert config.payload_length is not None
        assert config.cr is not None
        assert config.has_crc is not None
        header = LoRaHeader(
            config.payload_length,
            config.cr,
            config.has_crc,
            None,
            False,
        )
        payload_nibbles = first_nibbles

    if header.has_crc and header.payload_length < 2:
        raise LoRaFrameError("CRC-bearing LoRa payload must contain at least two bytes")

    required_nibbles = 2 * header.payload_length + (4 if header.has_crc else 0)
    sf_app = config.sf - 2 if config.resolved_ldro else config.sf
    cw_len = header.cr + 4
    remaining = max(0, required_nibbles - len(payload_nibbles))
    block_count = (remaining + sf_app - 1) // sf_app
    required_symbols = 8 + block_count * cw_len
    if len(symbols) < required_symbols:
        raise LoRaFrameError(
            f"LoRa frame is truncated: requires {required_symbols} symbols, got {len(symbols)}"
        )
    if len(symbols) > required_symbols:
        raise LoRaFrameError(
            f"LoRa frame has trailing symbols: requires {required_symbols}, got {len(symbols)}"
        )

    cursor = 8
    for _ in range(block_count):
        block = symbols[cursor : cursor + cw_len]
        codewords = deinterleave_block(block, sf_app, header.cr)
        nibbles, block_corrected, block_detected = _decode_codewords(codewords, header.cr)
        payload_nibbles.extend(nibbles)
        corrected += block_corrected
        detected += block_detected
        cursor += cw_len

    payload_nibbles = payload_nibbles[:required_nibbles]
    payload = _decode_payload_bytes(payload_nibbles, header.payload_length)
    crc_valid: bool | None = None
    if header.has_crc:
        crc_offset = 2 * header.payload_length
        received_crc = sum(
            payload_nibbles[crc_offset + index] << (4 * index) for index in range(4)
        )
        crc_valid = received_crc == lora_payload_crc(payload)
        if require_valid_crc and not crc_valid:
            raise LoRaIntegrityError("LoRa payload CRC is invalid")

    return LoRaDecodeResult(
        payload=payload,
        header=header,
        crc_valid=crc_valid,
        ldro=config.resolved_ldro,
        consumed_symbols=required_symbols,
        corrected_codewords=corrected,
        detected_codewords=detected,
    )

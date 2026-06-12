"""CRC helpers anchored to their published check values."""

from __future__ import annotations

from gfsk_ax25 import crc


def test_crc16_ccitt_false_check_vector():
    assert crc.crc16_ccitt_false(b"123456789") == 0x29B1


def test_crc32_ieee_check_vector():
    assert crc.crc32_ieee(b"123456789") == 0xCBF43926

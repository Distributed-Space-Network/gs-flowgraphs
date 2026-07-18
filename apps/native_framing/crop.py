"""Fail-closed helpers for packet length and bounded cropping."""

from __future__ import annotations


def cc11xx_packet(packet: bytes, *, crc_bytes: int = 2, maximum: int = 258) -> bytes | None:
    """Apply the CC11xx one-byte length convention and retain its CRC."""

    if crc_bytes < 0:
        raise ValueError("crc_bytes must be non-negative")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if not packet:
        return None
    packet_length = packet[0] + 1 + crc_bytes
    if packet_length > maximum or packet_length > len(packet):
        return None
    return bytes(packet[:packet_length])


def head_tail(packet: bytes, *, head: int = 0, tail: int = 0) -> bytes | None:
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    if head + tail > len(packet):
        return None
    end = len(packet) - tail if tail else len(packet)
    return bytes(packet[head:end])


def fixed(packet: bytes, *, size: int, maximum: int) -> bytes | None:
    if size < 0 or maximum < 0 or size > maximum:
        raise ValueError("invalid fixed crop bounds")
    return bytes(packet[:size]) if len(packet) >= size else None


__all__ = ["cc11xx_packet", "fixed", "head_tail"]

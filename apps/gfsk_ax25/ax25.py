"""Minimal AX.25 v2.2 UI (unnumbered information) frame build/parse.

A UI frame is the beacon/packet workhorse for cubesats: address field
(destination + source callsigns, each 6 shifted ASCII chars + an SSID octet),
a control octet ``0x03`` (UI), a PID octet ``0xF0`` (no layer-3), then the
information field. The FCS is handled by :mod:`.hdlc`, so the bytes here are the
frame *body* (address + control + PID + info).

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

from dataclasses import dataclass

CONTROL_UI = 0x03
PID_NO_LAYER3 = 0xF0

_ADDR_LEN = 7
_MIN_BODY = 2 * _ADDR_LEN + 2  # two addresses + control + pid


def _encode_callsign(call: str, ssid: int, *, last: bool) -> bytes:
    call = call.upper()[:6].ljust(6)
    out = bytearray(ch << 1 for ch in call.encode("ascii"))
    # SSID octet: bit0 = address-extension (1 on the last address), bits1-4 =
    # SSID, bits5-6 reserved (set to 1), bit7 = command/response (left 0).
    ssid_octet = 0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0)
    out.append(ssid_octet)
    return bytes(out)


def _decode_callsign(octets: bytes) -> tuple[str, int, bool]:
    call = "".join(chr(b >> 1) for b in octets[:6]).rstrip()
    ssid = (octets[6] >> 1) & 0x0F
    last = bool(octets[6] & 0x01)
    return call, ssid, last


@dataclass(frozen=True)
class Ui:
    dest: str
    src: str
    info: bytes
    dest_ssid: int = 0
    src_ssid: int = 0


def encode_ui(
    *,
    dest: str,
    src: str,
    info: bytes,
    dest_ssid: int = 0,
    src_ssid: int = 0,
) -> bytes:
    """Build a UI frame body (no FCS) ready for :func:`hdlc.frame`."""
    addr = _encode_callsign(dest, dest_ssid, last=False) + _encode_callsign(
        src, src_ssid, last=True
    )
    return addr + bytes((CONTROL_UI, PID_NO_LAYER3)) + info


def decode_ui(body: bytes) -> Ui | None:
    """Parse a UI frame body (FCS already stripped); None if not a UI frame."""
    if len(body) < _MIN_BODY:
        return None
    dest, dest_ssid, _ = _decode_callsign(body[0:7])
    src, src_ssid, _ = _decode_callsign(body[7:14])
    control = body[14]
    pid = body[15]
    if control != CONTROL_UI or pid != PID_NO_LAYER3:
        return None
    return Ui(dest=dest, src=src, info=body[16:], dest_ssid=dest_ssid, src_ssid=src_ssid)


__all__ = ["CONTROL_UI", "PID_NO_LAYER3", "Ui", "decode_ui", "encode_ui"]

"""KISS and SLIP TNC framing (docs/08 Tier 3 — uplink/relay framings not in gr-satellites).

Byte-oriented delimiter framing used by TNCs and packet radio: frames are bracketed by an END
byte (0xC0) with an escape mechanism so the delimiter can appear in the payload. KISS (used to
shuttle frames to/from a TNC) additionally carries a leading command/port byte; SLIP is the same
delimiting without it. Both are exact + reversible, so they are unit-tested by round trip
(including payloads that contain the reserved bytes). numpy/stdlib-only.
"""
from __future__ import annotations

FEND = 0xC0   # frame delimiter (SLIP END)
FESC = 0xDB   # escape (SLIP ESC)
TFEND = 0xDC  # transposed frame-end
TFESC = 0xDD  # transposed escape


def _escape(frame: bytes) -> bytes:
    out = bytearray()
    for b in frame:
        if b == FEND:
            out += bytes([FESC, TFEND])
        elif b == FESC:
            out += bytes([FESC, TFESC])
        else:
            out.append(b)
    return bytes(out)


def _unescape(payload: bytes, *, strict: bool = False) -> tuple[bytes | None, int]:
    """Undo FESC escaping; returns ``(payload, valid_escape_count)``. In ``strict`` mode an
    INVALID escape (FESC followed by anything but TFEND/TFESC, or a trailing FESC) is a
    protocol violation → ``(None, 0)`` (reject the chunk) — a real structural constraint that
    rejects most noise chunks containing 0xDB. The escape count lets callers treat a chunk with
    ≥1 VALID escape pair as deliberate framing (strongest structural evidence available)."""
    out = bytearray()
    escapes = 0
    i = 0
    while i < len(payload):
        b = payload[i]
        if b == FESC:
            nxt = payload[i + 1] if i + 1 < len(payload) else None
            if nxt == TFEND:
                out.append(FEND)
                escapes += 1
            elif nxt == TFESC:
                out.append(FESC)
                escapes += 1
            elif strict:
                return None, 0  # invalid escape — not a KISS/SLIP frame
            elif nxt is not None:
                out.append(nxt)
            else:
                out.append(b)
            i += 2 if nxt is not None else 1
        else:
            out.append(b)
            i += 1
    return bytes(out), escapes


def kiss_encode(frame: bytes, *, command: int = 0, port: int = 0) -> bytes:
    """Wrap ``frame`` in a KISS frame: FEND, command/port byte, escaped payload, FEND."""
    type_byte = ((port & 0x0F) << 4) | (command & 0x0F)
    return bytes([FEND, type_byte]) + _escape(bytes(frame)) + bytes([FEND])


_STRICT_MIN_PAYLOAD = 8   # KISS: noise chunks between chance FENDs are mostly short garbage
_SLIP_STRICT_MIN_PAYLOAD = 16  # SLIP has no type byte to gate on — compensate with length


def kiss_decode(stream: bytes, *, strict: bool = False) -> list[bytes]:
    """Extract KISS frame payloads from ``stream`` (command/port byte stripped). Empty frames and
    trailing partials are ignored.

    ``strict`` is for DEMODULATED bitstreams (vs a clean TNC pipe): KISS carries NO checksum, so
    on a noisy stream any two chance 0xC0 bytes would bracket a garbage "frame". Strict mode
    keeps only segments FEND-bracketed on BOTH sides, with a data-frame type byte (low nibble 0)
    and at least ``_STRICT_MIN_PAYLOAD`` payload bytes. This shrinks noise acceptance by >10x;
    it cannot eliminate it (an unchecksummed protocol has no integrity to verify)."""
    chunks = bytes(stream).split(bytes([FEND]))
    if strict:
        chunks = chunks[1:-1] if len(chunks) >= 2 else []  # bracketed-on-both-sides only
    out: list[bytes] = []
    for chunk in chunks:
        if not chunk:
            continue
        if strict and (chunk[0] & 0x0F) != 0:
            continue  # only type-0 (data) frames survive strict mode
        payload, escapes = _unescape(chunk[1:], strict=strict)  # drop the command/port byte
        if payload is None:  # invalid escape sequence — protocol violation
            continue
        if strict and len(payload) < _STRICT_MIN_PAYLOAD:
            continue
        if strict and escapes == 0 and len(set(payload)) < 2:
            continue  # constant idle fill (0x00 runs between frames) — but a payload built
            # from VALID escape pairs (e.g. all-FEND) is deliberate framing, keep it
        if payload:
            out.append(payload)
    return out


def slip_encode(frame: bytes) -> bytes:
    """Wrap ``frame`` in a SLIP frame: END, escaped payload, END (no command byte)."""
    return bytes([FEND]) + _escape(bytes(frame)) + bytes([FEND])


def slip_decode(stream: bytes, *, strict: bool = False) -> list[bytes]:
    """Extract SLIP frame payloads from ``stream``. ``strict`` (for demodulated bitstreams):
    bracketed-on-both-sides + minimum payload length only — SLIP has no type byte to gate on."""
    chunks = bytes(stream).split(bytes([FEND]))
    if strict:
        chunks = chunks[1:-1] if len(chunks) >= 2 else []
    out: list[bytes] = []
    for chunk in chunks:
        if not chunk:
            continue
        payload, escapes = _unescape(chunk, strict=strict)
        if payload is None:  # invalid escape sequence — protocol violation
            continue
        if strict and len(payload) < _SLIP_STRICT_MIN_PAYLOAD:
            continue  # SLIP has no type byte — the longer floor is its main noise gate
        if strict and escapes == 0 and len(set(payload)) < 2:
            continue  # constant idle fill — but valid-escape payloads are deliberate framing
        if payload:
            out.append(payload)
    return out

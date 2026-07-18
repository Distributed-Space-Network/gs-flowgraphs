"""Bounded SDSat LoRa relay-text telemetry preview.

Adapted on 2026-07-18 from ``ksy/sdsat.ksy`` and generated ``sdsat.py`` in
tinygs-decoders commit ``6b82a7f610349c2e46bcd97a0df38f9bdca1daf6``
under the project's explicit MIT treatment and attribution authorization.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import unicodedata

from native_telemetry.types import FrameContext, ParserPreview

MAGIC = b"SDSAT,LORA ACTIVE:"
MAX_PACKET_BYTES = 255
MAX_RELAY_BYTES = MAX_PACKET_BYTES - len(MAGIC)
_BIDI_CONTROLS = frozenset(
    {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI", "BN"}
)


def _safe_text(text: str) -> str:
    output: list[str] = []
    for char in text:
        if unicodedata.category(char).startswith("C") or (
            unicodedata.bidirectional(char) in _BIDI_CONTROLS
        ):
            codepoint = ord(char)
            output.append(
                f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            )
        else:
            output.append(char)
    return "".join(output)


def parse_sdsat(context: FrameContext) -> ParserPreview:
    """Validate SDSat identity and decode a bounded, display-safe UTF-8 relay."""

    data = context.payload
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"SDSat packet exceeds {MAX_PACKET_BYTES} bytes")
    if len(data) < len(MAGIC) or data[: len(MAGIC)] != MAGIC:
        raise ValueError("SDSat relay magic does not match")
    relay = data[len(MAGIC) :]
    if len(relay) > MAX_RELAY_BYTES:
        raise ValueError(f"SDSat relay exceeds {MAX_RELAY_BYTES} bytes")
    try:
        text = relay.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SDSat relay is not valid UTF-8") from exc
    return ParserPreview(
        status="ok",
        values={
            "kind": "relay_text",
            "magic": MAGIC.decode("ascii"),
            "relay_text": _safe_text(text),
            "relay_utf8_bytes": len(relay),
            "relay_hex": relay.hex(),
        },
    )


__all__ = ["MAGIC", "MAX_PACKET_BYTES", "MAX_RELAY_BYTES", "parse_sdsat"]

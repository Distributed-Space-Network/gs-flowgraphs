"""Schema-free, bounded printable UTF-8 string extraction."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator

from native_telemetry.types import FrameContext, ParserPreview

MIN_RUN_CHARS = 4
MAX_INPUT_BYTES = 65_536
MAX_STRINGS = 32
MAX_STRING_CHARS = 256
MAX_RENDERED_BYTES = 4_096

_BIDI_CONTROLS = frozenset(
    {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI", "BN"}
)


def _utf8_codepoints(data: bytes) -> Iterator[tuple[int, int, str | None]]:
    """Yield exact byte spans and decoded characters; invalid bytes yield ``None``."""

    offset = 0
    while offset < len(data):
        lead = data[offset]
        if lead < 0x80:
            yield offset, offset + 1, chr(lead)
            offset += 1
            continue
        if 0xC2 <= lead <= 0xDF:
            width = 2
        elif 0xE0 <= lead <= 0xEF:
            width = 3
        elif 0xF0 <= lead <= 0xF4:
            width = 4
        else:
            yield offset, offset + 1, None
            offset += 1
            continue
        end = offset + width
        try:
            char = data[offset:end].decode("utf-8")
        except UnicodeDecodeError:
            yield offset, offset + 1, None
            offset += 1
            continue
        yield offset, end, char
        offset = end


def _safe_printable(char: str | None) -> bool:
    if char is None or not char.isprintable():
        return False
    if unicodedata.category(char).startswith("C"):
        return False
    return unicodedata.bidirectional(char) not in _BIDI_CONTROLS


def _truncate_utf8(text: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    used = 0
    output: list[str] = []
    for char in text:
        width = len(char.encode("utf-8"))
        if used + width > byte_limit:
            break
        output.append(char)
        used += width
    return "".join(output)


def extract_strings(context: FrameContext) -> ParserPreview:
    """Extract printable runs with exact byte offsets and deterministic limits."""

    data = context.payload[:MAX_INPUT_BYTES]
    runs: list[dict[str, object]] = []
    aggregate_bytes = 0
    start: int | None = None
    end = 0
    chars: list[str] = []

    def finish_run() -> None:
        nonlocal aggregate_bytes, start, end, chars
        if start is None:
            return
        original = "".join(chars)
        if len(original) >= MIN_RUN_CHARS and any(not char.isspace() for char in original):
            rendered = original[:MAX_STRING_CHARS]
            rendered = _truncate_utf8(rendered, MAX_RENDERED_BYTES - aggregate_bytes)
            if rendered and len(runs) < MAX_STRINGS:
                rendered_bytes = len(rendered.encode("utf-8"))
                runs.append(
                    {
                        "byte_start": start,
                        "byte_end": end,
                        "text": rendered,
                        "truncated": rendered != original,
                    }
                )
                aggregate_bytes += rendered_bytes
        start = None
        end = 0
        chars = []

    for byte_start, byte_end, char in _utf8_codepoints(data):
        if _safe_printable(char):
            if start is None:
                start = byte_start
            end = byte_end
            chars.append(char or "")
            continue
        finish_run()
        if len(runs) >= MAX_STRINGS or aggregate_bytes >= MAX_RENDERED_BYTES:
            break
    finish_run()

    values: dict[str, object] = {
        "strings": runs,
        "rendered": _truncate_utf8(
            " | ".join(str(run["text"]) for run in runs), MAX_RENDERED_BYTES
        ),
        "input_truncated": len(context.payload) > MAX_INPUT_BYTES,
    }
    return ParserPreview(status="ok" if runs else "no_preview", values=values)

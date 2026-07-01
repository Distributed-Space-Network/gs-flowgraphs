"""CW / Morse code — keying encode + timing decode (docs/08 Tier 3).

International Morse at the timing level: a dot is 1 unit, a dash 3, the intra-character gap 1, the
inter-character gap 3, the word gap 7 (ITU-R M.1677). :func:`encode` renders text to an on/off
timeline (one sample per unit by default); :func:`decode` measures run lengths and reconstructs
text — tolerant of the exact unit length (it classifies by ratio, so a CW envelope sampled at any
rate decodes once reduced to on/off units). numpy/stdlib-only, fully unit-testable.
"""
from __future__ import annotations

import numpy as np

_CODE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
    "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
    "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
    "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "-": "-....-", "=": "-...-",
}
_DECODE = {v: k for k, v in _CODE.items()}

DOT, DASH = 1, 3
_GAP_INTRA, _GAP_CHAR, _GAP_WORD = 1, 3, 7


def encode(text: str, *, unit: int = 1) -> np.ndarray:
    """Text → on/off timeline (uint8, 1 = key-down), ``unit`` samples per Morse time unit. Unknown
    characters are skipped; runs of spaces collapse to a single word gap."""
    on: list[int] = []
    tokens = [w for w in text.upper().split(" ") if w != ""]
    for wi, word in enumerate(tokens):
        if wi:
            on += [0] * (_GAP_WORD * unit)
        letters = [c for c in word if c in _CODE]
        for li, ch in enumerate(letters):
            if li:
                on += [0] * (_GAP_CHAR * unit)
            for ei, el in enumerate(_CODE[ch]):
                if ei:
                    on += [0] * (_GAP_INTRA * unit)
                on += [1] * ((DASH if el == "-" else DOT) * unit)
    return np.array(on, dtype=np.uint8)


def _runs(timeline: np.ndarray):
    if timeline.size == 0:
        return []
    change = np.nonzero(np.diff(timeline))[0] + 1
    bounds = [0, *change.tolist(), timeline.size]
    return [(int(timeline[bounds[i]]), bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]


def decode(timeline, *, unit: float | None = None) -> str:
    """On/off timeline → text. If ``unit`` is not given it is estimated as the shortest on-run
    (one dot). On-runs classify dot/dash at 2 units; off-runs classify intra/char/word at 2 and 5
    units — the standard 1/3/7 spacing with margins.

    Ambiguity: a message whose only element is a single dash (e.g. ``"T"``) can't be told from a
    single dot (``"E"``) without a time reference — the auto estimate treats the lone element as a
    dot. Pass ``unit=`` (from the known WPM) to disambiguate; any message containing a dot estimates
    correctly."""
    arr = np.asarray(timeline, dtype=np.uint8)
    runs = _runs(arr)
    if not runs:
        return ""
    on_lens = [ln for v, ln in runs if v == 1]
    if not on_lens:
        return ""
    u = float(unit) if unit else float(min(on_lens))
    out: list[str] = []
    symbol = ""
    for v, ln in runs:
        if v == 1:
            symbol += "-" if ln / u >= 2.0 else "."
        elif ln / u >= 5.0:        # word gap: flush the symbol, then a space
            out.append(_DECODE.get(symbol, ""))
            out.append(" ")
            symbol = ""
        elif ln / u >= 2.0:        # inter-character gap: flush the symbol
            out.append(_DECODE.get(symbol, ""))
            symbol = ""
        # else intra-character gap → keep building the current symbol
    if symbol:
        out.append(_DECODE.get(symbol, ""))
    return "".join(out)

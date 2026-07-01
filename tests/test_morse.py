"""docs/08 Tier 3 — CW / Morse keying + timing decode."""
from __future__ import annotations

import numpy as np

from gfsk_ax25 import morse


def test_roundtrip_word():
    assert morse.decode(morse.encode("HELLO")) == "HELLO"


def test_roundtrip_with_word_gaps():
    text = "SOS DE N0CALL"
    assert morse.decode(morse.encode(text)) == text


def test_roundtrip_scaled_unit():
    # a CW envelope sampled at 5 samples/unit still decodes (classification is by ratio)
    text = "CQ 73"
    assert morse.decode(morse.encode(text, unit=5)) == text


def test_known_symbol_patterns():
    # 'A' = dot-dash: one 1-unit on, 1-unit gap, 3-unit on
    a = morse.encode("A")
    np.testing.assert_array_equal(a, np.array([1, 0, 1, 1, 1], dtype=np.uint8))


def test_empty_and_unknown():
    assert morse.decode(np.zeros(0, dtype=np.uint8)) == ""
    assert morse.decode(morse.encode("")) == ""
    assert morse.decode(morse.encode("A@B")) == "AB"  # unknown char skipped

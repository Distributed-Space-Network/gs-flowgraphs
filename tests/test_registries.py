"""Phase 0 (docs/08) — modem / framing / fec registry contract.

Locks the pluggable-registry skeleton the universal modem+framing build stands on: the modem
registry advertises the modulation families it can demodulate, the framing registry deframes
(and auto-detects) the link layers, and the FEC registry is a documented, empty skeleton. The
GNU Radio demod chains themselves are exercised on the bench; these tests cover the numpy-only
dispatch + the AX.25 deframe path end to end.
"""
from __future__ import annotations

import fec
import framings
import modem
import numpy as np

from gfsk_ax25 import ax25, framing


def test_modem_advertises_all_current_families():
    assert modem.demod_families() == {
        "gfsk", "fsk", "gmsk", "msk", "bpsk", "qpsk", "psk", "afsk"
    }


def test_framing_registry_lists_the_link_layers():
    assert framings.known_framings() == ("ax25", "endurosat")


def test_fec_registry_is_an_empty_skeleton():
    assert fec.known_codes() == ()


def test_deframe_empty_and_noise_return_no_match():
    assert framings.deframe(np.zeros(0, dtype=np.uint8)) == ([], None)
    noise = np.random.default_rng(0).integers(0, 2, 8192).astype(np.uint8)
    _, matched = framings.deframe(noise, "ax25")
    assert matched is None


def test_deframe_ax25_roundtrip_through_registry():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"telemetry frame payload")
    bits = framing.encode(body, preamble_flags=16)
    frames, matched = framings.deframe(bits, "ax25")  # backend told us the framing
    assert matched == "ax25"
    assert body in frames


def test_deframe_autodetects_ax25_when_framing_unknown():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"auto")
    bits = framing.encode(body, preamble_flags=16)
    frames, matched = framings.deframe(bits)  # no hint → try all known, report which matched
    assert matched == "ax25"
    assert body in frames

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


def test_modem_advertises_the_tier1_and_tier2_families():
    fams = modem.demod_families()
    # Tier-1 FSK + PSK + AFSK are all recognized …
    assert {"gfsk", "fsk", "gmsk", "msk", "cpfsk"} <= fams
    assert {"bpsk", "dbpsk", "qpsk", "dqpsk", "oqpsk", "8psk", "psk", "afsk"} <= fams
    # … and the Tier-2 keys classify (they route elsewhere, not built here).
    assert {"qam16", "qam256", "apsk32", "ofdm", "dvbs2"} <= fams
    # TX modulator families cover the Tier-1 set (Tier-2 modulators come later).
    assert {"gfsk", "bpsk", "qpsk", "afsk"} <= modem.mod_families()


def test_framing_registry_lists_local_and_grsatellites_layers():
    assert framings.local_framings() == ("ax25", "endurosat", "argos", "ccsds_tm")
    known = framings.known_framings()
    assert "ax25" in known and "argos" in known and "ccsds_tm" in known
    # the gr-satellites vocabulary is advertised (reused via synthetic SatYAML, decoded upstream).
    for f in ("USP", "AX100 ASM+Golay", "CCSDS Concatenated", "Mobitex"):
        assert f in known


def test_fec_registry_advertises_codes_and_implements_the_numpy_ones():
    codes = fec.known_codes()
    for c in ("ccsds_randomizer", "crc16", "crc32", "asm", "reed_solomon", "golay"):
        assert c in codes
    impl = fec.implemented_codes()
    assert impl == ("ccsds_randomizer", "crc16", "crc32", "asm", "reed_solomon")
    assert "golay" not in impl and "ldpc" not in impl  # bench / gr-satellites


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

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
    assert framings.local_framings() == ("ax25", "endurosat", "ccsds_tm", "kiss")
    assert framings.advertised_local_framings() == (
        "AX.25",
        "EnduroSat",
        "ccsds_tm",
        "KISS",
    )
    assert tuple(
        framings.normalize_framing(label)
        for label in framings.advertised_local_framings()
    ) == framings.local_framings()
    known = framings.known_framings()
    assert "ax25" in known and "ccsds_tm" in known and "kiss" in known
    assert "argos" not in known  # placeholder sync -> not advertised until bench-confirmed
    assert "slip" not in known   # byte-pipe codec only -> ungateable on demod bitstreams
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


def test_grsat_deframer_plan_matches_dotted_ax100_spelling():
    # BUG: the builder tested "ax100" in f but SatNOGS spells it "AX.100" (with a dot), so AX.100
    # birds (BRO/D-Orbit/SITRO) built NO deframer even with GS_GRSAT_LIVE=1. Dots are now stripped.
    assert framings.grsat_deframer_plan("AX.100 Mode 5") == [("ax100", "ASM")]
    assert framings.grsat_deframer_plan("AX100 ASM+Golay") == [("ax100", "ASM")]
    assert framings.grsat_deframer_plan("FSK AX.100 Mode 6") == [("ax100", "RS")]
    assert framings.grsat_deframer_plan("AX100 Reed Solomon") == [("ax100", "RS")]
    assert framings.grsat_deframer_plan("USP") == [("usp",)]
    assert framings.grsat_deframer_plan("AX.25 G3RUH") == [("ax25", True)]
    assert framings.grsat_deframer_plan("ax25") == [("ax25", False), ("ax25", True)]
    assert framings.grsat_deframer_plan("endurosat") == [("endurosat",)]
    assert framings.grsat_deframer_plan("") == []  # unknown → numpy/record-only carries it


def test_additive_grsat_plan_dedupes_components_and_retains_framing_identity():
    assert framings.additive_grsat_deframer_plan(
        ["AX.25", "EnduroSat", "AX.25", "USP"]
    ) == [
        ("AX.25", ("ax25", False)),
        ("AX.25", ("ax25", True)),
        ("EnduroSat", ("endurosat",)),
        ("USP", ("usp",)),
    ]
    assert len(framings.additive_grsat_deframer_plan(["AX.25", "AX.25 G3RUH"])) == 2


def test_ax25_address_check_rejects_crc16_false_positives():
    # AX.25's FCS is 16-bit → noise passes it ~1/65536; the address-field check rejects those, so a
    # decoded "frame" is trustworthy. Bytes are real bench data (cmd_101 IPoS pass).
    real = bytes.fromhex("86a240404040e0aea264b0969ee103f04d1101")  # CQ <- WQ2XKO (real SatNOGS)
    junk = bytes.fromhex("a99892ab3e26b6c58c60aedea984cd8b")  # bench CRC-16 false positive
    assert framings._valid_ax25_address(real)
    assert not framings._valid_ax25_address(junk)
    # A genuine round-trip frame is NOT rejected (regression guard).
    assert framings._valid_ax25_address(ax25.encode_ui(dest="DSN", src="ISS", info=b"x"))

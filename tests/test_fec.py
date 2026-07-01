"""docs/08 Tier 1 — FEC numpy primitives, checked against PUBLISHED reference vectors.

Rigorous means non-circular: the CCSDS randomizer is checked against the standard PN sequence,
the CRCs against their canonical check values, and the ASM against the known 0x1ACFFC1D marker —
not just self-round-tripped.
"""
from __future__ import annotations

import fec
import numpy as np


def test_ccsds_randomizer_matches_the_published_pn_sequence():
    # CCSDS 131.0-B pseudo-randomizer applied to all-zeros yields the PN sequence itself.
    zeros = bytes(8)
    pn = fec.ccsds_randomize(zeros)
    assert pn == bytes([0xFF, 0x48, 0x0E, 0xC0, 0x9A, 0x0D, 0x70, 0xBC])


def test_ccsds_randomizer_is_involutive():
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, 220).astype(np.uint8).tobytes()
    assert fec.ccsds_derandomize(fec.ccsds_randomize(data)) == data


def test_crc_check_values():
    assert fec.crc16_ccitt(b"123456789") == 0x29B1
    assert fec.crc32(b"123456789") == 0xCBF43926


def test_asm_locates_the_ccsds_marker_and_reports_frame_start():
    payload = np.random.default_rng(1).integers(0, 2, 200).astype(np.uint8)
    asm = np.array([(fec.ASM_CCSDS >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
    lead = np.random.default_rng(2).integers(0, 2, 40).astype(np.uint8)
    stream = np.concatenate([lead, asm, payload])
    idx = fec.find_asm(stream)
    assert idx == len(lead) + 32  # index just AFTER the marker = frame start
    np.testing.assert_array_equal(stream[idx:idx + 200], payload)


def test_asm_absent_returns_minus_one():
    noise = np.zeros(64, dtype=np.uint8)  # all zeros never contains 0x1ACFFC1D
    assert fec.find_asm(noise) == -1


def test_catalog_split():
    assert set(fec.implemented_codes()) <= set(fec.known_codes())
    assert "reed_solomon" in fec.known_codes()  # declared (gr-satellites), not numpy here
    assert "reed_solomon" not in fec.implemented_codes()

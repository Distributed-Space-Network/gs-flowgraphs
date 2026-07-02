"""docs/08 Phase 6 — systematic per-family validation + registry coverage.

Two guarantees:
  1. **Round-trip** every framing/FEC that is implementable in-process (frame→deframe recovers the
     body; encode→decode recovers the data) — the numpy path, no GNU Radio.
  2. **Coverage** — every capability the registries *advertise* is either exercised here or
     explicitly accounted for as bench/gr-satellites, so nothing is silently unimplemented.

GNU-Radio modem chains (FSK/PSK/QAM/OFDM/DVB-S2, analog) are validated on the bench against real
birds; that is out of CI scope and asserted here only as "recognized + routed".
"""
from __future__ import annotations

import compose
import fec
import framings
import modem
import numpy as np
import pytest

from gfsk_ax25 import argos, ax25, ccsds, endurosat_link, kiss, morse, ook
from gfsk_ax25 import framing as ax25_framing

_PAYLOAD = b"PHASE6-validation-body-0123456789"


# ── 1. Framing round-trips (registry path where the default params fit) ──────────────────────
def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def test_framing_ax25_roundtrip():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=_PAYLOAD)
    frames, matched = framings.deframe(ax25_framing.encode(body, preamble_flags=16), "ax25")
    assert matched == "ax25" and body in frames


def test_framing_endurosat_roundtrip():
    frames, matched = framings.deframe(endurosat_link.frame_bits(_PAYLOAD), "endurosat")
    assert matched == "endurosat" and _PAYLOAD in frames


def test_framing_ccsds_tm_roundtrip():
    h = ccsds.TMHeader(0, 0x2AB, 3, 0, 1, 2, 0, 0, 0)
    bits = ccsds.build_tm_frame(h, _PAYLOAD, frame_len=223)
    frames, matched = framings.deframe(bits, "ccsds_tm")
    assert matched == "ccsds_tm" and frames and frames[0][6:6 + len(_PAYLOAD)] == _PAYLOAD


def test_framing_kiss_roundtrip():
    frames, matched = framings.deframe(_bits(kiss.kiss_encode(_PAYLOAD)), "kiss")
    assert matched == "kiss" and frames == [_PAYLOAD]


def test_framing_slip_roundtrip_module_level():
    # SLIP is a byte-pipe codec (uplink/relay TNC), NOT a demodulated-bitstream framing —
    # with no checksum and no type byte it is ungateable on noise, so the registry does not
    # wire it (docs/10 §10). Round-trip at the module level, its real interface.
    assert kiss.slip_decode(kiss.slip_encode(_PAYLOAD)) == [_PAYLOAD]


def test_framing_argos_roundtrip_module_level():
    # Registry argos uses a PLACEHOLDER 8-bit sync (bench-confirm); round-trip at the module level
    # with an explicit long sync, which is what a real deployment supplies.
    sync, sync_bits, msg = 0xABCDEF, 24, 0x1ABCD
    s = [(sync >> (sync_bits - 1 - i)) & 1 for i in range(sync_bits)]
    code = argos.bch3121_encode(msg)
    c = [(code >> (31 - 1 - i)) & 1 for i in range(31)]
    frames = argos.deframe(np.array(s + c, dtype=np.uint8), sync=sync, sync_bits=sync_bits)
    assert frames and int.from_bytes(frames[0][:3], "big") == msg


# argos round-trips at MODULE level (explicit sync; not registry-wired until benched);
# slip round-trips at MODULE level (byte-pipe codec; ungateable on demodulated bitstreams).
_ROUNDTRIPPED_FRAMINGS = {"ax25", "endurosat", "ccsds_tm", "kiss"}


def test_every_local_framing_is_round_trip_validated():
    # Coverage guard: if a new local framing is added it MUST get a round-trip test above.
    assert set(framings.local_framings()) == _ROUNDTRIPPED_FRAMINGS


def test_known_framings_partition_into_local_or_grsatellites():
    known = set(framings.known_framings())
    accounted = set(framings.local_framings()) | set(framings.grsatellites_framings())
    assert known == accounted  # nothing advertised that isn't local or gr-satellites


# ── 2. FEC round-trips / reference vectors ───────────────────────────────────────────────────
def test_fec_reed_solomon_roundtrip_and_correction():
    data = bytes(range(223))
    cw = bytearray(fec.reed_solomon_encode(data))
    cw[7] ^= 0xFF
    cw[240] ^= 0x33
    assert fec.reed_solomon_decode(bytes(cw)) == data


def test_fec_randomizer_crc_asm_reference():
    assert fec.ccsds_randomize(bytes(4)) == bytes([0xFF, 0x48, 0x0E, 0xC0])
    assert fec.ccsds_derandomize(fec.ccsds_randomize(b"abc")) == b"abc"
    assert fec.crc16_ccitt(b"123456789") == 0x29B1
    assert fec.crc32(b"123456789") == 0xCBF43926
    asm = np.array([(fec.ASM_CCSDS >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
    assert fec.find_asm(np.concatenate([np.zeros(8, np.uint8), asm])) == 8 + 32


def test_every_implemented_fec_code_is_exercised_or_declared():
    impl = set(fec.implemented_codes())
    assert impl == {"ccsds_randomizer", "crc16", "crc32", "asm", "reed_solomon"}
    # the rest of the catalog is declared bench / gr-satellites (not runnable in CI)
    assert set(fec.known_codes()) - impl >= {"viterbi", "ldpc", "golay"}


# ── 3. Modulation coverage + numpy modem round-trips ─────────────────────────────────────────
def test_every_advertised_modulation_classifies():
    for kind in modem.demod_families() | modem.mod_families():
        assert modem.modulation_spec(kind) is not None, kind


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_each_tier_has_representatives(tier):
    reps = {1: "gfsk", 2: "qam16", 3: "ook"}
    assert modem.modulation_spec(reps[tier]).tier == tier


def test_numpy_modulation_roundtrips():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(ook.demodulate(ook.modulate(bits, sps=8), sps=8), bits)
    assert morse.decode(morse.encode("PHASE SIX")) == "PHASE SIX"


# ── 4. Composer coverage — representative rfLinks map to the right path ───────────────────────
def test_composer_covers_the_decode_paths():
    ours = compose.plan_decode({"modulation": "bpsk", "symbol_rate_hz": 2e6, "framing": "ccsds_tm"})
    assert ours.our_engine
    syn = compose.plan_decode({"modulation": "gfsk", "symbol_rate_hz": 4800, "framing": "USP"})
    assert syn.grsatellites and not syn.our_engine
    cat = compose.plan_decode({}, catalogued=True)
    assert cat.grsatellites
    rec = compose.plan_decode({"modulation": "qam64", "symbol_rate_hz": 2e7, "framing": "unknown"})
    assert not rec.decodable  # record-only

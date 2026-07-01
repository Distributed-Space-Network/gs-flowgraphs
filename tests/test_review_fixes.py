"""docs/10 (Document J) review fixes — regression locks for the P0/P1/P2 batch.

Each test names the finding it locks. GNU-Radio-side fixes (pmt import, TED, null sinks,
generic_mod, packbits TX) are bench-validated; what's testable here is the pure layer:
framing-vocabulary normalization (P0-2), can_synthesize validation (#13), KISS/SLIP bit
phase (P2-16), argos explicit sync (P2-17), compose coercion (P2-18), RS erasures (P2-19),
and the OOK noise gate (P2-20).
"""
from __future__ import annotations

import compose
import framings
import grsat_synth
import numpy as np
import pytest

from gfsk_ax25 import argos, ax25, kiss, ook
from gfsk_ax25 import framing as ax25_framing
from gfsk_ax25.reedsolomon import RSCodec


# ── P0-2: the framing registry is the single normalization point ─────────────────────────────
@pytest.mark.parametrize(
    ("label", "local"),
    [
        ("AX.25 G3RUH", "ax25"), ("AX.25", "ax25"), ("ax25", "ax25"), ("APRS", "ax25"),
        ("EnduroSat", "endurosat"), ("AirMAC", "endurosat"),
        ("KISS", "kiss"), ("SLIP", "slip"),
        ("ccsds_tm", "ccsds_tm"), ("CCSDS", "ccsds_tm"),
        ("Argos PTT-A2", "argos"),
        # NOT local: AX.100 is not AX.25; FX.25 needs its RS layer; coded CCSDS variants;
        # gr-satellites-only vocabularies; garbage.
        ("AX100 ASM+Golay", None), ("FX.25 NRZI", None), ("CCSDS Concatenated", None),
        ("USP", None), ("Mobitex", None), ("something-else", None), ("", None), (None, None),
    ],
)
def test_normalize_framing(label, local):
    assert framings.normalize_framing(label) == local


def test_deframe_accepts_verbatim_satyaml_label():
    # The exact end-to-end failure from the review: a backend label ("AX.25 G3RUH") must reach
    # the local AX.25 deframer, not fall through as unknown.
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"verbatim-label")
    bits = ax25_framing.encode(body, preamble_flags=16)
    frames, matched = framings.deframe(bits, "AX.25 G3RUH")
    assert matched == "ax25" and body in frames


def test_deframe_upstream_label_returns_nothing_not_wrong_layer():
    # An AX100 label must NOT be forced through the AX.25 deframer (it is decoded upstream).
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"x")
    bits = ax25_framing.encode(body, preamble_flags=16)
    frames, matched = framings.deframe(bits, "AX100 ASM+Golay")
    assert frames == [] and matched is None


# ── #13: can_synthesize validates the gr-satellites vocabulary ────────────────────────────────
def test_can_synthesize_accepts_satyaml_vocabulary_and_rejects_local_tokens():
    assert grsat_synth.can_synthesize("gfsk", 9600, "AX.25 G3RUH")
    assert grsat_synth.can_synthesize("fsk", 4800, "USP")
    assert grsat_synth.can_synthesize("bpsk", 1200, "AX100 Reed Solomon")  # needle family
    # local-only tokens / garbage are NOT gr-satellites vocabulary
    assert not grsat_synth.can_synthesize("gfsk", 9600, "ax25")
    assert not grsat_synth.can_synthesize("gfsk", 9600, "endurosat")
    assert not grsat_synth.can_synthesize("gfsk", 9600, "airmac")
    assert not grsat_synth.can_synthesize("gfsk", 9600, None)
    # modulation/baud gates unchanged
    assert not grsat_synth.can_synthesize("qam16", 9600, "USP")
    assert not grsat_synth.can_synthesize("gfsk", 0, "USP")


# ── P2-18: compose coerces non-str framing instead of crashing ────────────────────────────────
def test_plan_decode_survives_non_string_framing():
    plan = compose.plan_decode({"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": 123})
    assert not plan.our_framing  # "123" normalizes to nothing; no AttributeError


# ── P2-16: KISS/SLIP tolerate an arbitrary bit phase ─────────────────────────────────────────
@pytest.mark.parametrize("offset", [0, 1, 3, 7])
def test_kiss_deframes_at_any_bit_phase(offset):
    wire = kiss.kiss_encode(b"phase-test-\xc0\xdb")
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8))
    shifted = np.concatenate([np.zeros(offset, dtype=np.uint8), bits])
    frames, matched = framings.deframe(shifted, "kiss")
    assert matched == "kiss" and frames == [b"phase-test-\xc0\xdb"]


# ── P2-17: argos sync is explicit (no dangerous default) ─────────────────────────────────────
def test_argos_deframe_requires_explicit_sync():
    with pytest.raises(TypeError):
        argos.deframe(np.zeros(64, dtype=np.uint8))  # no sync kwargs -> refuse


# ── P2-19: RS errors-and-erasures ────────────────────────────────────────────────────────────
def test_rs_erasures_extend_correction_capability():
    rs = RSCodec(32)
    rng = np.random.default_rng(0)
    for _ in range(60):
        m = bytes(rng.integers(0, 256, 223).tolist())
        cw = bytearray(rs.encode(m))
        ne = int(rng.integers(0, 17))               # errors
        nv = int(rng.integers(0, 32 - 2 * ne + 1))  # erasures: 2e+v <= 32
        pos = rng.choice(255, size=ne + nv, replace=False).tolist()
        for p in pos[:ne]:
            cw[p] ^= int(rng.integers(1, 256))
        for p in pos[ne:]:
            cw[p] ^= int(rng.integers(0, 256))  # erased bytes may even be correct
        assert rs.decode(bytes(cw), erase_pos=pos[ne:]) == m


def test_rs_32_pure_erasures_decode_but_33_do_not():
    rs = RSCodec(32)
    m = bytes(range(223))
    cw = bytearray(rs.encode(m))
    pos = list(range(0, 64, 2))  # 32 positions
    for p in pos:
        cw[p] ^= 0xFF
    assert rs.decode(bytes(cw), erase_pos=pos) == m
    assert rs.decode(bytes(cw), erase_pos=pos + [200]) is None  # 33 erasures > nsym


def test_rs_erasure_positions_validated():
    rs = RSCodec(32)
    with pytest.raises(ValueError):
        rs.decode(rs.encode(bytes(223)), erase_pos=[999])


def test_rs_plain_decode_unchanged():
    rs = RSCodec(32)
    m = bytes(range(223))
    cw = bytearray(rs.encode(m))
    cw[5] ^= 0xFF
    assert rs.decode(bytes(cw)) == m  # no erase_pos -> the validated errors-only path


# ── P2-20: OOK gate rejects pure noise instead of emitting random bits ────────────────────────
def test_ook_gate_rejects_pure_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 800) + 1j * rng.normal(0, 1, 800)
    assert ook.demodulate(noise, sps=8).size == 0  # judged unmodulated


def test_ook_gate_keeps_real_signal_and_can_be_disabled():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 100).astype(np.uint8)
    iq = ook.modulate(bits, sps=10, amp=3.0)
    iq = iq + (rng.normal(0, 0.3, iq.shape) + 1j * rng.normal(0, 0.3, iq.shape))
    out = ook.demodulate(iq, sps=10)
    assert out.size == 100 and np.mean(out == bits) > 0.99
    # gate off -> noise slices to (garbage) bits rather than being rejected
    noise = rng.normal(0, 1, 800) + 1j * rng.normal(0, 1, 800)
    assert ook.demodulate(noise, sps=8, min_separation=0).size == 100

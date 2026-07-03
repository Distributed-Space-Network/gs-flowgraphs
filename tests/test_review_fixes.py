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
        ("KISS", "kiss"), ("SLIP", None),  # slip = byte-pipe codec only (docs/10 §10)
        ("ccsds_tm", "ccsds_tm"),
        # bare "CCSDS" = unknown coding (spec birds use dual-basis RS/concatenated we don't
        # implement locally) -> upstream; argos = placeholder sync -> record-only until benched
        ("CCSDS", None), ("Argos PTT-A2", None),
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
    # ``ax25`` is the ONE local token that also maps to a buildable gr-satellites label (AX.25),
    # so it synthesizes and RACES the local numpy AX.25 deframer (P0-2 outbound normalization).
    assert grsat_synth.can_synthesize("gfsk", 9600, "ax25")
    # local-ONLY / unbuildable tokens are NOT synthesizable gr-satellites vocabulary
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


# ── recheck round 2: KISS/SLIP strict mode (docs/10 ledger, adversarial recheck) ─────────────
def test_kiss_noise_acceptance_is_bounded():
    # KISS has no checksum, so noise acceptance can be reduced but NOT eliminated (there is
    # nothing to verify). Pre-recheck a single noise drain emitted ~17 garbage "frames"; strict
    # mode (bracketed both sides + data type byte + valid escapes + min length + non-constant
    # payload) with the ALL-PHASES UNION keeps it to ~2 per drain window — the union trades a
    # little extra chance garbage for NEVER losing the real frame (a single-phase pick lost it
    # ~10% of the time under noise). No race-poisoning risk: KISS is not gr-satellites
    # vocabulary, so a KISS bird never races gr-satellites.
    total = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        noise_bits = rng.integers(0, 2, 2400 * 8).astype(np.uint8)
        frames, _ = framings.deframe(noise_bits, "KISS")
        assert len(frames) <= 6, f"seed {seed}: {len(frames)} garbage frames in one window"
        total += len(frames)
    assert total <= 55  # ~2/window residual; the pre-recheck behavior would be ~340 here


def test_kiss_real_frame_recovered_from_noise_surroundings():
    # A real KISS frame embedded in noise must still be recovered at the correct phase: strict
    # gating suppresses the wrong phases' chance frames, so the real phase wins the count.
    rng = np.random.default_rng(4)
    payload = b"real-kiss-frame-payload-123456"
    wire = kiss.kiss_encode(payload)
    lead = rng.integers(0, 256, 64, dtype=np.uint8).tobytes().replace(b"\xc0", b"\x00")
    tail_noise = rng.integers(0, 256, 64, dtype=np.uint8).tobytes().replace(b"\xc0", b"\x00")
    bits = np.unpackbits(np.frombuffer(lead + wire + tail_noise, dtype=np.uint8))
    frames, matched = framings.deframe(bits, "kiss")
    assert matched == "kiss" and payload in frames


def test_kiss_strict_mode_gates():
    # bracketed-both-sides: a trailing partial (no closing FEND) is dropped in strict mode
    partial = kiss.kiss_encode(b"complete-frame-x")[:-1] + b"\x00\x01"
    assert kiss.kiss_decode(partial, strict=True) == []
    # non-data type byte dropped; short payload dropped
    wire = bytes([kiss.FEND, 0x06]) + b"longenoughpayload" + bytes([kiss.FEND])
    assert kiss.kiss_decode(wire, strict=True) == []
    short = kiss.kiss_encode(b"tiny")
    assert kiss.kiss_decode(short, strict=True) == []
    assert kiss.kiss_decode(short) == [b"tiny"]  # non-strict unchanged


def test_kiss_constant_idle_fill_between_frames_is_rejected():
    # An idle gap of constant bytes between two frames is FEND-bracketed, type-0, escape-valid
    # and long enough — without the non-constant gate it emitted one deterministic garbage
    # "frame" per drain, forever (and positional dedup cannot subtract a chunk that was
    # unbracketed in the previous tail).
    wire = kiss.kiss_encode(b"first-frame-payload") + b"\x00" * 40 + kiss.kiss_encode(
        b"second-frame-payload")
    frames = kiss.kiss_decode(wire, strict=True)
    assert frames == [b"first-frame-payload", b"second-frame-payload"]


def test_bare_ccsds_label_is_not_synthesizable():
    # gr-satellites has only QUALIFIED CCSDS labels; a bare "CCSDS" would fail its constructor,
    # so the plan must say record-only rather than claim gr-satellites(synthetic).
    assert not grsat_synth.can_synthesize("bpsk", 1200, "CCSDS")
    assert grsat_synth.can_synthesize("bpsk", 1200, "CCSDS Concatenated")
    plan = compose.plan_decode(
        {"modulation": "bpsk", "symbol_rate_hz": 1200, "framing": "CCSDS"})
    assert not plan.decodable  # record-only (+ the IQ recording, always)


def test_kiss_identical_fast_repeat_beacons_in_one_drain_both_emit():
    # Round 3: the union dedup must be ACROSS phases only — two identical frames at the SAME
    # phase are a genuine fast repeat beacon (a global seen-set emitted only one).
    wire = kiss.kiss_encode(b"fast-repeat-beacon-xx") * 2
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8))
    frames, matched = framings.deframe(bits, "kiss")
    assert matched == "kiss" and frames == [b"fast-repeat-beacon-xx"] * 2

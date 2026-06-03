"""Protocol-layer tests: FCS, HDLC framing, NRZI, G3RUH, AX.25, framing."""

from __future__ import annotations

import numpy as np

from gfsk_ax25 import ax25, fcs, framing, g3ruh, hdlc


def test_fcs_roundtrip_and_residue():
    body = b"\x9c\x94\x6e\xa0\x40\x40\x60hello-world"
    f = fcs.fcs_bytes(body)
    assert len(f) == 2
    assert fcs.check(body + f)
    # Corrupting any octet must fail the check.
    bad = bytearray(body + f)
    bad[3] ^= 0x01
    assert not fcs.check(bytes(bad))


def test_bitstuff_roundtrip():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=2000).astype(np.uint8)
    stuffed = np.array(hdlc.bit_stuff(bits), dtype=np.uint8)
    # A stuffed stream never contains six consecutive 1s.
    run = 0
    for b in stuffed.tolist():
        run = run + 1 if b == 1 else 0
        assert run < 6
    destuffed = np.array(hdlc._destuff(stuffed.tolist()), dtype=np.uint8)
    assert np.array_equal(destuffed, bits)


def test_byte_bit_roundtrip_lsb_first():
    data = bytes(range(256))
    bits = hdlc.bytes_to_bits(data)
    assert hdlc.bits_to_bytes(bits) == data
    # LSB-first: 0x01 -> first bit 1.
    assert hdlc.bytes_to_bits(b"\x01")[0] == 1


def test_hdlc_frame_deframe():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"beacon 42")
    bits = hdlc.frame(body, preamble_flags=4, postamble_flags=2)
    got = hdlc.deframe(bits)
    assert body in got


def test_hdlc_deframe_amid_noise():
    rng = np.random.default_rng(7)
    body = ax25.encode_ui(dest="DSN", src="SAT", info=b"x" * 60)
    frame_bits = hdlc.frame(body, preamble_flags=8)
    noise_a = rng.integers(0, 2, size=137).astype(np.uint8)
    noise_b = rng.integers(0, 2, size=200).astype(np.uint8)
    stream = np.concatenate([noise_a, frame_bits, noise_b])
    assert body in hdlc.deframe(stream)


def test_nrzi_inverse():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=500).astype(np.uint8)
    assert np.array_equal(g3ruh.nrzi_decode(g3ruh.nrzi_encode(bits)), bits)


def test_g3ruh_inverse_and_self_sync():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=1000).astype(np.uint8)
    scrambled = g3ruh.scramble(bits)
    assert np.array_equal(g3ruh.descramble(scrambled), bits)
    # Self-synchronizing: a descrambler started in the WRONG state recovers
    # after at most 17 bits.
    descr = g3ruh.descramble(scrambled, state=0x1FFFF)
    assert np.array_equal(descr[17:], bits[17:])


def test_ax25_ui_roundtrip():
    body = ax25.encode_ui(
        dest="NOCALL", src="DSN1", info=b"\x00\x01\x02hi", dest_ssid=0, src_ssid=11
    )
    ui = ax25.decode_ui(body)
    assert ui is not None
    assert ui.dest == "NOCALL"
    assert ui.src == "DSN1"
    assert ui.src_ssid == 11
    assert ui.info == b"\x00\x01\x02hi"


def test_framing_link_roundtrip():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"telemetry frame payload")
    bits = framing.encode(body, preamble_flags=16)
    bodies = framing.decode(bits)
    assert body in bodies
    ui = ax25.decode_ui(bodies[0])
    assert ui is not None and ui.info == b"telemetry frame payload"

"""docs/08 Tier 2 — CCSDS TM/AOS transfer-frame framing over the full channel-coding chain.

Round-trips build_tm_frame → deframe_tm across every (RS, randomize, FECF) combination, verifies
RS actually corrects byte errors inside a real frame, checks the primary-header parse, and
confirms noise doesn't produce spurious frames (ASM+RS+CRC gate).
"""
from __future__ import annotations

import numpy as np
import pytest

from gfsk_ax25 import ccsds

_H = ccsds.TMHeader(
    version=0, spacecraft_id=0x2AB, virtual_channel_id=3, ocf_flag=0,
    master_channel_frame_count=42, virtual_channel_frame_count=7,
    secondary_header_flag=0, sync_flag=0, first_header_pointer=0,
)


@pytest.mark.parametrize("rs", [True, False])
@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("fecf", [True, False])
def test_tm_frame_roundtrip(rs, randomize, fecf):
    data = bytes(range(100))
    flen = 223 if rs else 160
    bits = ccsds.build_tm_frame(_H, data, frame_len=flen, randomize=randomize, rs=rs, fecf=fecf)
    lead = np.random.default_rng(0).integers(0, 2, 50).astype(np.uint8)
    stream = np.concatenate([lead, bits, np.zeros(40, dtype=np.uint8)])
    frames = ccsds.deframe_tm(stream, frame_len=flen, randomize=randomize, rs=rs, fecf=fecf)
    assert len(frames) == 1
    hdr = ccsds.parse_tm_primary_header(frames[0])
    assert hdr.spacecraft_id == 0x2AB and hdr.virtual_channel_id == 3
    assert frames[0][6:6 + 100] == data


def test_rs_corrects_byte_errors_inside_a_frame():
    data = bytes(range(100))
    bits = ccsds.build_tm_frame(_H, data, frame_len=223, randomize=True, rs=True, fecf=True)
    by = bytearray(np.packbits(bits))  # ASM(4) + codeblock(255)
    rng = np.random.default_rng(3)
    for p in rng.choice(range(4, 4 + 255), size=16, replace=False):  # 16 symbol errors (= t)
        by[int(p)] ^= 0xFF
    noisy = np.unpackbits(np.frombuffer(bytes(by), dtype=np.uint8))
    frames = ccsds.deframe_tm(noisy, frame_len=223)
    assert len(frames) == 1 and frames[0][6:6 + 100] == data


def test_tm_header_fields_roundtrip_through_parse():
    frame = ccsds._pack_tm_header(_H) + bytes(200)
    h = ccsds.parse_tm_primary_header(frame)
    assert h == _H


def test_aos_header_parse():
    # version=1, scid=0xAB, vcid=0x2A, vc frame count=0x010203, replay=1
    w0 = (1 << 14) | (0xAB << 6) | 0x2A
    frame = bytes([(w0 >> 8) & 0xFF, w0 & 0xFF, 0x01, 0x02, 0x03, 0x80]) + bytes(10)
    a = ccsds.parse_aos_primary_header(frame)
    assert a.version == 1 and a.spacecraft_id == 0xAB and a.virtual_channel_id == 0x2A
    assert a.virtual_channel_frame_count == 0x010203 and a.replay_flag == 1


def test_noise_does_not_produce_frames():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(30):
        noise = rng.integers(0, 2, 3000).astype(np.uint8)
        hits += len(ccsds.deframe_tm(noise, frame_len=223))
    assert hits == 0  # ASM (32-bit) + RS + CRC make false frames negligible


def test_multiple_frames_in_one_stream():
    b1 = ccsds.build_tm_frame(_H, b"\x01\x02", frame_len=223)
    b2 = ccsds.build_tm_frame(_H, b"\x03\x04", frame_len=223)
    frames = ccsds.deframe_tm(np.concatenate([b1, b2]), frame_len=223)
    assert len(frames) == 2
    assert frames[0][6:8] == b"\x01\x02" and frames[1][6:8] == b"\x03\x04"

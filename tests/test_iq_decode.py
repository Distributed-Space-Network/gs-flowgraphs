"""Tests for the post-pass decode tool (apps/iq_decode.py).

Covers the Doppler de-rotation (the crux — the recorded .cf32 is raw/pre-NCO), a full CCSDS
round-trip through a SIMULATED Doppler offset re-corrected by a matching gs-orbitd track, and the
safety property that a non-CCSDS (EnduroSat) capture yields ZERO false ccsds frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import iq_decode
import numpy as np

from gfsk_ax25 import ccsds, gfsk
from gfsk_ax25 import endurosat_link as el

_FS = 96_000.0
_SYM = 9600.0
_H = ccsds.TMHeader(
    version=0,
    spacecraft_id=0x2AB,
    virtual_channel_id=3,
    ocf_flag=0,
    master_channel_frame_count=42,
    virtual_channel_frame_count=7,
    secondary_header_flag=0,
    sync_flag=0,
    first_header_pointer=0,
)


def _modulate_bits(bits: np.ndarray) -> np.ndarray:
    gp = gfsk.GfskParams(sample_rate_hz=_FS, symbol_rate_hz=_SYM)
    return gfsk.modulate(np.asarray(bits, np.uint8), gp)


def _apply_offset(iq: np.ndarray, f_hz: float) -> np.ndarray:
    n = np.arange(len(iq))
    return (np.asarray(iq, np.complex64) * np.exp(2j * np.pi * f_hz * n / _FS)).astype(np.complex64)


def _write_cf32(tmp_path: Path, iq: np.ndarray) -> Path:
    p = tmp_path / "cap.cf32"
    np.asarray(iq, np.complex64).tofile(p)
    return p


def _guard(iq: np.ndarray, n: int = 2000) -> np.ndarray:
    z = np.zeros(n, np.complex64)
    return np.concatenate([z, np.asarray(iq, np.complex64), z]).astype(np.complex64)


def test_derotate_doppler_returns_tone_to_dc():
    n = 8192
    f0 = 5000.0
    shifted = _apply_offset(np.ones(n, np.complex64), f0)
    out = iq_decode._derotate_doppler(shifted, _FS, [(0.0, f0), (n / _FS, f0)])
    spec = np.abs(np.fft.fft(out))
    peak_hz = np.fft.fftfreq(n, 1.0 / _FS)[int(np.argmax(spec))]
    assert abs(peak_hz) < 2.0 * _FS / n  # back at DC after de-rotation


def test_derotate_doppler_no_track_is_identity():
    iq = _apply_offset(np.ones(1024, np.complex64), 3000.0)
    assert np.array_equal(iq_decode._derotate_doppler(iq, _FS, []), iq)


def test_decode_capture_ccsds_roundtrip_through_doppler(tmp_path):
    # A CCSDS TM frame shifted +35 kHz — LARGE enough that per-window CFO alone can't pull it back,
    # so the gs-orbitd track is REQUIRED. This makes the test non-vacuous: without the track (or if
    # the de-rotation were reverted) it decodes 0 frames; only the matching track recovers it.
    data = bytes(range(100))
    bits = ccsds.build_tm_frame(_H, data)  # defaults match framings.deframe("ccsds_tm")
    rx = _apply_offset(_guard(_modulate_bits(bits)), 35_000.0)
    cap = _write_cf32(tmp_path, rx)
    dur = len(rx) / _FS

    # No track → per-window CFO can't recover +35 kHz on a bursty capture → nothing decodes.
    assert iq_decode.decode_capture(
        cap, sample_rate_hz=_FS, symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",), doppler_track=None,
    ) == []

    recs = iq_decode.decode_capture(
        cap,
        sample_rate_hz=_FS,
        symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",),
        doppler_track=[(0.0, 35_000.0), (dur, 35_000.0)],
    )
    assert len(recs) == 1
    assert recs[0]["framing"] == "ccsds_tm"
    assert recs[0]["post_pass"] is True
    assert bytes.fromhex(recs[0]["payload_hex"])[6 : 6 + 100] == data
    # ...and appended to the pass frames.jsonl, tagged post_pass.
    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["post_pass"] is True


def test_decode_capture_no_false_positives_on_endurosat(tmp_path):
    # An EnduroSat (light-framing) capture must yield ZERO ccsds_tm frames — the RS/FECF gate
    # rejects non-CCSDS bits, so the post-pass sweep never emits garbage (the "no-spam" property).
    cap = _write_cf32(tmp_path, _guard(el.transmit(bytes(range(24)), _FS)))
    recs = iq_decode.decode_capture(
        cap,
        sample_rate_hz=_FS,
        symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",),
        doppler_track=[(0.0, 0.0)],
    )
    assert recs == []
    assert not (tmp_path / "frames.jsonl").exists()


def test_decode_capture_missing_file_and_empty_framings(tmp_path):
    assert iq_decode.decode_capture(
        tmp_path / "nope.cf32", sample_rate_hz=_FS, framings_to_try=("ccsds_tm",)
    ) == []
    cap = _write_cf32(tmp_path, _guard(_modulate_bits(ccsds.build_tm_frame(_H, b"\x01\x02"))))
    assert iq_decode.decode_capture(cap, sample_rate_hz=_FS, framings_to_try=()) == []

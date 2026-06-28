"""Post-pass decoder helpers — the pure (no-GNU-Radio) record/parse seam.

The DSP (``gnuradio_satellites.decode_file``) is bench-only; here we test the parts that run
off the bench: sidecar/params reading, the AX.25 upper-layer summary, the frames.jsonl record
shape, and the append writer."""

from __future__ import annotations

import json
from pathlib import Path

from iq_decode import (
    ax25_summary,
    frame_record,
    read_params,
    read_sidecar_rate,
    write_frames,
)


def _ax25_ui(dest: str, src: str, info: bytes = b"hi") -> bytes:
    """A minimal AX.25 UI frame: dest+src addresses (callsign<<1, SSID byte), control, PID."""
    def addr(call: str, ssid: int, last: bool) -> bytes:
        c = call.ljust(6).encode("ascii")
        out = bytes((b << 1) & 0xFE for b in c)
        return out + bytes([(0x60 | (ssid << 1)) | (1 if last else 0)])

    return addr(dest, 0, False) + addr(src, 2, True) + bytes([0x03, 0xF0]) + info


def test_read_sidecar_rate_prefers_sidecar(tmp_path: Path) -> None:
    cf32 = tmp_path / "p.cf32"
    cf32.write_bytes(b"\x00" * 16)
    (tmp_path / "p.cf32.json").write_text('{"sample_rate_hz": 96000.0, "center_hz": 4e8}')
    assert read_sidecar_rate(cf32, 48000.0) == 96000.0
    # missing sidecar → the fallback
    assert read_sidecar_rate(tmp_path / "none.cf32", 48000.0) == 48000.0


def test_read_params(tmp_path: Path) -> None:
    pf = tmp_path / "params.json"
    pf.write_text('{"satellite":"62083","symbol_rate_hz":800.0}')
    sat, params = read_params(pf)
    assert sat == "62083"
    assert params["symbol_rate_hz"] == 800.0
    assert read_params(None) == ("", {})
    assert read_params(tmp_path / "nope.json") == ("", {})


def test_ax25_summary_parses_valid_frame() -> None:
    frame = _ax25_ui("ON4ISS", "PE0SAT")
    summ = ax25_summary(frame)
    assert summ is not None
    assert summ["dest"] == "ON4ISS"
    assert summ["src"] == "PE0SAT-2"
    assert summ["control"] == 0x03
    assert summ["pid"] == 0xF0


def test_ax25_summary_rejects_non_ax25() -> None:
    assert ax25_summary(b"\x00\x01\x02") is None  # too short
    assert ax25_summary(bytes(range(20))) is None  # not callsign-shaped


def test_frame_record_shape_and_upper_layer() -> None:
    frame = _ax25_ui("ON4ISS", "PE0SAT")
    rec = frame_record("gfsk9600", frame, ts=123.0)
    assert rec["ts"] == 123.0
    assert rec["phase"] == "postpass"
    assert rec["decoder"] == "gfsk9600"
    assert rec["len"] == len(frame)
    assert rec["raw_hex"] == frame.hex()
    assert rec["ax25"]["src"] == "PE0SAT-2"
    # a non-AX.25 frame still records raw, just no ax25 key
    raw = frame_record("bpsk1200", b"\xde\xad\xbe\xef", ts=1.0)
    assert "ax25" not in raw
    assert raw["raw_hex"] == "deadbeef"


def test_write_frames_appends(tmp_path: Path) -> None:
    out = tmp_path / "frames.jsonl"
    assert write_frames(out, []) == 0
    assert not out.exists()
    n = write_frames(out, [frame_record("gfsk9600", b"\x01\x02", ts=1.0)])
    assert n == 1
    write_frames(out, [frame_record("bpsk1200", b"\x03\x04", ts=2.0)])  # appends, not overwrites
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decoder"] == "gfsk9600"
    assert json.loads(lines[1])["decoder"] == "bpsk1200"

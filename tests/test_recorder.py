"""Pre-demod IQ recorder — pure (numpy-only) artifact generation.

Covers the SDF byte format (Keysight N5106A: int16 big-endian interleaved I/Q), the
SDF round-trip, and deriving the VSA CSV + waterfall PNG. The real-time GNU Radio
sink (``make_sdf_sink`` / ``PassRecorder.maybe_start``) is bench-only and not
exercised here; ``PassRecorder.finalize`` (pure) is.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
from _recorder import (
    PassRecorder,
    StreamRecorder,
    finalize_recording,
    iq_to_sdf_bytes,
    parse_formats,
    read_sdf_iq,
)

_UTC = _dt.datetime(2026, 6, 25, 13, 30, tzinfo=_dt.UTC)


def _tone(n: int = 4096, fs: float = 48000.0, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(n) / fs
    return (0.5 * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def test_parse_formats_normalizes() -> None:
    assert parse_formats("sdf, csv ,PNG") == ("sdf", "csv", "png")
    assert parse_formats("") == ()


def test_sdf_is_int16_be_interleaved_and_roundtrips(tmp_path: Path) -> None:
    iq = _tone()
    raw = iq_to_sdf_bytes(iq)
    assert len(raw) == iq.size * 4  # two int16 (I,Q) per complex sample
    assert np.frombuffer(raw, dtype=">i2").size == iq.size * 2  # big-endian int16

    sdf = tmp_path / "cap.sdf"
    sdf.write_bytes(raw)
    back = read_sdf_iq(sdf)
    assert back.size == iq.size
    np.testing.assert_allclose(back, iq, atol=2e-4)  # int16 quantization only


def test_finalize_derives_vsa_csv_and_waterfall_png(tmp_path: Path) -> None:
    sdf = tmp_path / "cmd_31_31.sdf"
    sdf.write_bytes(iq_to_sdf_bytes(_tone()))
    finalize_recording(
        sdf,
        center_hz=401_200_000.0,
        sample_rate_hz=48000.0,
        formats=("sdf", "csv", "png"),
        started_utc=_UTC,
    )
    csv, png = sdf.with_suffix(".csv"), sdf.with_suffix(".png")
    assert csv.exists() and png.exists()
    lines = csv.read_text(encoding="ascii").splitlines()
    assert any(ln.startswith("InputCenter,401200000") for ln in lines)
    assert "Y" in lines  # VSA data marker
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_passrecorder_finalize_keeps_cf32_and_derives_png(tmp_path: Path) -> None:
    # The GR engine records raw cf32 via a native sink; finalize derives a waterfall
    # PNG from the head and ALWAYS keeps the cf32 (the replayable record-everything
    # artifact), regardless of which view formats were requested.
    cf32 = tmp_path / "cmd_42.cf32"
    _tone().astype(np.complex64).tofile(cf32)
    PassRecorder(cf32, 401_200_000.0, 48000.0, ("png",)).finalize()
    assert cf32.exists()                      # raw IQ kept
    assert cf32.with_suffix(".png").exists()  # waterfall derived from the head


def test_stream_recorder_dsp_path_writes_all_three(tmp_path: Path) -> None:
    from types import SimpleNamespace

    out = tmp_path / "cmd_32_32"
    out.mkdir()
    args = SimpleNamespace(
        record_iq=True,
        record_formats="sdf,csv,png",
        output_dir=str(out),
        center_freq_hz=401_762_500,
    )
    rec = StreamRecorder.maybe_start(args, sample_rate_hz=48000.0)
    assert rec is not None
    for _ in range(4):  # stream several chunks, as the dsp reader does
        rec.write(_tone(1024))
    rec.finalize()
    assert (out / "cmd_32_32.sdf").exists()  # named after the pass dir
    assert (out / "cmd_32_32.csv").exists()
    assert (out / "cmd_32_32.png").exists()


def test_stream_recorder_off_when_disabled() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(record_iq=False, record_formats="", output_dir=".", center_freq_hz=1)
    assert StreamRecorder.maybe_start(args, sample_rate_hz=1.0) is None


def test_stub_rx_synthetic_capture_writes_all_three(tmp_path: Path) -> None:
    # The stub honours --record-iq with a synthetic capture (no SDR) so the whole
    # record→file path is E2E-testable off the bench — uniform with the real engines.
    from types import SimpleNamespace

    import stub_rx

    out = tmp_path / "cmd_stub"
    out.mkdir()
    args = SimpleNamespace(
        record_iq=True,
        record_formats="sdf,csv,png",
        output_dir=str(out),
        sample_rate=48000,
        center_freq_hz=401_200_000,
    )
    stub_rx._write_stub_capture(args)
    assert (out / "cmd_stub.sdf").exists()
    assert (out / "cmd_stub.csv").exists()
    assert (out / "cmd_stub.png").exists()

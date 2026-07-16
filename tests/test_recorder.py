"""Pre-demod IQ recorder — pure (numpy-only) artifact generation.

Covers the SDF byte format (Keysight N5106A: int16 big-endian interleaved I/Q), the
SDF round-trip, and deriving the VSA CSV + waterfall PNG. The real-time GNU Radio cf32
sink (``PassRecorder.maybe_start``) is bench-only; the GR-engine PNG/CSV views are
derived post-pass by the ``iq_views`` tool (see test_iq_views.py)."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pytest
from _recorder import (
    _WF_SPAN_DB,
    StreamRecorder,
    _display_window_dbfs,
    _spectrogram_dbfs,
    finalize_recording,
    iq_to_sdf_bytes,
    parse_formats,
    read_sdf_iq,
    write_waterfall_png,
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


def test_stream_recorder_dsp_path_writes_cf32(tmp_path: Path) -> None:
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
    rec.close()
    cf32 = out / "cmd_32_32.cf32"
    assert cf32.exists()
    assert cf32.stat().st_size == 4 * 1024 * 8  # 4 chunks × 1024 complex64 (8 B each)
    assert (out / "cmd_32_32.cf32.json").exists()  # self-describing sidecar
    # views (sdf/csv/png) are derived POST-pass by iq_views, not in-pass
    assert not (out / "cmd_32_32.sdf").exists()


def test_stream_recorder_off_when_disabled() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(record_iq=False, record_formats="", output_dir=".", center_freq_hz=1)
    assert StreamRecorder.maybe_start(args, sample_rate_hz=1.0) is None


def test_stub_rx_synthetic_capture_writes_cf32(tmp_path: Path) -> None:
    # The stub honours --record-iq with a synthetic cf32 capture (no SDR) so the
    # record→file path is E2E-testable off the bench — uniform with the real engines.
    # Views (sdf/csv/png) are derived post-pass by iq_views, not here.
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
    assert (out / "cmd_stub.cf32").exists()
    assert (out / "cmd_stub.cf32.json").exists()
    assert not (out / "cmd_stub.sdf").exists()


# ------------------------------------------------- waterfall dBFS pipeline (EXT-WF-001)
# docs/waterfall_guide.md is the contract: absolute dBFS, complex Hann FFTs,
# linear-domain averaging, DC interpolation, ONE global display window.

_NFFT = 2048


def _bin_tone(k: int, n: int, *, amp: float = 1.0) -> np.ndarray:
    """Complex exponential exactly on FFT bin ``k`` (no scalloping)."""
    t = np.arange(n)
    return (amp * np.exp(2j * np.pi * k * t / _NFFT)).astype(np.complex64)


def test_spectrogram_is_absolute_dbfs() -> None:
    # A FULL-SCALE tone on an exact bin peaks at 0 dBFS; half amplitude at -6.02.
    spec = _spectrogram_dbfs(_bin_tone(256, 6 * _NFFT, amp=1.0))
    assert spec.shape[1] == _NFFT
    assert float(spec.max()) == pytest.approx(0.0, abs=0.1)
    assert int(spec[0].argmax()) == _NFFT // 2 + 256  # DC-centred, +f on the right
    half = _spectrogram_dbfs(_bin_tone(256, 6 * _NFFT, amp=0.5))
    assert float(half.max()) == pytest.approx(-6.02, abs=0.15)


def test_spectrogram_matches_the_guide_reference_pipeline() -> None:
    # Independent double-precision oracle implementing the guide's pseudocode
    # verbatim (windowed complex FFT, |.|^2 / sum(win)^2, LINEAR mean, dBFS,
    # DC interpolation). N sized for exactly one row of 8 averaged FFTs.
    rng = np.random.default_rng(7)
    n = _NFFT + 14 * (_NFFT // 4)  # total_ffts = 15 -> one row
    iq = (0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)

    win = np.hanning(_NFFT).astype(np.float32)
    norm = float(np.sum(win)) ** 2
    starts = np.linspace(0, n - _NFFT, 8).astype(np.int64)
    acc = np.zeros(_NFFT, dtype=np.float64)
    for s in starts:
        seg = (iq[s : s + _NFFT].astype(np.complex64) * win)  # complex — never a real cast
        acc += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2 / norm
    ref = 10.0 * np.log10(acc / 8.0 + 1e-30)
    c, w = _NFFT // 2, 3
    left, right = ref[c - w // 2 - 1], ref[c + w // 2 + 1]
    for j, d in enumerate(range(-(w // 2), w // 2 + 1)):
        t = (j + 1) / (w + 1)
        ref[c + d] = left * (1 - t) + right * t

    spec = _spectrogram_dbfs(iq)
    assert spec.shape == (1, _NFFT)
    np.testing.assert_allclose(spec[0], ref, atol=0.05)


def test_rows_average_in_the_linear_domain_not_db() -> None:
    # One row whose windows see a full-scale tone for half the capture and a
    # -40 dB one for the rest: the LINEAR mean keeps the peak within a few dB of
    # full scale; averaging dB values instead would sink it toward -20 dBFS.
    n = _NFFT + 14 * (_NFFT // 4)
    hot = _bin_tone(256, n // 2, amp=1.0)
    cold = _bin_tone(256, n - n // 2, amp=0.01)
    spec = _spectrogram_dbfs(np.concatenate([hot, cold]))
    assert spec.shape[0] == 1
    assert float(spec.max()) > -8.0


def test_dc_bins_are_interpolated_no_black_line() -> None:
    # A strong DC offset (SDR LO leakage) must not survive as a centre spike:
    # the 3 centre bins become a linear ramp between their outer neighbours.
    rng = np.random.default_rng(3)
    n = 12 * _NFFT
    iq = (0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) + 0.5).astype(
        np.complex64
    )
    spec = _spectrogram_dbfs(iq)
    c = _NFFT // 2
    assert np.isfinite(spec).all()
    np.testing.assert_allclose(
        spec[:, c], (spec[:, c - 2] + spec[:, c + 2]) / 2.0, atol=1e-3
    )
    assert int(spec[0].argmax()) != c, "the DC spike survived interpolation"


def test_spectrum_is_not_mirrored() -> None:
    # Guide §4.1/§10: a real-cast of the IQ produces a conjugate-symmetric
    # spectrum. A +f tone must appear ONLY right of centre.
    spec = _spectrogram_dbfs(_bin_tone(300, 6 * _NFFT, amp=0.8))
    c = _NFFT // 2
    assert float(spec[0, c + 300]) - float(spec[0, c - 300]) > 20.0


def test_display_window_is_global_and_floor_anchored() -> None:
    rng = np.random.default_rng(11)
    n = 24 * _NFFT
    noise = (0.003 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )
    quiet = _spectrogram_dbfs(noise)
    lo_q, hi_q = _display_window_dbfs(quiet)
    assert hi_q - lo_q == pytest.approx(_WF_SPAN_DB)
    assert lo_q == pytest.approx(float(np.median(quiet)) - 10.0, abs=1e-6)
    # A strong burst in part of the capture must NOT stretch the window (global
    # floor anchoring is robust; per-row/percent-of-max AGC would move it).
    burst = noise.copy()
    burst[: 2 * _NFFT] += _bin_tone(256, 2 * _NFFT, amp=0.8)
    lo_b, _hi_b = _display_window_dbfs(_spectrogram_dbfs(burst))
    assert abs(lo_b - lo_q) < 1.0


def test_waterfall_png_uses_the_satnogs_colormap_no_red(tmp_path: Path) -> None:
    # Guide §7 verification: strongest signal renders YELLOW, never red.
    import matplotlib.image as mpimg

    rng = np.random.default_rng(5)
    n = 48 * _NFFT
    iq = (0.005 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    iq += _bin_tone(200, n, amp=0.2)
    png = tmp_path / "wf.png"
    assert write_waterfall_png(png, iq, sample_rate_hz=48000.0, center_hz=110_110_000.0)
    img = mpimg.imread(str(png))
    inner = img[30:-30, 30:-30, :3]
    pixels = inner[inner.sum(axis=2) > (30 / 255.0)]
    red_fraction = float(np.mean(pixels[:, 0] > (200 / 255.0)))
    assert red_fraction < 0.05, f"red fraction {red_fraction:.3f} — wrong colormap?"


def test_grayscale_fallback_uses_the_same_global_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import _recorder as rec_mod

    def _no_mpl(*_a: object, **_k: object) -> None:
        raise RuntimeError("matplotlib unavailable")

    monkeypatch.setattr(rec_mod, "_write_waterfall_matplotlib", _no_mpl)
    png = tmp_path / "wf_gray.png"
    assert write_waterfall_png(png, _tone(24 * _NFFT), sample_rate_hz=48000.0)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- DS-G8 (Phase 2C)


def test_stream_recorder_write_failure_ends_the_recording_not_the_pass(tmp_path, caplog):
    """DS-G8: a mid-pass write failure (ENOSPC) must end the RECORDING loudly — closing the file and
    dropping later writes — NOT propagate into the reader thread (which would kill the whole pass,
    frames included, for the sake of an optional capture)."""
    import logging

    import numpy as np
    from _recorder import StreamRecorder

    rec = StreamRecorder(tmp_path / "cap.cf32", 401_500_000.0, 48_000.0)
    rec.write(np.zeros(16, dtype=np.complex64))  # a healthy write first

    class _FullDisk:
        closed = False

        def write(self, _b):
            msg = "No space left on device"
            raise OSError(28, msg)

        def close(self):
            self.closed = True

    rec._fh = _FullDisk()
    with caplog.at_level(logging.CRITICAL):
        rec.write(np.zeros(16, dtype=np.complex64))  # must NOT raise
    assert rec._fh.closed, "the recorder did not close the file after the write failure"
    assert "TRUNCATED" in caplog.text
    rec.write(np.zeros(16, dtype=np.complex64))  # later writes drop cleanly

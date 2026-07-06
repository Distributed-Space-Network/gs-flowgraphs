"""iq_analyze capture loading — the cf32 (whole-pass) path via the metadata sidecar.

The VSA CSV path is the EnduroSat lab workflow; the cf32 path lets the same tool decode
the WHOLE pass the GR engines record (not just a VSA window)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from iq_analyze import ax25_sweep, load_capture, load_cf32, spectrum_summary

from gfsk_ax25 import ax25
from gfsk_ax25 import framing as ax25_framing
from gfsk_ax25.gfsk import GfskParams, modulate


def test_load_cf32_reads_sidecar(tmp_path: Path) -> None:
    cf = tmp_path / "p.cf32"
    np.zeros(1000, dtype=np.complex64).tofile(cf)
    (tmp_path / "p.cf32.json").write_text(
        '{"sample_rate_hz": 96000.0, "center_hz": 401000000.0, "format": "cf32le"}'
    )
    cap = load_cf32(cf)
    assert cap.fs == 96000.0
    assert cap.center_hz == 401000000.0
    assert len(cap.iq) == 1000


def test_load_cf32_sample_rate_fallback_when_no_sidecar(tmp_path: Path) -> None:
    cf = tmp_path / "n.cf32"
    np.zeros(10, dtype=np.complex64).tofile(cf)
    cap = load_cf32(cf, sample_rate_hz=48000.0)
    assert cap.fs == 48000.0


def test_load_capture_dispatches_and_requires_rate(tmp_path: Path) -> None:
    cf = tmp_path / "x.cf32"
    np.zeros(10, dtype=np.complex64).tofile(cf)
    with pytest.raises(ValueError):  # no sidecar + no --sample-rate → can't analyze
        load_capture(cf)
    cap = load_capture(cf, sample_rate_hz=48000.0)
    assert cap.fs == 48000.0


# ── spectrum_summary: the carrier-presence check (weak/continuous, not just bursts) ───────────
def test_spectrum_summary_flat_noise_is_no_carrier() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 1, 200_000) + 1j * rng.normal(0, 1, 200_000)).astype(np.complex64)
    sp = spectrum_summary(noise, 48000.0)
    assert sp is not None
    assert sp["snr_db"] < 6.0  # flat → below the carrier threshold (dead capture)


def test_spectrum_summary_detects_a_carrier() -> None:
    fs, n = 48000.0, 200_000
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * 3000.0 * t).astype(np.complex64)
    rng = np.random.default_rng(1)
    iq = tone + 0.1 * (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)).astype(np.complex64)
    sp = spectrum_summary(iq, fs)
    assert sp is not None
    assert sp["snr_db"] >= 6.0
    assert abs(sp["peak_hz"] - 3000.0) < 50.0  # peak lands on the tone


# ── ax25_sweep: our real FCS-checked deframer picks the true baud out of the sweep ────────────
def test_ax25_sweep_decodes_synthetic_frame_at_its_baud() -> None:
    fs, baud = 48000.0, 1200.0
    body = ax25.encode_ui(dest="CQ", src="DSN", info=b"HELLO DSN")
    bits = ax25_framing.encode(body, scramble=True, nrzi=True)  # G3RUH + NRZI, like a 9k6 bird
    iq = modulate(bits, GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    results = {b: n for b, n, _ in ax25_sweep(iq, fs)}
    assert results.get(1200.0, 0) >= 1  # decodes at the true baud
    # A wrong-baud demod of the same IQ must NOT forge an FCS-valid frame.
    assert results.get(9600.0, 0) == 0

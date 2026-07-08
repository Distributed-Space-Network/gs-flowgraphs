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
    results = {b: n for b, _c, n, _f in ax25_sweep(iq, fs)}
    assert results.get(1200.0, 0) >= 1  # decodes at the true baud
    # A wrong-baud demod of the same IQ must NOT forge an FCS-valid frame.
    assert results.get(9600.0, 0) == 0


def test_ax25_sweep_recovers_an_off_dc_carrier() -> None:
    # Doppler comp removes the sweep but NOT the bird's oscillator offset, so the carrier parks off
    # DC (the cmd_101 / IPoS-TDsM bug: −17.9 kHz). The sweep must estimate it from the SPECTRUM and
    # de-rotate to DC, else the narrow demod filter rejects it. The frame is a BURST surrounded by
    # noise so mean-angle CFO (over the whole window) is noise-dominated and can't recover it — only
    # the spectral-peak estimate can. Offset a +8 kHz.
    fs, baud, offset = 48000.0, 1200.0, 8000.0
    body = ax25.encode_ui(dest="CQ", src="DSN", info=b"OFFSET BIRD")
    bits = ax25_framing.encode(body, scramble=True, nrzi=True)
    sig = modulate(bits, GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    n = np.arange(len(sig))
    sig = (sig * np.exp(2j * np.pi * offset * n / fs)).astype(np.complex64)
    rng = np.random.default_rng(7)

    def _noise(k: int) -> np.ndarray:
        return (rng.normal(0, 0.15, k) + 1j * rng.normal(0, 0.15, k)).astype(np.complex64)

    gap = len(sig)
    iq = np.concatenate([_noise(gap), sig + _noise(gap), _noise(gap)])  # burst in noise
    res = {b: (c, nf) for b, c, nf, _f in ax25_sweep(iq, fs)}
    assert res[1200.0][1] >= 1  # decoded despite the +8 kHz offset (via spectral carrier recovery)
    assert abs(res[1200.0][0] - offset) < 800.0  # and recovered ~the true carrier offset


# ── EnduroSat framing extraction off a raw capture WITH a carrier offset (docs/13 fix) ────────
def test_demodulate_burst_recovers_off_dc_endurosat_framing() -> None:
    # The EnduroSat framing must be extractable from a raw capture that carries a Doppler/oscillator
    # offset: shift a framed burst off DC, and demodulate_burst(carrier_hz=offset) must SYNC on the
    # 0xAA/0x7E preamble and yield the full framed bytes (len + payload + CRC), not NO-SYNC.
    from iq_analyze import demodulate_burst, find_sync, frame_bytes

    from gfsk_ax25 import endurosat_link as el

    fs, baud, offset = 96_000.0, 9600.0, 12_000.0
    payload = bytes(range(20))
    iq = modulate(el.frame_bits(payload), GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    guard = np.zeros(2000, dtype=np.complex64)
    seg = np.concatenate([guard, iq, guard]).astype(np.complex64)
    n = np.arange(len(seg))
    seg_off = (seg * np.exp(2j * np.pi * offset * n / fs)).astype(np.complex64)

    demod = demodulate_burst(seg_off, fs, symbol_rate=baud, carrier_hz=offset)
    idx = find_sync(demod)
    assert idx is not None  # synced despite the +12 kHz offset (de-rotated to DC first)
    fb = frame_bytes(demod[idx:])
    assert payload in fb  # full framed bytes recovered, not just a 12-byte preview


# ── framing_sweep --endurosat: carrier-recovered, CRC-16-gated EnduroSat extraction ───────────
def test_endurosat_sweep_recovers_off_dc_frame() -> None:
    from iq_analyze import framing_sweep

    from gfsk_ax25 import endurosat_link as el

    fs, baud, offset = 96_000.0, 9600.0, 8000.0
    payload = bytes(range(24))
    iq = modulate(el.frame_bits(payload), GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    guard = np.zeros(2000, dtype=np.complex64)
    seg = np.concatenate([guard, iq, guard]).astype(np.complex64)
    n = np.arange(len(seg))
    seg_off = (seg * np.exp(2j * np.pi * offset * n / fs)).astype(np.complex64)
    # Force the known carrier (the coarse grid would also find it): CRC-gated → decodes at 9600.
    res = {b: (nf, fr) for b, _c, nf, fr in framing_sweep(seg_off, fs, "endurosat", (baud,),
                                                          carriers=[offset])}
    nframes, frames = res[baud]
    assert nframes >= 1
    assert any(payload in f for f in frames)  # the EnduroSat payload is recovered


def test_endurosat_sweep_no_false_positive_on_noise() -> None:
    from iq_analyze import framing_sweep

    rng = np.random.default_rng(3)
    noise = (rng.normal(0, 1, 96_000) + 1j * rng.normal(0, 1, 96_000)).astype(np.complex64)
    res = {b: nf for b, _c, nf, _f in framing_sweep(noise, 96_000.0, "endurosat", (9600.0,),
                                                    carriers=[0.0])}
    assert res[9600.0] == 0  # CRC-16 gate → no garbage frames from noise

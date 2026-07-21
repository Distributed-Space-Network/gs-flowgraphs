"""iq_analyze capture loading — the cf32 (whole-pass) path via the metadata sidecar.

The VSA CSV path is the EnduroSat lab workflow; the cf32 path lets the same tool decode
the WHOLE pass the GR engines record (not just a VSA window)."""

from __future__ import annotations

from pathlib import Path

import iq_analyze
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


@pytest.mark.parametrize(
    ("sweep_baud", "expected"),
    [(False, (2400.0,)), (True, iq_analyze.DEFAULT_SWEEP_BAUDS)],
)
def test_grind_honors_baud_sweep_selection(
    monkeypatch: pytest.MonkeyPatch,
    sweep_baud: bool,
    expected: tuple[float, ...],
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        iq_analyze,
        "load_capture",
        lambda _path, _sample_rate: iq_analyze.Capture(
            iq=np.zeros(16, dtype=np.complex64),
            fs=48_000.0,
            center_hz=401_175_000.0,
            meta={},
        ),
    )
    monkeypatch.setattr(
        iq_analyze,
        "grind_pass",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    iq_analyze.analyze_file(
        "capture.cf32",
        symbol_rate=2400.0,
        sample_rate_hz=48_000.0,
        sweep_baud=sweep_baud,
        want_grind=True,
    )

    assert captured["bauds"] == expected


@pytest.mark.parametrize(
    ("channel_bw_hz", "expected_half_bw_hz"),
    [(0.0, iq_analyze.GRIND_INBAND_HALF_HZ), (4800.0, 2400.0)],
)
def test_grind_snr_band_honors_explicit_full_channel_bandwidth(
    channel_bw_hz: float,
    expected_half_bw_hz: float,
) -> None:
    assert iq_analyze.grind_inband_half_bw_hz(channel_bw_hz) == expected_half_bw_hz


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
    # DC (the cmd_101 / IPoS-TDsM bug: −17.9 kHz). The sweep must bring it back to DC — either via
    # the SPECTRAL-peak candidate or via the amplitude-weighted CFO (which locks the loud burst even
    # in noise). Either way the +8 kHz-offset frame must be recovered. Offset a +8 kHz.
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
    assert res[1200.0][1] >= 1  # decoded despite the +8 kHz offset (spectral or CFO recovery)
    # A wrong-baud demod must not forge an FCS-valid frame (the CRC gate holds).
    assert res.get(9600.0, (0, 0))[1] == 0


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


# ── decode_pass: whole-pass decode of bursty data BESIDE a strong continuous carrier ──────────
def _place(iq: np.ndarray, burst: np.ndarray, pos: int, carrier_hz: float, fs: float) -> None:
    m = np.arange(len(burst))
    iq[pos : pos + len(burst)] += (burst * np.exp(2j * np.pi * carrier_hz * m / fs)).astype(
        np.complex64
    )


def test_decode_pass_recovers_bursts_and_sweeps_baud() -> None:
    # decode_pass slides short windows over the WHOLE capture, recovering multiple bursts (at an
    # off-DC carrier) and deduping — and, given a baud list, finds them even when the caller's
    # labelled baud is wrong (a real pass labelled 9600 carried a 2400-baud bird).
    from iq_analyze import decode_pass

    from gfsk_ax25 import endurosat_link as el

    fs, baud = 96_000.0, 9600.0
    payloads = [bytes([k]) + bytes(range(20)) for k in (1, 2)]
    iq = np.zeros(int(fs * 4.0), np.complex64)
    for pl, pos in zip(payloads, (int(fs * 0.6), int(fs * 2.2)), strict=True):
        _place(iq, el.transmit(pl, fs, symbol_rate_hz=baud), pos, 10_000.0, fs)  # data at +10 kHz

    got = decode_pass(iq, fs, baud, ("endurosat",))["endurosat"]["frames"]
    assert len(got) >= 2  # both bursts recovered across the whole pass, deduped
    for pl in payloads:
        assert any(pl in f for f in got)
    # Baud sweep: same recovery when the true baud is one of several candidates; reports which won.
    swept = decode_pass(iq, fs, 1200.0, ("endurosat",), bauds=(2400.0, 9600.0))
    assert len(swept["endurosat"]["frames"]) >= 2
    assert 9600 in swept["endurosat"]["bauds"]


def test_strongest_burst_window_short_capture_no_crash() -> None:
    # Regression: a capture shorter than the 0.5 s baud-detect probe that contains a carrier used to
    # broadcast-crash (fixed-size hanning window vs a shorter seg), aborting the whole analyze run.
    from iq_analyze import SWEEP_BAUDS, _strongest_burst_window, detect_baud

    fs = 48_000.0
    n = np.arange(2000)  # ~42 ms @ 48 kHz, well under the 0.5 s probe
    iq = (5.0 * np.exp(2j * np.pi * 5000.0 * n / fs)).astype(np.complex64)
    got = _strongest_burst_window(iq, fs, None)
    assert got is not None  # returns a window instead of raising
    wseg, wcar = got
    assert abs(wcar - 5000.0) < 200.0
    assert len(detect_baud(wseg, fs)) == len(SWEEP_BAUDS)  # baud detect runs on the short window
    assert _strongest_burst_window(np.zeros(32, np.complex64), fs, None) is None  # tiny -> None


def test_detect_baud_finds_true_rate_from_preamble() -> None:
    # The label can be wrong; detect_baud demodulates at each candidate and scores the 0xAA-preamble
    # run. The TRUE baud yields a long clean run; wrong bauds give only noise-level runs. Here the
    # signal is 2400 baud (a long 0x55 preamble) but the "expected" label would be 9600.
    from iq_analyze import detect_baud

    fs, baud = 48_000.0, 2400.0
    pre = modulate(np.tile([0, 1], 300).astype(np.uint8),  # 600-bit alternating preamble
                   GfskParams(sample_rate_hz=fs, symbol_rate_hz=baud))
    ranked = dict(detect_baud(pre, fs, carrier_hz=0.0))
    assert ranked[2400.0] > 200  # true baud → long preamble run
    assert ranked[9600.0] < 40  # wrong baud → noise-level run
    assert max(ranked, key=ranked.get) == 2400.0  # 2400 wins the detection


def test_find_bursts_excludes_continuous_carrier() -> None:
    # A plain |iq| gate is DEFEATED by a strong continuous carrier (it pins |iq| high the whole pass
    # so nothing clears the threshold — the "0 bursts" on cmd_107). find_bursts(exclude_hz=)
    # detects on off-interferer spectral energy instead.
    from iq_analyze import find_bursts

    from gfsk_ax25 import endurosat_link as el

    fs, baud = 96_000.0, 9600.0
    n = np.arange(int(fs * 4.0))
    iq = (5.0 * np.exp(2j * np.pi * -25_000.0 * n / fs)).astype(np.complex64)  # strong continuous
    for pos in (int(fs * 0.6), int(fs * 2.2)):
        _place(iq, el.transmit(bytes(range(20)), fs, symbol_rate_hz=baud), pos, 10_000.0, fs)

    assert len(find_bursts(iq, fs, exclude_hz=-25_000.0)) >= 2  # bursts found beside the carrier
    assert len(find_bursts(iq, fs)) < 2  # plain magnitude gate is swamped by the continuous carrier


def test_decode_pass_no_false_frames_on_carrier_only() -> None:
    # A capture that is ONLY the continuous carrier (no data burst) must yield ZERO frames — the
    # CRC/FCS gate rejects the interferer, so the whole-pass sweep never forges frames from it.
    from iq_analyze import decode_pass

    fs, baud = 48_000.0, 9600.0
    n = np.arange(int(fs * 2.0))
    iq = (4.0 * np.exp(2j * np.pi * -18_000.0 * n / fs)).astype(np.complex64)
    res = decode_pass(iq, fs, baud, ("ax25", "endurosat"), exclude_hz=-18_000.0)
    assert res["ax25"]["frames"] == []
    assert res["endurosat"]["frames"] == []

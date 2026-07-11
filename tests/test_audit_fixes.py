"""Audit-round fixes (docs/10 follow-up): MED-1 race CRC-gating, MED-2 deaf TX sink,
LOW-1/2/4/6 GNU-Radio-side statics.

MED-1 and MED-2 are tested for real at the pure layer (compose.race_winner / framings
registry / configure_tx_sink parameter merge). The GNU-Radio wiring itself cannot import
here (no gnuradio on this box), so the wiring facts each fix depends on are locked with
source-level checks that fail if the fix is reverted.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import compose
import cubesat_gfsk_ax25_tx as txapp
import framings
import numpy as np
import pytest

from gfsk_ax25 import ax25
from gfsk_ax25 import framing as ax25_framing

_APPS = Path(__file__).resolve().parent.parent / "apps"


# ══ MED-1: only CRC/FCS/RS-gated framings may declare an engine-race win ═══════════════════


def test_crc_gated_registry_excludes_kiss():
    gated = framings.crc_gated_framings()
    assert set(gated) <= set(framings.local_framings())
    assert "kiss" not in gated  # KISS has no checksum — must never gate off gr-satellites
    assert {"ax25", "endurosat", "ccsds_tm"} == set(gated)


@pytest.mark.parametrize(
    ("label", "gated"),
    [
        ("ax25", True), ("AX.25 G3RUH", True), ("AX.25", True), ("APRS", True),
        ("endurosat", True), ("EnduroSat", True), ("AirMAC", True),
        ("ccsds_tm", True),
        ("kiss", False), ("KISS", False),          # no checksum
        ("SLIP", False), ("USP", False),           # not local at all
        ("AX100 ASM+Golay", False), ("", False), (None, False),
    ],
)
def test_is_crc_gated_accepts_any_vocabulary(label, gated):
    assert framings.is_crc_gated(label) is gated


def test_noise_decoded_kiss_frames_do_not_flip_the_valve():
    # The original loss scenario: for a catalogued bird whose backend framing is KISS, the
    # first noise drain measurably yields chance KISS "frames" (no checksum). Pre-fix, ANY
    # our-engine frame declared "ours" the winner and gated off gr-satellites for the whole
    # pass. Now: a KISS hit must produce NO winner — the race keeps running.
    noise_frames = []
    matched = None
    for seed in range(20):  # strict KISS passes ~2 chance frames per 2400-byte noise window
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, 2400 * 8).astype(np.uint8)
        frames, m = framings.deframe(bits, "KISS")
        if frames:
            noise_frames, matched = frames, m
            break
    assert noise_frames, "expected chance KISS frames out of noise (the MED-1 premise)"
    assert matched == "kiss"
    # Drain 1: our engine decoded garbage KISS, gr-satellites nothing → NO winner.
    assert compose.race_winner([matched], grsat_produced=False) is None
    # Drain N (later, strong burst): the REAL decoder produces a frame → it wins, despite our
    # engine having "produced" first. Pre-fix it was already starved and this frame was lost.
    assert compose.race_winner([matched], grsat_produced=True) == "grsatellites"
    # And with no production at all, still no winner.
    assert compose.race_winner([], grsat_produced=False) is None


def test_fcs_valid_ax25_frame_does_flip_the_valve():
    body = ax25.encode_ui(dest="DSN", src="ISS", info=b"fcs-valid")
    bits = ax25_framing.encode(body, preamble_flags=16)
    frames, matched = framings.deframe(bits, "AX.25 G3RUH")
    assert body in frames and matched == "ax25"
    assert compose.race_winner([matched], grsat_produced=False) == "ours"
    # Tie within one drain still goes to OUR engine — but only because it is CRC-gated.
    assert compose.race_winner([matched], grsat_produced=True) == "ours"
    # A mixed drain (KISS garbage + a real FCS-valid frame) also wins.
    assert compose.race_winner(["kiss", matched], grsat_produced=True) == "ours"


def test_plan_decode_race_prediction_agrees_with_race_winner():
    # P0-2's lesson: the plan and the engine race logic must not drift. Both must derive
    # from the SAME registry property (framings.is_crc_gated).
    for label in ("ax25", "endurosat", "ccsds_tm", "kiss", "AX.25 G3RUH", "KISS"):
        plan = compose.plan_decode(
            {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": label}, catalogued=True)
        assert plan.our_crc_gated is framings.is_crc_gated(label)
        assert plan.race  # catalogued + our engine → both run
        ours_can_win = compose.race_winner([framings.normalize_framing(label)],
                                           grsat_produced=True) == "ours"
        assert plan.race_ours_can_win is ours_can_win


def test_kiss_race_plan_is_race_but_ours_cannot_gate():
    plan = compose.plan_decode(
        {"modulation": "gfsk", "symbol_rate_hz": 9600, "framing": "KISS"}, catalogued=True)
    assert plan.our_engine and plan.race and not plan.race_ours_can_win
    assert "ungated" in plan.describe()


def test_engine_race_wiring_consults_race_winner():
    # The GR-side valve logic (not importable here) must delegate to compose.race_winner and
    # feed it the framing that matched — reverting to "any our-frame wins" fails this.
    src = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")
    assert "compose.race_winner(" in src
    assert "race_framing" in src
    drain = src[src.index("def drain_frames(self) -> list[tuple[str, bytes]]"):]
    drain = drain[: drain.index("def _gate_off")]
    assert "compose.race_winner(" in drain
    assert 'if our_frames:\n                self._winner = "ours"' not in drain


# ══ MED-2: the default cubesat TX engine must configure the SDR front-end ══════════════════


class _FakeSoapyDevice:
    """Records the raw SoapySDR.Device calls configure_tx_sink makes."""

    def __init__(self, *, bandwidth_raises: bool = False) -> None:
        self.calls: list[tuple] = []
        self._bandwidth_raises = bandwidth_raises

    def setAntenna(self, direction, channel, name):  # noqa: N802 — SoapySDR API casing
        self.calls.append(("setAntenna", direction, channel, name))

    def setGainMode(self, direction, channel, automatic):  # noqa: N802
        self.calls.append(("setGainMode", direction, channel, automatic))

    def setGain(self, direction, channel, *args):  # noqa: N802
        self.calls.append(("setGain", direction, channel, *args))

    def setFrequencyCorrection(self, direction, channel, ppm):  # noqa: N802
        self.calls.append(("setFrequencyCorrection", direction, channel, ppm))

    def setBandwidth(self, direction, channel, bw):  # noqa: N802
        if self._bandwidth_raises:
            msg = "driver has no TX bandwidth setter"
            raise RuntimeError(msg)
        self.calls.append(("setBandwidth", direction, channel, bw))


_TX = 1  # SoapySDR.SOAPY_SDR_TX


def _clear_sdr_env(monkeypatch) -> None:
    for name in ("GS_SDR_ANTENNA", "GS_SDR_GAIN_DB", "GS_SDR_GAINS", "GS_SDR_AGC",
                 "GS_SDR_LO_OFFSET", "GS_SDR_PPM", "GS_SDR_DC_REMOVAL", "GS_SDR_CAPTURE_RATE",
                 "GS_SDR_TX_ANTENNA", "GS_SDR_TX_GAIN_DB", "GS_SDR_TX_GAINS"):
        monkeypatch.delenv(name, raising=False)


def test_tx_sink_never_left_at_zero_gain(monkeypatch):
    # The deaf-TX fix itself: with no params and no env, the sink must still get the sane
    # manual default gain (the 0 dB default radiates nothing), TX-direction-addressed.
    _clear_sdr_env(monkeypatch)
    dev = _FakeSoapyDevice()
    applied = txapp.configure_tx_sink(dev, _TX, None, 96_000.0)
    assert ("setGain", _TX, 0, 30.0) in dev.calls
    assert applied["gain_db"] == 30.0 and applied.get("gain_default") is True


def test_tx_sink_analog_bandwidth_is_sample_rate_not_channel(monkeypatch):
    # Station hardware rule (XTRX analog filter floor ~0.8 MHz): analog BW = the SAMPLE rate.
    _clear_sdr_env(monkeypatch)
    dev = _FakeSoapyDevice()
    txapp.configure_tx_sink(dev, _TX, {}, 2_048_000.0)
    assert ("setBandwidth", _TX, 0, 2_048_000.0) in dev.calls


def test_tx_sink_merges_tx_env_and_per_pass_tx_params(monkeypatch):
    # R-22: TX settings come from TX-explicit sources ONLY — per-pass sdr_tx_*
    # wins over GS_SDR_TX_* env; ppm (direction-neutral) still applies TX-side.
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_TX_ANTENNA", "BAND1")
    monkeypatch.setenv("GS_SDR_TX_GAIN_DB", "20")
    monkeypatch.setenv("GS_SDR_PPM", "-1.5")
    dev = _FakeSoapyDevice()
    applied = txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gain_db": 12.0}, 96_000.0)
    assert ("setAntenna", _TX, 0, "BAND1") in dev.calls
    assert ("setGain", _TX, 0, 12.0) in dev.calls
    assert ("setGain", _TX, 0, 20.0) not in dev.calls
    assert ("setFrequencyCorrection", _TX, 0, -1.5) in dev.calls
    assert applied["gain_db"] == 12.0 and "gain_default" not in applied


def test_tx_sink_never_receives_rx_oriented_names(monkeypatch):
    # THE R-22 repro: station RX env (LNAW antenna, LNA/TIA/PGA staging) and
    # generic per-pass RX keys must NOT reach a TX endpoint — on LMS7/XTRX
    # setAntenna(TX, LNAW)/setGain(TX, "LNA", ..) raise and kill the pass.
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_ANTENNA", "LNAW")
    monkeypatch.setenv("GS_SDR_GAINS", "LNA=30,TIA=9,PGA=3")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "45")
    dev = _FakeSoapyDevice()
    applied = txapp.configure_tx_sink(
        dev, _TX, {"sdr_antenna": "LNAL", "sdr_gains": {"LNA": 10}, "sdr_gain_db": 45.0},
        96_000.0,
    )
    assert not any(c[0] == "setAntenna" for c in dev.calls)
    assert not any(c[0] == "setGain" and isinstance(c[3], str) for c in dev.calls)
    # ... and the deaf-TX default still applies instead (never 0 dB).
    assert applied["gain_db"] == 30.0 and applied.get("gain_default") is True


def test_tx_sink_per_element_gains(monkeypatch):
    _clear_sdr_env(monkeypatch)
    dev = _FakeSoapyDevice()
    txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": 40, "IAMP": 6}}, 96_000.0)
    assert ("setGain", _TX, 0, "PAD", 40.0) in dev.calls
    assert ("setGain", _TX, 0, "IAMP", 6.0) in dev.calls


def test_tx_sink_survives_driver_without_bandwidth_setter(monkeypatch):
    _clear_sdr_env(monkeypatch)
    dev = _FakeSoapyDevice(bandwidth_raises=True)
    applied = txapp.configure_tx_sink(dev, _TX, None, 96_000.0)  # must not raise
    assert applied["gain_db"] == 30.0


def test_soapy_sink_actually_calls_configure_tx_sink():
    # The hardware path itself can't run here; lock the wiring so a revert (raw sink with
    # only rate+frequency — the deaf transmitter) fails.
    src = inspect.getsource(txapp._soapy_sink)
    assert "configure_tx_sink(" in src
    # and the app threads the pass params through to it (F-03 added the
    # on_first_accept hook to the same call)
    assert "_sink_iq, args, iq, params" in inspect.getsource(txapp.amain)


def test_sink_iq_file_path_accepts_params(tmp_path):
    import argparse
    cap = tmp_path / "tx.cf32"
    args = argparse.Namespace(sample_rate=96_000, sdr_args=f"file:{cap}")
    txapp._sink_iq(args, np.zeros(16, np.complex64), {"sdr_gain_db": 10})
    assert cap.stat().st_size == 16 * 8


# ══ LOW-1/2/4/6: GNU-Radio-only fixes — source-level regression locks ══════════════════════


def test_low1_qam_docstring_promises_only_buildable_orders():
    src = (_APPS / "gnuradio_hirate.py").read_text(encoding="utf-8")
    assert "16/32/64/128/256" not in src  # 32/128 raise in GR 3.10 with GRAY_CODE
    assert "16/64/256" in src


@pytest.mark.parametrize("fname", ["gnuradio_gfsk.py", "gnuradio_hirate.py"])
def test_low2_rx_applies_inverted_pre_diff_code(fname):
    # GR's generic_demod applies mod_codes.invert_code(pre_diff_code) on RX; applying the
    # FORWARD code is wrong the moment a pre-diff-coded constellation lands.
    src = (_APPS / fname).read_text(encoding="utf-8")
    assert "digital.map_bb(digital.mod_codes.invert_code(constel.pre_diff_code()))" in src
    assert "digital.map_bb(constel.pre_diff_code())" not in src


def test_low4_afsk_connects_nothing_before_all_blocks_exist():
    # A ctor raise inside connect_gfsk_demod is caught by _build_fallbacks, but a pre-connected
    # src→audio→xlate tap would leave xlate dangling → tb.start() aborts → RECORDING lost.
    # The FSK chain must therefore be constructed+wired BEFORE anything connects to src.
    src = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
    fn = src[src.index("def connect_afsk_demod"):]
    fn = fn[: fn.index("\ndef ")]
    assert "tb.connect(src, audio, xlate)" in fn
    assert fn.index("connect_gfsk_demod(") < fn.index("tb.connect(src, audio, xlate)")


def test_low6_fm_tx_refuses_non_multiple_sdr_rate():
    # interp = rate // 48000 truncated (2.048 M → 2.016 M fed to a 2.048 M sink, ~1.6 % audio
    # mistune). The app must validate divisibility and refuse loudly.
    src = (_APPS / "amateur_fm_narrowband_tx.py").read_text(encoding="utf-8")
    build = src[src.index("def build_top_block"):]
    build = build[: build.index("\nasync def amain")]
    assert "% _AUDIO_RATE_HZ" in build and "raise ValueError" in build
    assert "max(1, args.sample_rate // _AUDIO_RATE_HZ)" not in build  # the silent truncation

"""Unit tests for the SoapySDR front-end config helper.

``configure_soapy_source`` is import-safe (no gnuradio/SoapySDR), so we drive it
against a fake source that records the gr-soapy calls it would make. This is the
one piece of the GNU-Radio engines we CAN test without GNU Radio.
"""

from __future__ import annotations

import math

import pytest
from _soapy import (
    DEFAULT_LO_OFFSET_HZ,
    apply_corrections,
    auto_lo_offset,
    capture_plan,
    configure_soapy_source,
    lo_phase_inc,
    merge_sdr_params,
    resample_ratio,
    retune_source,
    sdr_env,
    tune_below,
    tune_source,
)


def test_auto_lo_offset_uses_default_only_when_requested() -> None:
    sdr = 2_048_000.0
    d = DEFAULT_LO_OFFSET_HZ  # 100 kHz
    # The rotator RX engines pass default_offset_hz → unset/0 AUTO-USES the 100 kHz default when
    # the wideband capture has room to hold the offset carrier (dodges the DC/LO spike in software).
    assert auto_lo_offset(sdr, 48_000.0, 0.0, default_offset_hz=d) == 100_000.0
    assert auto_lo_offset(sdr, 200_000.0, 0.0, default_offset_hz=d) == 100_000.0
    # WITHOUT default_offset_hz (the amateur-FM engine, still on the hardware BB split) an unset
    # offset stays ON-CENTER (0) — a forced offset would push FM off-band on the XTRX no-op BB.
    assert auto_lo_offset(sdr, 48_000.0, 0.0) == 0.0
    assert auto_lo_offset(sdr, 200_000.0, 0.0) == 0.0
    # An explicit GS_SDR_LO_OFFSET wins over the default and is honored as-is when it fits the band.
    assert auto_lo_offset(sdr, 48_000.0, 250_000.0, default_offset_hz=d) == 250_000.0
    assert auto_lo_offset(sdr, 48_000.0, 250_000.0) == 250_000.0
    # An explicit offset larger than the band can hold falls back to ON-CENTER (0), NOT the band
    # edge — the historical safe behavior (a hardware-split FM caller can't dodge an off-band offset
    # on the XTRX; a min(off,room) clamp regressed it). room = 1_024_000 - 24_000 = 1_000_000.
    assert auto_lo_offset(sdr, 48_000.0, 2_000_000.0, default_offset_hz=d) == 0.0
    assert auto_lo_offset(sdr, 48_000.0, 2_000_000.0) == 0.0  # FM (no-default) caller: also 0


def test_auto_lo_offset_on_center_when_no_headroom() -> None:
    # Capturing directly at the channel rate (RTL-class, no decimation) → no room for an offset,
    # so ON-CENTER (0) even for a rotator engine; GS_SDR_DC_REMOVAL notches the spike.
    assert auto_lo_offset(48_000.0, 48_000.0, 0.0, default_offset_hz=DEFAULT_LO_OFFSET_HZ) == 0.0
    assert auto_lo_offset(48_000.0, 48_000.0, 100_000.0) == 0.0


def test_tune_below_tunes_lo_below_carrier() -> None:
    # tune_below = a PLAIN tune to (center - lo_offset) (no RF/BB split), so the carrier lands at
    # +lo_offset at baseband for the rotator to bring to DC.
    src = FakeSoapy()
    tune_below(src, 401_000_000.0, 100_000.0)
    assert src.frequencies == [(0, 400_900_000.0)]
    src2 = FakeSoapy()
    tune_below(src2, 401_000_000.0, 0.0)  # no offset → plain on-center tune
    assert src2.frequencies == [(0, 401_000_000.0)]


def test_lo_phase_inc_shifts_carrier_to_dc() -> None:
    # phase_inc = -2π(lo_offset + doppler)/sdr_rate. A +100 kHz carrier at 2.048 Msps:
    assert lo_phase_inc(2_048_000.0, 100_000.0, 0.0) == \
        -2.0 * math.pi * 100_000.0 / 2_048_000.0
    # Doppler adds to the offset (positive doppler = carrier higher = shift down more):
    assert lo_phase_inc(2_048_000.0, 100_000.0, 1_200.0) == \
        -2.0 * math.pi * 101_200.0 / 2_048_000.0
    # Zero offset + zero doppler → no rotation.
    assert lo_phase_inc(2_048_000.0, 0.0, 0.0) == 0.0


class FakeSoapy:
    """Records the gr-soapy source/sink calls the helper makes."""

    def __init__(self, *, named_freq_raises: bool = False) -> None:
        self.antenna: tuple[int, str] | None = None
        self.gain_mode: tuple[int, bool] | None = None
        self.gains: list[tuple] = []  # (channel, name?, value)
        self.frequencies: list[tuple] = []  # (channel, name?, value)
        self.ppm: tuple[int, float] | None = None
        self.dc_offset_mode: tuple[int, bool] | None = None
        self._named_freq_raises = named_freq_raises

    def set_antenna(self, channel: int, name: str) -> None:
        self.antenna = (channel, name)

    def set_gain_mode(self, channel: int, automatic: bool) -> None:
        self.gain_mode = (channel, automatic)

    def set_gain(self, channel: int, *args: object) -> None:
        self.gains.append((channel, *args))

    def set_frequency(self, channel: int, *args: object) -> None:
        # Overload 2 is (channel, name, freq); a driver without RF/BB named
        # components raises on it — emulate that when asked.
        if self._named_freq_raises and len(args) == 2:
            msg = "no such frequency component"
            raise ValueError(msg)
        self.frequencies.append((channel, *args))

    def set_frequency_correction(self, channel: int, ppm: float) -> None:
        self.ppm = (channel, ppm)

    def set_dc_offset_mode(self, channel: int, enable: bool) -> None:
        self.dc_offset_mode = (channel, enable)


def test_default_gain_applied_when_nothing_configured() -> None:
    src = FakeSoapy()
    applied = configure_soapy_source(src, None)
    # The whole point: no params must NOT leave the front-end at 0 dB.
    assert src.gains == [(0, 30.0)]
    assert applied == {"gain_db": 30.0, "gain_default": True}
    assert src.antenna is None and src.gain_mode is None


def test_overall_gain_and_antenna() -> None:
    src = FakeSoapy()
    applied = configure_soapy_source(src, {"sdr_gain_db": 42, "sdr_antenna": "LNAL"})
    assert src.antenna == (0, "LNAL")
    assert src.gains == [(0, 42.0)]
    assert applied == {"antenna": "LNAL", "gain_db": 42.0}


def test_per_element_gains() -> None:
    src = FakeSoapy()
    # The SatNOGS-style "LNA=20,TIA=6,PGA=0" surface, as a dict.
    applied = configure_soapy_source(src, {"sdr_gains": {"LNA": 20, "TIA": 6, "PGA": 0}})
    assert (0, "LNA", 20.0) in src.gains
    assert (0, "TIA", 6.0) in src.gains
    assert (0, "PGA", 0.0) in src.gains
    assert applied["gains"] == {"LNA": 20.0, "TIA": 6.0, "PGA": 0.0}
    # Per-element gains count as configuring gain -> no default added.
    assert "gain_default" not in applied


def test_per_element_gains_win_over_overall_gain() -> None:
    # docs/J LOW-3: SoapySDR's overall setGain RE-DISTRIBUTES across elements,
    # so applying it after the per-element staging overrides the staging. The
    # documented precedence (sdr_env: GAINS wins over GAIN_DB) means the overall
    # gain must not be applied at all when per-element gains took effect.
    src = FakeSoapy()
    applied = configure_soapy_source(
        src, {"sdr_gains": {"PAD": 40, "IAMP": 6}, "sdr_gain_db": 20.0}
    )
    assert (0, "PAD", 40.0) in src.gains
    assert (0, "IAMP", 6.0) in src.gains
    assert (0, 20.0) not in src.gains  # overall NOT applied on top of the staging
    assert applied["gains"] == {"PAD": 40.0, "IAMP": 6.0}
    assert "gain_db" not in applied and "gain_default" not in applied


def test_env_gains_plus_gain_db_apply_only_the_staging(monkeypatch) -> None:
    # Same precedence through the station-env merge path (the probe's M1 seam).
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAINS", "PAD=40,IAMP=6")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "20")
    src = FakeSoapy()
    configure_soapy_source(src, merge_sdr_params({}))
    assert (0, "PAD", 40.0) in src.gains and (0, "IAMP", 6.0) in src.gains
    assert all(len(call) != 2 for call in src.gains)  # no overall set_gain call


def test_overall_gain_still_applies_when_gains_dict_is_all_garbage() -> None:
    # A gains dict with no usable entry must not eat the overall gain.
    src = FakeSoapy()
    applied = configure_soapy_source(
        src, {"sdr_gains": {"LNA": "loud", 7: 3}, "sdr_gain_db": 18.0}
    )
    assert src.gains == [(0, 18.0)]
    assert applied["gain_db"] == 18.0 and "gains" not in applied


def test_agc_on_suppresses_default_gain() -> None:
    src = FakeSoapy()
    applied = configure_soapy_source(src, {"sdr_agc": True})
    assert src.gain_mode == (0, True)
    assert src.gains == []  # AGC handles gain; no manual default forced
    assert "gain_default" not in applied


def test_agc_off_still_gets_default_gain() -> None:
    src = FakeSoapy()
    configure_soapy_source(src, {"sdr_agc": False})
    assert src.gain_mode == (0, False)
    assert src.gains == [(0, 30.0)]  # AGC explicitly off -> manual default applies


def test_default_gain_db_none_leaves_sdr_untouched() -> None:
    src = FakeSoapy()
    applied = configure_soapy_source(src, {}, default_gain_db=None)
    assert src.gains == []
    assert applied == {}


def test_bad_types_ignored() -> None:
    src = FakeSoapy()
    configure_soapy_source(
        src,
        {"sdr_antenna": 5, "sdr_gain_db": True, "sdr_gains": {"LNA": "loud", 7: 3}},
        default_gain_db=None,
    )
    # bool is not a number; non-str antenna ignored; bad element entries skipped.
    assert src.antenna is None
    assert src.gains == []


def test_channel_override() -> None:
    src = FakeSoapy()
    configure_soapy_source(src, {"sdr_gain_db": 12.0, "sdr_antenna": "RX2"}, channel=1)
    assert src.antenna == (1, "RX2")
    assert src.gains == [(1, 12.0)]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_per_element_gain_is_refused_before_the_driver(bad) -> None:
    """VR-004/VR-011 (DS-016 root): configure_soapy_source is the ONE choke point every engine's
    gain application goes through — a NaN/inf per-element gain must raise (fail the spawn closed),
    never reach set_gain. This closes the amateur_fm_narrowband_tx bypass (it calls configure
    directly, skipping named_tx_gains/verify_named_tx_gains) and NaN RX gains."""
    src = FakeSoapy()
    with pytest.raises(ValueError, match="non-finite"):
        configure_soapy_source(src, {"sdr_gains": {"PAD": bad}}, default_gain_db=None)
    assert src.gains == [], "a non-finite gain reached the driver"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_overall_gain_is_refused_before_the_driver(bad) -> None:
    src = FakeSoapy()
    with pytest.raises(ValueError, match="non-finite"):
        configure_soapy_source(src, {"sdr_gain_db": bad}, default_gain_db=None)
    assert src.gains == []


def test_env_non_finite_gains_are_ignored_as_malformed(monkeypatch) -> None:
    """VR-004: 'PAD=nan' / GS_SDR_GAIN_DB='inf' parse without ValueError in bare float() — they
    must be dropped at env parse (like any malformed entry), not forwarded toward the driver."""
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAINS", "PAD=nan,IAMP=6")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "inf")
    env = sdr_env()
    assert env["gains"] == {"IAMP": 6.0}
    assert env["gain_db"] is None


# --------------------------------------------------------------------------
# Station-wide GS_SDR_* env settings + LO-offset / ppm / DC-removal helpers
# --------------------------------------------------------------------------

_SDR_ENV_VARS = (
    "GS_SDR_ANTENNA",
    "GS_SDR_GAIN_DB",
    "GS_SDR_GAINS",
    "GS_SDR_AGC",
    "GS_SDR_LO_OFFSET",
    "GS_SDR_PPM",
    "GS_SDR_DC_REMOVAL",
    "GS_SDR_CAPTURE_RATE",
    "GS_SDR_TX_ANTENNA",
    "GS_SDR_TX_GAIN_DB",
    "GS_SDR_TX_GAINS",
)


def _clear_sdr_env(monkeypatch) -> None:
    for name in _SDR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_sdr_env_defaults_when_unset(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    env = sdr_env()
    assert env == {
        "antenna": None,
        "gain_db": None,
        "gains": None,
        "agc": False,
        "lo_offset_hz": 0.0,
        "ppm": 0.0,
        "dc_removal": False,
        "capture_rate_hz": 2_048_000.0,  # default = SatNOGS rate
        "tx_antenna": None,  # R-22: TX has its OWN keys; RX names never leak
        "tx_gain_db": None,
        "tx_gains": None,
    }


def test_sdr_env_reads_environment(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_ANTENNA", "LNAW")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "44.5")
    monkeypatch.setenv("GS_SDR_AGC", "true")
    monkeypatch.setenv("GS_SDR_LO_OFFSET", "250000")
    monkeypatch.setenv("GS_SDR_PPM", "-1.5")
    monkeypatch.setenv("GS_SDR_DC_REMOVAL", "1")
    env = sdr_env()
    assert env["antenna"] == "LNAW"
    assert env["gain_db"] == 44.5
    assert env["agc"] is True
    assert env["lo_offset_hz"] == 250000.0
    assert env["ppm"] == -1.5
    assert env["dc_removal"] is True


def test_sdr_env_ignores_non_numeric(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAIN_DB", "loud")
    monkeypatch.setenv("GS_SDR_LO_OFFSET", "")
    env = sdr_env()
    assert env["gain_db"] is None
    assert env["lo_offset_hz"] == 0.0


def test_merge_sdr_params_fills_from_env(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_ANTENNA", "LNAH")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "30")
    merged = merge_sdr_params(None)
    assert merged["sdr_antenna"] == "LNAH"
    assert merged["sdr_gain_db"] == 30.0
    assert merged["sdr_agc"] is False


def test_merge_sdr_params_per_pass_wins(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_ANTENNA", "LNAH")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "30")
    merged = merge_sdr_params({"sdr_antenna": "LNAW", "sdr_gain_db": 12.0})
    assert merged["sdr_antenna"] == "LNAW"  # per-pass overrides station env
    assert merged["sdr_gain_db"] == 12.0


def test_sdr_env_parses_per_element_gains(monkeypatch) -> None:
    # The SatNOGS GAIN_MODE="Settings Field" / OTHER_SETTINGS="LNA=30,TIA=9,PGA=3" surface.
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAINS", "LNA=30,TIA=9,PGA=3")
    assert sdr_env()["gains"] == {"LNA": 30.0, "TIA": 9.0, "PGA": 3.0}


def test_sdr_env_gains_skips_malformed(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAINS", "LNA=30, bogus ,TIA=loud,PGA=3")
    assert sdr_env()["gains"] == {"LNA": 30.0, "PGA": 3.0}


def test_merge_sdr_params_injects_per_element_gains(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_GAINS", "LNA=30,TIA=9,PGA=3")
    merged = merge_sdr_params(None)
    assert merged["sdr_gains"] == {"LNA": 30.0, "TIA": 9.0, "PGA": 3.0}
    # And configure_soapy_source consumes that dict as per-element set_gain calls.
    src = FakeSoapy()
    configure_soapy_source(src, merged)
    assert (0, "LNA", 30.0) in src.gains
    assert (0, "TIA", 9.0) in src.gains
    assert (0, "PGA", 3.0) in src.gains


def test_tune_source_no_offset_is_plain_tune() -> None:
    src = FakeSoapy()
    tune_source(src, 401_000_000.0, 0.0)
    assert src.frequencies == [(0, 401_000_000.0)]


def test_tune_source_lo_offset_splits_rf_bb() -> None:
    src = FakeSoapy()
    tune_source(src, 401_000_000.0, 250_000.0)
    assert src.frequencies == [
        (0, "RF", 401_250_000.0),
        (0, "BB", -250_000.0),
    ]


def test_tune_source_falls_back_when_named_unsupported() -> None:
    src = FakeSoapy(named_freq_raises=True)
    tune_source(src, 401_000_000.0, 250_000.0)
    # RF/BB rejected -> a single plain tune at the carrier (no offset benefit).
    assert src.frequencies == [(0, 401_000_000.0)]


def test_retune_source_preserves_offset() -> None:
    src = FakeSoapy()
    retune_source(src, 401_000_000.0, 250_000.0, 1_200.0)
    assert src.frequencies == [(0, "RF", 401_251_200.0)]


def test_retune_source_no_offset() -> None:
    src = FakeSoapy()
    retune_source(src, 401_000_000.0, 0.0, 1_200.0)
    assert src.frequencies == [(0, 401_001_200.0)]


def test_apply_corrections_sets_ppm_and_dc() -> None:
    src = FakeSoapy()
    apply_corrections(src, ppm=-2.0, dc_removal=True)
    assert src.ppm == (0, -2.0)
    assert src.dc_offset_mode == (0, True)


def test_apply_corrections_noop_when_zero() -> None:
    src = FakeSoapy()
    apply_corrections(src, ppm=0.0, dc_removal=False)
    assert src.ppm is None
    assert src.dc_offset_mode is None


# --------------------------------------------------------------------------
# Capture-rate / decimation (XTRX can't stream the narrow channel rate)
# --------------------------------------------------------------------------

def test_sdr_env_capture_rate_default(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    assert sdr_env()["capture_rate_hz"] == 2_048_000.0


def test_sdr_env_capture_rate_override(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_CAPTURE_RATE", "2400000")
    assert sdr_env()["capture_rate_hz"] == 2_400_000.0


def test_sdr_env_capture_rate_zero_disables(monkeypatch) -> None:
    _clear_sdr_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_CAPTURE_RATE", "0")
    assert sdr_env()["capture_rate_hz"] == 0.0


def test_capture_plan_decimates_when_capture_above_channel() -> None:
    sdr_rate, decimate = capture_plan(2_048_000.0, 48_000.0)
    assert sdr_rate == 2_048_000.0 and decimate is True


def test_capture_plan_disabled_when_zero() -> None:
    sdr_rate, decimate = capture_plan(0.0, 48_000.0)
    assert sdr_rate == 48_000.0 and decimate is False


def test_capture_plan_no_decimation_when_equal() -> None:
    sdr_rate, decimate = capture_plan(48_000.0, 48_000.0)
    assert sdr_rate == 48_000.0 and decimate is False


def test_resample_ratio_exact() -> None:
    # 2.048 Msps -> 48 ksps is exactly 3/128.
    assert resample_ratio(2_048_000.0, 48_000.0) == (3, 128)
    # and it actually reconstructs the channel rate
    interp, decim = resample_ratio(2_048_000.0, 48_000.0)
    assert 2_048_000.0 * interp / decim == 48_000.0

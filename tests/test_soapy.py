"""Unit tests for the SoapySDR front-end config helper.

``configure_soapy_source`` is import-safe (no gnuradio/SoapySDR), so we drive it
against a fake source that records the gr-soapy calls it would make. This is the
one piece of the GNU-Radio engines we CAN test without GNU Radio.
"""

from __future__ import annotations

from _soapy import configure_soapy_source


class FakeSoapy:
    """Records the gr-soapy source/sink calls the helper makes."""

    def __init__(self) -> None:
        self.antenna: tuple[int, str] | None = None
        self.gain_mode: tuple[int, bool] | None = None
        self.gains: list[tuple] = []  # (channel, name?, value)

    def set_antenna(self, channel: int, name: str) -> None:
        self.antenna = (channel, name)

    def set_gain_mode(self, channel: int, automatic: bool) -> None:
        self.gain_mode = (channel, automatic)

    def set_gain(self, channel: int, *args: object) -> None:
        self.gains.append((channel, *args))


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

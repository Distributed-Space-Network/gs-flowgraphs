"""SoapySDR front-end configuration shared by the GNU Radio engines.

GNU Radio (and gr-satellites) give us the *demodulation*, but never the SDR
front-end setup — a SoapySDR source/sink defaults to ~0 dB gain and an arbitrary
antenna port, so a perfectly correct flowgraph still receives nothing on real
hardware (LimeSDR/XTRX/USRP). ``configure_soapy_source`` applies the gain and
antenna from the directive's ``waveform_parameters`` (Document C C.5.5.2),
mirroring how SatNOGS drives its flowgraphs (``--antenna``/``--gain-mode``/
``--other-settings``) — but as our own code, learnt from that interface, not
copied from the AGPL gr-satnogs scripts.

Import-safe: it only calls methods on the ``src`` object passed in, so it needs
no ``gnuradio``/``SoapySDR`` import and is unit-tested against a fake source.

Honoured ``params`` keys (all optional):
  * ``sdr_antenna``  (str)             -- e.g. "LNAL", "RX2"; ``src.set_antenna``
  * ``sdr_agc``      (bool)            -- hardware AGC on/off; ``set_gain_mode``
  * ``sdr_gain_db``  (number)          -- overall gain; ``src.set_gain(ch, db)``
  * ``sdr_gains``    (dict[str,num])   -- per-element gains, e.g.
                                          {"LNA": 20, "TIA": 6, "PGA": 0}
When none of the gain keys are given and AGC is not enabled, ``default_gain_db``
is applied so the front-end is never stuck at 0 dB (the common "hears nothing"
trap). Pass ``default_gain_db=None`` to leave the SDR default untouched.
"""

from __future__ import annotations

from typing import Any, Protocol


class _SoapyEndpoint(Protocol):
    """The subset of a gr-soapy source/sink this helper drives (both implement it)."""

    def set_antenna(self, channel: int, name: str) -> object: ...
    def set_gain_mode(self, channel: int, automatic: bool) -> object: ...
    def set_gain(self, channel: int, *args: object) -> object: ...


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def configure_soapy_source(
    src: _SoapyEndpoint,
    params: dict[str, Any] | None,
    *,
    channel: int = 0,
    default_gain_db: float | None = 30.0,
) -> dict[str, Any]:
    """Apply antenna + gain settings from ``params`` to a SoapySDR ``src``.

    Returns a dict describing what was applied (handy for a log line). Only keys
    that are present (and well-typed) take effect; everything else is ignored.
    """
    p = params or {}
    applied: dict[str, Any] = {}

    antenna = p.get("sdr_antenna")
    if isinstance(antenna, str) and antenna:
        src.set_antenna(channel, antenna)
        applied["antenna"] = antenna

    agc = p.get("sdr_agc")
    agc_on = isinstance(agc, bool) and agc
    if isinstance(agc, bool):
        src.set_gain_mode(channel, agc)
        applied["agc"] = agc

    gave_gain = False
    per_element = p.get("sdr_gains")
    if isinstance(per_element, dict):
        elems: dict[str, float] = {}
        for name, val in per_element.items():
            if isinstance(name, str) and _is_number(val):
                src.set_gain(channel, name, float(val))
                elems[name] = float(val)
                gave_gain = True
        if elems:
            applied["gains"] = elems

    overall = p.get("sdr_gain_db")
    if _is_number(overall):
        src.set_gain(channel, float(overall))  # type: ignore[arg-type]
        applied["gain_db"] = float(overall)  # type: ignore[arg-type]
        gave_gain = True

    # Nothing configured the gain and AGC is off -> apply a sane manual default
    # rather than leave the front-end at 0 dB.
    if not gave_gain and not agc_on and default_gain_db is not None:
        src.set_gain(channel, float(default_gain_db))
        applied["gain_db"] = float(default_gain_db)
        applied["gain_default"] = True

    return applied


__all__ = ["configure_soapy_source"]

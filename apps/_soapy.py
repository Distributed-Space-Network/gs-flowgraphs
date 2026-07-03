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
                                          (applied only when ``sdr_gains`` is absent
                                          -- per-element staging wins)
  * ``sdr_gains``    (dict[str,num])   -- per-element gains, e.g.
                                          {"LNA": 20, "TIA": 6, "PGA": 0}
When none of the gain keys are given and AGC is not enabled, ``default_gain_db``
is applied so the front-end is never stuck at 0 dB (the common "hears nothing"
trap). Pass ``default_gain_db=None`` to leave the SDR default untouched.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Protocol

_log = logging.getLogger("gs_flowgraphs._soapy")


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

    # Per-element staging WINS over the overall gain (the documented GS_SDR_GAINS
    # precedence — see sdr_env): SoapySDR's overall setGain re-distributes across
    # the elements, so applying it after the staging would override it (docs/J
    # LOW-3). Overall gain applies only when no per-element gain took effect,
    # matching the dsp RX path's elif chain.
    overall = p.get("sdr_gain_db")
    if not gave_gain and _is_number(overall):
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


def make_source(device_args: str, *, dtype: str = "fc32", nchan: int = 1) -> Any:
    """Build a gr-soapy RX **source** with the installed gr-soapy signature —
    ``source(device, type, nchan, dev_args: str, stream_args: str,
    tune_args: Seq[str], other_settings: Seq[str])``. Centralized here so a gr-soapy
    API change is ONE edit, not one per engine (the per-engine calls had drifted:
    they passed a list for ``stream_args`` and an extra positional, which raised
    TypeError at construction → the flowgraph never reached 'ready'). Lazy gnuradio
    import keeps this module import-safe; the caller still sets rate/freq/gain."""
    from gnuradio import soapy  # noqa: PLC0415 — bench-only; keeps the module GR-free

    return soapy.source(device_args, dtype, nchan, "", "", [""] * nchan, [""] * nchan)


def make_sink(device_args: str, *, dtype: str = "fc32", nchan: int = 1) -> Any:
    """Build a gr-soapy TX **sink** (same signature as :func:`make_source`)."""
    from gnuradio import soapy  # noqa: PLC0415 — bench-only

    return soapy.sink(device_args, dtype, nchan, "", "", [""] * nchan, [""] * nchan)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        _log.warning("ignoring non-numeric %s=%r", name, raw)
        return None


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_gains(name: str) -> dict[str, float] | None:
    """Parse a SatNOGS-style per-element gain string ``"LNA=30,TIA=9,PGA=3"`` into a
    ``{name: dB}`` dict. Returns None when unset/empty; skips malformed entries."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    out: dict[str, float] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, _, val = pair.partition("=")
        key = key.strip()
        try:
            out[key] = float(val.strip())
        except ValueError:
            _log.warning("ignoring bad gain element %r in %s", pair.strip(), name)
    return out or None


def sdr_env() -> dict[str, Any]:
    """Station-wide SDR settings from the environment (SatNOGS-style ``GS_SDR_*``),
    applied by every engine on top of any per-pass params. All optional — unset keys
    leave today's behaviour unchanged.

    * ``GS_SDR_ANTENNA``    — RX antenna name (e.g. ``LNAW``); else auto / per-pass.
    * ``GS_SDR_GAIN_DB``    — overall RF gain in dB; else engine default.
    * ``GS_SDR_GAINS``      — per-element gain staging ``"LNA=30,TIA=9,PGA=3"`` (the
      SatNOGS ``GAIN_MODE="Settings Field"`` / ``OTHER_SETTINGS`` equivalent); wins
      over ``GS_SDR_GAIN_DB`` when both are set.
    * ``GS_SDR_AGC``        — ``1/true`` to enable hardware AGC.
    * ``GS_SDR_LO_OFFSET``  — Hz to shift the LO off the carrier (dodge the DC spike).
    * ``GS_SDR_PPM``        — oscillator frequency-error correction, ppm.
    * ``GS_SDR_DC_REMOVAL`` — ``1/true`` to enable automatic DC-offset correction.
    * ``GS_SDR_CAPTURE_RATE`` — Hz the SDR actually samples at; the engine decimates
      down to the channel ``--sample-rate``. Many real SDRs (XTRX/LMS7 floor ~2.1 Msps)
      can't stream narrow channel rates. Default **2.048 Msps** (SatNOGS's value); set
      ``0`` to disable (capture directly at the channel rate, e.g. RTL-class SDRs).
    """
    return {
        "antenna": os.environ.get("GS_SDR_ANTENNA", "").strip() or None,
        "gain_db": _env_float("GS_SDR_GAIN_DB"),
        "gains": _env_gains("GS_SDR_GAINS"),
        "agc": _env_bool("GS_SDR_AGC"),
        "lo_offset_hz": _env_float("GS_SDR_LO_OFFSET") or 0.0,
        "ppm": _env_float("GS_SDR_PPM") or 0.0,
        "dc_removal": _env_bool("GS_SDR_DC_REMOVAL"),
        "capture_rate_hz": _capture_rate(),
    }


_DEFAULT_CAPTURE_RATE_HZ = 2_048_000.0  # SatNOGS SATNOGS_RX_SAMP_RATE; XTRX RX floor ~2.1 Msps


def _capture_rate() -> float:
    """GS_SDR_CAPTURE_RATE: unset → 2.048 Msps default; explicit 0 → disabled."""
    v = _env_float("GS_SDR_CAPTURE_RATE")
    return _DEFAULT_CAPTURE_RATE_HZ if v is None else v


def capture_plan(env_capture_rate_hz: float, channel_rate_hz: float) -> tuple[float, bool]:
    """Return ``(sdr_rate_hz, decimate)``: the rate to set on the SDR and whether the
    engine must decimate to ``channel_rate_hz``. Decimation is used only when a capture
    rate is configured (>0) and differs from the channel rate."""
    if env_capture_rate_hz and abs(env_capture_rate_hz - channel_rate_hz) > 1.0:
        return float(env_capture_rate_hz), True
    return float(channel_rate_hz), False


def auto_lo_offset(
    sdr_rate_hz: float, channel_rate_hz: float, configured_offset_hz: float
) -> float:
    """Resolve the LO offset, honoring ``GS_SDR_LO_OFFSET`` LITERALLY: unset/0 ⇒ tune
    **ON-CENTER** (no offset) and let ``GS_SDR_DC_REMOVAL`` notch the DC/LO spike — the
    standard narrowband approach, and now the default.

    We no longer AUTO-force an offset. The spike-dodge only works if the −offset back-shift
    is done where we control it (a software rotator, the SatNOGS way); ``tune_source`` does
    it via the driver's hardware **BB CORDIC**, which the **XTRX silently no-ops** — so a
    forced offset shoved the signal off-band. An explicit offset is still honored (for a
    driver that does the RF/BB split), but clamped to what the captured band can hold, else
    it too would push the signal out (→ 0 = on-center)."""
    if not configured_offset_hz:
        return 0.0
    off = float(configured_offset_hz)
    room = float(sdr_rate_hz) / 2.0 - float(channel_rate_hz) / 2.0
    return off if off <= room else 0.0


def resample_ratio(capture_rate_hz: float, channel_rate_hz: float) -> tuple[int, int]:
    """(interpolation, decimation) to convert ``capture_rate`` → ``channel_rate``,
    exact for integer rates (2.048 Msps → 48 ksps = 3/128)."""
    from fractions import Fraction  # noqa: PLC0415

    frac = Fraction(int(round(channel_rate_hz)), int(round(capture_rate_hz)))
    return frac.numerator, frac.denominator


def make_decimator(capture_rate_hz: float, channel_rate_hz: float) -> Any:
    """An anti-aliased rational resampler converting an SDR capture rate down to the
    channel/processing rate — the SatNOGS "capture wide, decimate in software" model."""
    from gnuradio import filter as gr_filter  # noqa: PLC0415 — bench-only

    interp, decim = resample_ratio(capture_rate_hz, channel_rate_hz)
    return gr_filter.rational_resampler_ccf(interpolation=interp, decimation=decim)


def merge_sdr_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Per-pass ``params`` with station ``GS_SDR_*`` antenna/gain/agc filled in as
    defaults (per-pass values win). Feed the result to :func:`configure_soapy_source`."""
    env = sdr_env()
    merged: dict[str, Any] = dict(params or {})
    if env["antenna"] and "sdr_antenna" not in merged:
        merged["sdr_antenna"] = env["antenna"]
    if env["gains"] and "sdr_gains" not in merged:
        merged["sdr_gains"] = env["gains"]
    if env["gain_db"] is not None and "sdr_gain_db" not in merged:
        merged["sdr_gain_db"] = env["gain_db"]
    if "sdr_agc" not in merged:
        merged["sdr_agc"] = env["agc"]
    return merged


def tune_source(src: Any, center_hz: float, lo_offset_hz: float, *, channel: int = 0) -> None:
    """Tune a gr-soapy source/sink to ``center_hz`` using an LO offset: the analog LO
    goes to ``center+offset`` (RF) and the baseband CORDIC to ``-offset`` (BB), so the
    SDR's DC spike lands at ``+offset`` instead of on the signal. ``lo_offset_hz`` 0 →
    a plain tune. Falls back to a direct tune if the driver lacks named RF/BB
    components (then there is simply no offset benefit)."""
    if lo_offset_hz:
        try:
            src.set_frequency(channel, "RF", float(center_hz) + float(lo_offset_hz))
            src.set_frequency(channel, "BB", -float(lo_offset_hz))
            return
        except Exception:  # noqa: BLE001 — driver without RF/BB split → plain tune
            _log.warning("LO offset unsupported by driver; tuning RF directly")
    src.set_frequency(channel, float(center_hz))


def retune_source(
    src: Any,
    center_hz: float,
    lo_offset_hz: float,
    doppler_hz: float,
    *,
    channel: int = 0,
) -> None:
    """Doppler retune that preserves the LO offset by moving only the RF component."""
    if lo_offset_hz:
        try:
            src.set_frequency(
                channel, "RF", float(center_hz) + float(lo_offset_hz) + float(doppler_hz)
            )
            return
        except Exception:  # noqa: BLE001 — driver without RF/BB split → plain retune
            pass
    src.set_frequency(channel, float(center_hz) + float(doppler_hz))


def apply_corrections(
    src: Any, *, ppm: float = 0.0, dc_removal: bool = False, channel: int = 0
) -> None:
    """Best-effort ppm + DC-removal on a gr-soapy endpoint (drivers vary; never raises)."""
    if ppm:
        with contextlib.suppress(Exception):
            src.set_frequency_correction(channel, float(ppm))
    if dc_removal:
        with contextlib.suppress(Exception):
            src.set_dc_offset_mode(channel, True)


__all__ = [
    "apply_corrections",
    "capture_plan",
    "configure_soapy_source",
    "make_decimator",
    "make_sink",
    "make_source",
    "merge_sdr_params",
    "resample_ratio",
    "retune_source",
    "sdr_env",
    "tune_source",
]

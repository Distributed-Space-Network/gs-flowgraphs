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


DEFAULT_LO_OFFSET_HZ = 100_000.0  # SatNOGS lo-offset default; the software rotator dodges the spike


def auto_lo_offset(
    sdr_rate_hz: float, channel_rate_hz: float, configured_offset_hz: float,
    *, default_offset_hz: float = 0.0,
) -> float:
    """Resolve the LO offset. ``GS_SDR_LO_OFFSET`` (``configured_offset_hz``) wins when set;
    otherwise ``default_offset_hz`` is used. Returns 0 (ON-CENTER) when there is no wideband
    headroom OR when the offset can't fit the captured band (``offset + channel/2 > sdr/2``).

    ``default_offset_hz`` is the key knob: the software-rotator RX engines
    (satellite/gfsk) pass :data:`DEFAULT_LO_OFFSET_HZ` (100 kHz) so the carrier is captured
    off-center and a downstream rotator (:func:`tune_below` + :func:`make_lo_rotator`) shifts it to
    DC while the decimator's LPF rejects the DC/LO spike — this works on ANY driver because the
    shift is in software, unlike the driver BB CORDIC the **XTRX silently no-ops**. Callers that
    still drive the hardware RF/BB split (:func:`tune_source`, e.g. the amateur-FM engine) OMIT it,
    so an unset offset stays ON-CENTER (0) and ``GS_SDR_DC_REMOVAL`` notches the spike — giving
    them a forced offset would push the signal off-band on the XTRX.

    An offset that OVERFLOWS the band (``off > room``) falls back to ON-CENTER, NOT the band edge:
    the 100 kHz default is well below ``room`` for any realistic capture/channel, so only a mis-set
    explicit ``GS_SDR_LO_OFFSET`` (or an unphysical near-full-band channel) reaches here, and a
    hardware-split (FM) caller can't dodge an off-band offset at all — on-center is the only safe
    result. (Restores the pre-rotator semantics; a ``min(off, room)`` clamp regressed FM-on-XTRX to
    off-band tuning.)"""
    room = float(sdr_rate_hz) / 2.0 - float(channel_rate_hz) / 2.0
    if room <= 0.0:
        return 0.0
    off = float(configured_offset_hz) if configured_offset_hz else float(default_offset_hz)
    if off <= 0.0 or off > room:
        return 0.0  # on-center: unset, or the offset can't fit the captured band
    return off


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


def lo_phase_inc(sdr_rate_hz: float, lo_offset_hz: float, doppler_hz: float = 0.0) -> float:
    """Rotator phase increment (rad/sample) that shifts a carrier sitting at ``+lo_offset``
    (``+doppler``) at baseband DOWN to DC: ``-2π(lo_offset + doppler)/sdr_rate``. Pure — so the
    LO/Doppler math is unit-testable without GNU Radio. Feed to ``blocks.rotator_cc`` /
    ``rotator.set_phase_inc`` (see :func:`make_lo_rotator` and each engine's ``set_doppler``)."""
    import math  # noqa: PLC0415

    return -2.0 * math.pi * (float(lo_offset_hz) + float(doppler_hz)) / float(sdr_rate_hz)


def make_lo_rotator(sdr_rate_hz: float, lo_offset_hz: float, doppler_hz: float = 0.0) -> Any:
    """A ``blocks.rotator_cc`` (at the capture rate) that brings the ``+lo_offset`` (``+doppler``)
    carrier to DC — the software replacement for BOTH the XTRX-broken hardware BB offset AND
    hardware Doppler retuning. Update Doppler mid-pass with
    ``rotator.set_phase_inc(lo_phase_inc(sdr_rate, lo_offset, doppler))`` — a pure NCO retune with
    no PLL settle glitch, mirroring SatNOGS' ``doppler_compensation`` rotator. Sits right after the
    source and before :func:`make_decimator`, so the decimator's LPF rejects the spike (now at
    ``-lo_offset``, far outside the channel)."""
    from gnuradio import blocks  # noqa: PLC0415 — bench-only

    return blocks.rotator_cc(lo_phase_inc(sdr_rate_hz, lo_offset_hz, doppler_hz))


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


def tune_below(src: Any, center_hz: float, lo_offset_hz: float, *, channel: int = 0) -> None:
    """Tune a gr-soapy source PLAINLY to ``center - lo_offset`` so the carrier lands at
    ``+lo_offset`` at baseband, off the DC/LO spike — the SatNOGS "capture off-center, correct in
    software" model. A downstream software rotator (:func:`make_lo_rotator`) shifts the carrier to
    DC and the decimator's LPF rejects the spike left at ``-lo_offset``. Unlike :func:`tune_source`
    this uses NO driver RF/BB CORDIC split (the XTRX no-ops the BB half, throwing a hardware offset
    off-band), so the offset actually takes effect. ``lo_offset`` 0 → a plain on-center tune. This
    is the tuning half of the Phase-1 LO-rotator path (docs/12); ``tune_source`` remains for the
    amateur-FM engine, which still drives the driver RF/BB split directly."""
    src.set_frequency(channel, float(center_hz) - float(lo_offset_hz))


def open_analog_bandwidth(src: Any, sdr_rate_hz: float, *, channel: int = 0) -> None:
    """Widen the SDR ANALOG RX filter to ~the capture rate so an LO-offset carrier is NOT rolled
    off before the ADC. :func:`tune_below` parks the carrier at ``+lo_offset`` at baseband (e.g.
    +500 kHz for ``GS_SDR_LO_OFFSET=500000``); the XTRX analog filter floor is ~0.8 MHz, so at a
    large offset that carrier sits past a narrow default passband edge and is attenuated toward the
    ADC floor — a silent capture on a bird that WAS transmitting. Channel selectivity is done
    DOWNSTREAM in DSP (the decimator + channel filter), so the analog filter must pass the WHOLE
    capture band. The satellite/gfsk RX engines MUST call this (the amateur-FM engine already does);
    an on-center tune (lo_offset 0) doesn't strictly need it but a wide analog BW never hurts.
    Guarded — a driver without a settable analog BW (RTL-class) just ignores it."""
    with contextlib.suppress(Exception):  # noqa: BLE001 — driver may lack a settable analog BW
        src.set_bandwidth(int(channel), float(sdr_rate_hz))


def tune_source(src: Any, center_hz: float, lo_offset_hz: float, *, channel: int = 0) -> None:
    """Tune a gr-soapy source/sink to ``center_hz`` using an LO offset: the analog LO
    goes to ``center+offset`` (RF) and the baseband CORDIC to ``-offset`` (BB), so the
    SDR's DC spike lands at ``+offset`` instead of on the signal. ``lo_offset_hz`` 0 →
    a plain tune. Falls back to a direct tune if the driver lacks named RF/BB
    components (then there is simply no offset benefit).

    NOTE: the driver BB CORDIC is a no-op on the XTRX, so this offset does NOT take effect there —
    the satellite/gfsk RX path uses :func:`tune_below` + :func:`make_lo_rotator` instead (docs/12
    Phase 1). This remains for the amateur-FM engine (wideband, less spike-sensitive)."""
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
) -> dict[str, Any]:
    """Best-effort ppm + DC-removal on a gr-soapy endpoint (drivers vary; never
    raises). R-21: returns a report of what actually applied — a correction the
    driver refused is WARNED and recorded (``ppm_error``/``dc_removal_error``),
    never silently dropped: an unapplied ppm on a warm XTRX is hundreds of Hz of
    unexplained carrier offset."""
    report: dict[str, Any] = {}
    if ppm:
        try:
            src.set_frequency_correction(channel, float(ppm))
            report["ppm"] = float(ppm)
        except Exception as e:  # noqa: BLE001 — driver-dependent; report, don't raise
            report["ppm_error"] = repr(e)
            _log.warning("ppm correction %.2f NOT applied: %s", ppm, e)
    if dc_removal:
        try:
            src.set_dc_offset_mode(channel, True)
            report["dc_removal"] = True
        except Exception as e:  # noqa: BLE001 — driver-dependent; report, don't raise
            report["dc_removal_error"] = repr(e)
            _log.warning("DC-offset removal NOT applied: %s", e)
    return report


# R-21 readback: key → candidate getter names. snake_case = gr-soapy block
# surface, takes (channel); camelCase = native ``SoapySDR.Device``, takes
# (direction, channel) — callers with a native device pass ``direction``.
_READBACK_GETTERS: dict[str, tuple[str, ...]] = {
    "antenna": ("get_antenna", "getAntenna"),
    "gain_db": ("get_gain", "getGain"),
    "agc": ("get_gain_mode", "getGainMode"),
    "sample_rate_hz": ("get_sample_rate", "getSampleRate"),
    "bandwidth_hz": ("get_bandwidth", "getBandwidth"),
    "frequency_hz": ("get_frequency", "getFrequency"),
    "ppm": ("get_frequency_correction", "getFrequencyCorrection"),
    "dc_removal": ("get_dc_offset_mode", "getDCOffsetMode"),
}


def readback_soapy_settings(
    endpoint: Any, *, channel: int = 0, direction: object = None
) -> dict[str, Any]:
    """R-21: the ACTUAL front-end state read back from the device — what the
    hardware settled on, not what we asked for. Works on both surfaces: a
    gr-soapy block (snake_case getters, per-channel) and a native
    ``SoapySDR.Device`` (camelCase getters, pass ``direction``). Each key is
    individually guarded; keys the driver/binding cannot read back are listed
    under ``"unreadable"`` so the report says so explicitly instead of looking
    complete."""
    actual: dict[str, Any] = {}
    unreadable: list[str] = []
    for key, names in _READBACK_GETTERS.items():
        got = False
        for name in names:
            fn = getattr(endpoint, name, None)
            if fn is None:
                continue
            native = name[3:4].isupper()  # getAntenna vs get_antenna
            if native and direction is None:
                continue  # native getters need the direction constant
            try:
                value = fn(direction, channel) if native else fn(channel)
            except Exception:  # noqa: BLE001 — getter exists but driver refuses
                continue
            actual[key] = value
            got = True
            break
        if not got:
            unreadable.append(key)
    if unreadable:
        actual["unreadable"] = unreadable
    return actual


def sdr_ready_fields(
    *,
    device: str,
    requested: dict[str, Any] | None,
    applied: dict[str, Any] | None,
    actual: dict[str, Any] | None,
    stream_active: bool,
    first_samples: bool | None,
) -> dict[str, Any]:
    """R-11/R-21: the ``ready`` event block proving the front-end works —
    device identity, requested vs applied vs read-back settings, an ACTIVE
    stream, and first-sample proof. ``first_samples`` is ``None`` when no
    probe is available (recorded as such — absence of proof is stated, not
    implied)."""
    return {
        "sdr": {
            "device": device,
            "requested": dict(requested or {}),
            "applied": dict(applied or {}),
            "actual": dict(actual or {}),
        },
        "stream_active": bool(stream_active),
        "first_samples": first_samples,
    }


__all__ = [
    "DEFAULT_LO_OFFSET_HZ",
    "apply_corrections",
    "auto_lo_offset",
    "capture_plan",
    "configure_soapy_source",
    "lo_phase_inc",
    "make_decimator",
    "make_lo_rotator",
    "make_sink",
    "make_source",
    "merge_sdr_params",
    "open_analog_bandwidth",
    "readback_soapy_settings",
    "resample_ratio",
    "retune_source",
    "sdr_env",
    "sdr_ready_fields",
    "tune_below",
    "tune_source",
]

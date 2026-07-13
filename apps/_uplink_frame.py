"""The ONE payload + framing selector. Both TX engines go through it (R2-43).

THE BUG THIS EXISTS TO KILL. The two engines each resolved the uplink payload and built the frame
themselves, and they did it differently:

    dsp        payload from ``uplink_b64``, or ``uplink_file``, or an on-disk ``uplink.bin``;
               framing ``ax25`` OR ``endurosat`` (the chip packet), per ``params["framing"]``.
    gnuradio   payload from ``uplink_b64`` ONLY — the other two sources silently produced an EMPTY
               payload — and ALWAYS built AX.25, ignoring ``framing`` entirely.

The waveform schema advertises the ``gnuradio`` + ``endurosat`` pair. Fly that pair and the station
keys the PA and transmits an **AX.25 UI frame at a satellite that speaks EnduroSat chip packets** —
a well-formed, correctly-modulated, completely wrong protocol, radiated at a real spacecraft. No
error, no warning; the burst reports success. Ask for a file-sourced uplink on that engine and you
transmit an EMPTY frame instead.

THE FIX. Framing is decided ONCE, here, and produces BITS. An engine's only remaining job is to turn
those bits into IQ. There is no longer anywhere for the two paths to disagree, because there is no
longer a second place that knows what a frame is.

:func:`supported_pairs` exposes the capability table so config validation can reject an
(engine, framing) tuple nobody implemented, rather than discovering it at the PA.
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _fallback_select import symbol_rate_hz_of

from gfsk_ax25 import ax25, endurosat, endurosat_link, framing

log = logging.getLogger(__name__)

ENGINES = ("dsp", "gnuradio")

# ROUND 9: hard ceilings on what a single burst may ask the machine for. gfsk.modulate() expands
# every bit into `sps` samples via np.repeat; with nothing bounding sps or the bit count, a bad
# sample-rate/symbol-rate pair asks numpy for hundreds of gigabytes and the station dies mid-pass
# with the PA keyed.
_MAX_SPS = 1024                     # a 9600-baud link at 100 Msps is ~10k — far past anything real
_MAX_IQ_SAMPLES = 200_000_000       # 200 M complex64 = 1.6 GB, ~100 s at 2 Msps. Nothing legitimate
                                    # comes close; a single EnduroSat burst is ~30k samples.
FRAMINGS = ("ax25", "endurosat")


def supported_pairs() -> tuple[tuple[str, str], ...]:
    """Every (engine, framing) tuple that is actually implemented, end to end.

    Both engines now share this module's frame, so every combination is real. That is the point of
    publishing the table: before R2-43, ``("gnuradio", "endurosat")`` was ADVERTISED and silently
    transmitted AX.25. A table nothing consults is a table that can lie again, so
    ``gs_client.config.waveforms`` validates against it."""
    return tuple((e, f) for e in ENGINES for f in FRAMINGS)


@dataclass(frozen=True)
class UplinkFrame:
    """A framed uplink, as BITS, plus the modulation the frame must be sent with.

    Both engines receive this and do nothing but modulate it. The symbol rate, modulation index
    and BT travel WITH the bits: they are properties of the link the framing chose — the EnduroSat
    chip packet is 9600 sym/s h=0.5 BT=0.5 by default, and AX.25 takes them from the LinkProfile.
    An engine that re-derived them from its own defaults was the other half of this bug."""

    bits: np.ndarray  # 0/1, MSB-first
    sample_rate_hz: float
    symbol_rate_hz: float
    mod_index: float
    bt: float
    framing: str
    payload_len: int
    payload_source: str  # for the log line: which of the three sources actually supplied bytes

    @property
    def sps(self) -> float:
        return self.sample_rate_hz / self.symbol_rate_hz


def resolve_payload(args, params: dict[str, object]) -> tuple[bytes, str]:
    """The uplink bytes, and WHERE they came from.

    Three sources, in priority order. The gnuradio engine used to know about exactly one of them and
    quietly transmit an empty frame for the other two, which is why the source is returned and
    logged rather than left implicit."""
    b64 = params.get("uplink_b64")
    if isinstance(b64, str) and b64:
        return base64.b64decode(b64), "uplink_b64"
    named = params.get("uplink_file")
    if named and Path(str(named)).exists():
        return Path(str(named)).read_bytes(), f"uplink_file={named}"
    on_disk = Path(getattr(args, "output_dir", None) or ".") / "uplink.bin"
    if on_disk.exists():
        return on_disk.read_bytes(), str(on_disk)
    return b"", "NONE"


# The labels the BACKEND and the SatNOGS/transmitter catalogues actually emit, mapped onto the two
# framings we implement. R2-43 (round 5): selection matched the exact strings "ax25"/"endurosat" and
# SILENTLY fell back to AX.25 for everything else — so a pass whose framing said "AirMAC" (the
# customer's own label for the EnduroSat session layer), or "EnduroSat AirMAC", or anything
# mis-cased, transmitted an AX.25 frame at an EnduroSat bird and reported success. A fallback is
# fine for an ABSENT framing. For a framing that was explicitly REQUESTED and is not understood, a
# fallback is a wrong-protocol transmission dressed up as a default.
_FRAMING_ALIASES: dict[str, str] = {
    "ax25": "ax25",
    "ax.25": "ax25",
    "ax_25": "ax25",
    "endurosat": "endurosat",
    "endurosat_link": "endurosat",
    "endurosat-link": "endurosat",
    "airmac": "endurosat",  # AirMAC rides INSIDE the EnduroSat chip packet, opaquely
    "endurosat airmac": "endurosat",
    "endurosat_airmac": "endurosat",
    "endurosat-airmac": "endurosat",
}


class UnknownFraming(ValueError):
    """An explicitly-requested framing we do not implement. NEVER silently downgraded."""


class UnknownEngine(ValueError):
    """An explicitly-requested engine we do not implement. NEVER silently downgraded to dsp."""


class PayloadRejected(ValueError):
    """The payload cannot be carried by the chosen framing. We refuse rather than truncate."""


class RateUnusable(ValueError):
    """The sample rate is not an integer multiple of the framing's symbol rate."""


class ModulationUnusable(ValueError):
    """A modulation parameter is non-finite or out of range (bt=0, NaN, Inf...)."""


def preflight(
    args, params: dict[str, object], profile, *, engine: str, framing_name: str
) -> UplinkFrame:
    """Build and CHECK the entire transmission before the app says ``ready`` — and long before the
    orchestrator arms a PA.

    Round 7: every one of these was discovered mid-pass, AFTER `ready`, AFTER the T/R relay had
    been thrown and the PA keyed:

      * the sample rate was not an integer multiple of the symbol rate, so the modulator raised
        ``sample_rate/symbol_rate must be integer``. The canonical AX.25 TX waveform shipped at
        96 kHz against 12480 sym/s (7.69 samples/symbol) — it could never have transmitted.
      * an unknown engine silently became ``dsp``; an unknown framing silently became AX.25.
      * an oversized payload was silently truncated and radiated as a successful command.

    A flowgraph that cannot transmit must fail BEFORE anything is energized."""
    if engine not in ENGINES:
        msg = f"unknown engine {engine!r} — refusing to fall back to dsp. Known: {sorted(ENGINES)}"
        raise UnknownEngine(msg)

    frame = build_uplink_frame(args, params, profile, framing_name=framing_name)

    # ROUND 8: every modulation number must be FINITE and BOUNDED.
    #
    # bt = 0 raised ZeroDivisionError inside the Gaussian filter; bt/mod_index of NaN or Inf sailed
    # through and produced non-finite IQ that the SDR would happily accept. All of it AFTER `ready`,
    # with the T/R relay thrown and the PA keyed. A number that cannot be modulated must stop the
    # pass on the ground.
    for name, value, lo, hi in (
        # ROUND 9: protocol-specific floors. A 10-baud symbol rate is not a slow link, it is a
        # nonsense one — the REST backend accepted baud=10 into the live bidirectional waveform.
        ("symbol_rate_hz", frame.symbol_rate_hz, 1200.0, 10e6),
        ("sample_rate_hz", frame.sample_rate_hz, 1200.0, 100e6),
        ("mod_index", frame.mod_index, 1e-3, 10.0),
        ("bt", frame.bt, 1e-3, 10.0),
    ):
        if not math.isfinite(value):
            msg = f"{name} is {value!r} — not finite. Refusing to key a PA on a number like that."
            raise ModulationUnusable(msg)
        if not (lo <= value <= hi):
            msg = (
                f"{name} = {value!r} is outside the usable range [{lo}, {hi}]. Refusing to key the "
                f"PA — bt=0 divides by zero inside the Gaussian filter, and out-of-range values "
                f"produce a waveform no receiver can demodulate."
            )
            raise ModulationUnusable(msg)
    sps = frame.sps
    if abs(sps - round(sps)) > 1e-9 or round(sps) < 2:
        msg = (
            f"sample_rate {frame.sample_rate_hz:.0f} Hz is not a usable integer multiple of the "
            f"{frame.framing} symbol rate {frame.symbol_rate_hz:.0f} sym/s "
            f"({sps:.4f} samples/symbol). BOTH engines require an integer sps, so this pass could "
            f"never transmit. Choose a rate valid for every advertised framing (124800 Hz gives 10 "
            f"sps at 12480 and 13 sps at 9600)."
        )
        raise RateUnusable(msg)
    # ROUND 9: BOUND THE ALLOCATION.
    #
    # gfsk.modulate() does np.repeat(symbols, sps) — so samples-per-symbol and the bit count
    # together
    # decide how much memory this asks for. Nothing bounded either. A large sample_rate against a
    # small symbol rate gives an enormous sps, and np.repeat cheerfully attempts to allocate
    # hundreds
    # of gigabytes: the station dies, mid-pass, with the PA keyed.
    sps_i = round(frame.sps)
    if sps_i > _MAX_SPS:
        msg = (
            f"samples-per-symbol is {sps_i} (sample_rate {frame.sample_rate_hz:.0f} / symbol_rate "
            f"{frame.symbol_rate_hz:.0f}); the ceiling is {_MAX_SPS}. Refusing: np.repeat "
            f"would allocate an absurd buffer and kill the station mid-pass."
        )
        raise ModulationUnusable(msg)
    total_samples = int(frame.bits.size) * sps_i
    if total_samples > _MAX_IQ_SAMPLES:
        msg = (
            f"this burst would be {total_samples:,} IQ samples "
            f"({total_samples * 8 / 1e6:.0f} MB of complex64) and last "
            f"{total_samples / frame.sample_rate_hz:.1f} s. The ceiling is {_MAX_IQ_SAMPLES:,} "
            f"samples. Refusing to key a PA for a transmission that size."
        )
        raise ModulationUnusable(msg)

    log.info(
        "TX preflight OK: engine=%s framing=%s payload=%dB from %s | %d bits @ %.0f sym/s, %d sps",
        engine, frame.framing, frame.payload_len, frame.payload_source,
        frame.bits.size, frame.symbol_rate_hz, round(sps),
    )
    return frame


def normalize_framing(label: str) -> str:
    """A backend/catalogue framing label -> one of FRAMINGS. Raises on an unknown EXPLICIT label."""
    key = " ".join(str(label).strip().lower().split())
    if key in _FRAMING_ALIASES:
        return _FRAMING_ALIASES[key]
    msg = (
        f"unknown framing {label!r} — refusing to fall back to AX.25. Transmitting the wrong link "
        f"layer at a spacecraft is worse than transmitting nothing. Known: "
        f"{sorted(set(_FRAMING_ALIASES))}"
    )
    raise UnknownFraming(msg)


def select_framing(params: dict[str, object], *, env: str = "") -> str:
    """The framing for this pass. ABSENT -> ax25 (the app's documented default). PRESENT but unknown
    -> raise, because the caller asked for something specific and we cannot do it.

    Round 7: PARAMS WIN OVER THE ENVIRONMENT. It was the other way round, so the env var
    silently overrode a framing gs-client had VALIDATED against its capability table and written
    into params.json — turning a configured EnduroSat uplink back into AX.25. An env var is a
    developer convenience; it does not get to overrule the station's reviewed configuration."""
    from_params = str(params.get("framing", "")) if isinstance(params, dict) else ""
    chosen = from_params or env
    if not chosen.strip():
        return "ax25"
    if from_params and env and normalize_framing(env) != normalize_framing(from_params):
        log.warning(
            "GS_FLOWGRAPH_FRAMING=%r IGNORED: params.json says %r, and the station's validated "
            "configuration wins.",
            env, from_params,
        )
    return normalize_framing(chosen)


def build_uplink_frame(
    args, params: dict[str, object], profile: endurosat.LinkProfile, *, framing_name: str
) -> UplinkFrame:
    """Resolve the payload, apply the chosen framing, and return the BITS to modulate.

    This is the single point at which "what do we transmit" is decided. Both engines call it."""
    payload, source = resolve_payload(args, params)
    sample_rate = float(getattr(args, "sample_rate", 0) or 96_000)

    if framing_name == "endurosat":
        # The payload is the already-built (encrypted AirMAC) frame; it rides VERBATIM inside the
        # EnduroSat chip packet (preamble + sync + len + CRC-16). Nothing about AX.25 applies —
        # no HDLC, no NRZI, no G3RUH scrambling — which is precisely what the gnuradio path used to
        # wrap it in.
        if len(payload) > endurosat_link.MAX_PAYLOAD:
            msg = (
                f"uplink payload is {len(payload)} B; the EnduroSat chip packet carries at most "
                f"{endurosat_link.MAX_PAYLOAD} B. REFUSING to transmit. This used to be silently "
                f"TRUNCATED and radiated as a successful command — a truncated command is a "
                f"DIFFERENT command, and the spacecraft would have executed whatever the first "
                f"{endurosat_link.MAX_PAYLOAD} bytes happened to mean."
            )
            raise PayloadRejected(msg)
        if not payload:
            msg = (
                "uplink payload is EMPTY and the framing is EnduroSat. An empty chip "
                "packet does not deframe at the far end, so this is a PA key with no "
                "possible effect. REFUSING."
            )
            raise PayloadRejected(msg)
        clipped = payload
        return UplinkFrame(
            bits=endurosat_link.frame_bits(clipped),
            sample_rate_hz=sample_rate,
            symbol_rate_hz=symbol_rate_hz_of(params, default=endurosat_link.DEFAULT_SYMBOL_RATE_HZ),
            mod_index=float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
            bt=float(params.get("bt", endurosat_link.DEFAULT_BT)),
            framing="endurosat",
            payload_len=len(clipped),
            payload_source=source,
        )

    if len(payload) > endurosat.AX25_INFO_MAX_BYTES:
        msg = (
            f"uplink payload is {len(payload)} B; an AX.25 UI info field carries at most "
            f"{endurosat.AX25_INFO_MAX_BYTES} B. REFUSING to transmit rather than silently "
            f"TRUNCATING and radiating a different command than the one that was booked."
        )
        raise PayloadRejected(msg)
    clipped = payload
    body = ax25.encode_ui(
        dest=str(params.get("dest", "CQ")),
        src=str(params.get("src", "DSN")),
        info=clipped,
    )
    return UplinkFrame(
        bits=framing.encode(body, scramble=profile.scramble, nrzi=profile.nrzi),
        sample_rate_hz=sample_rate,
        symbol_rate_hz=profile.symbol_rate_hz,
        mod_index=profile.mod_index,
        bt=profile.bt,
        framing="ax25",
        payload_len=len(clipped),
        payload_source=source,
    )


__all__ = [
    "ENGINES",
    "ModulationUnusable",
    "FRAMINGS",
    "UplinkFrame",
    "build_uplink_frame",
    "resolve_payload",
    "select_framing",
    "supported_pairs",
]

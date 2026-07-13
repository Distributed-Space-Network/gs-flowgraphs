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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _fallback_select import symbol_rate_hz_of

from gfsk_ax25 import ax25, endurosat, endurosat_link, framing

log = logging.getLogger(__name__)

ENGINES = ("dsp", "gnuradio")
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
    """The framing for this pass. ABSENT -> ax25 (the app's documented default). PRESENT but
    unknown -> raise, because the caller asked for something specific and we cannot do it."""
    chosen = env or (str(params.get("framing", "")) if isinstance(params, dict) else "")
    if not chosen.strip():
        return "ax25"
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
        clipped = payload[: endurosat_link.MAX_PAYLOAD]
        if len(payload) > endurosat_link.MAX_PAYLOAD:
            log.warning(
                "uplink payload %d B exceeds the EnduroSat chip-packet maximum %d B — TRUNCATED",
                len(payload), endurosat_link.MAX_PAYLOAD,
            )
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

    clipped = payload[: endurosat.AX25_INFO_MAX_BYTES]
    if len(payload) > endurosat.AX25_INFO_MAX_BYTES:
        log.warning(
            "uplink payload %d B exceeds the AX.25 info maximum %d B — TRUNCATED",
            len(payload), endurosat.AX25_INFO_MAX_BYTES,
        )
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
    "FRAMINGS",
    "UplinkFrame",
    "build_uplink_frame",
    "resolve_payload",
    "select_framing",
    "supported_pairs",
]

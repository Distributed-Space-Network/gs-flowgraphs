"""Framing registry — framing name → deframe (docs/08 Tier 1).

Two classes of framing, so the registry is the single source of "what we can deframe":

  * **Local (numpy, here):** ``ax25`` (G3RUH-scrambled + plain), ``endurosat``/AirMAC, and
    ``argos`` (PTT/PMT-A3 — a framing gr-satellites does NOT have; see :mod:`gfsk_ax25.argos`).
    Each is CRC/FCS/BCH-gated, so trying several is safe.
  * **gr-satellites (upstream):** the ~50 gr-satellites deframers (AX.100, USP, Mobitex, GEOSCAN,
    AO-40 FEC, CCSDS Concatenated/RS, NGHam, U482C, …). These are NOT re-implemented here — they
    are reused whole by handing gr-satellites a synthetic SatYAML (see
    ``gnuradio_satellites``/``grsat_synth``). We ADVERTISE them via :func:`grsatellites_framings`
    so the composer/backend knows the full deframable set; :func:`deframe` itself only runs the
    local numpy deframers (the gr-satellites path decodes them in the GNU Radio flowgraph).

numpy-only (no GNU Radio) so the local deframe path stays fully unit-testable.
"""
from __future__ import annotations

import numpy as np

# Link layers our own engine deframes in-process (numpy). ``argos`` and ``ccsds_tm`` run ONLY when
# explicitly requested (Argos' sync is a placeholder pending bench confirmation; CCSDS TM needs
# per-bird channel-coding params) — kept out of autodetect to avoid spurious/mis-parametrized runs.
_LOCAL = ("ax25", "endurosat", "argos", "ccsds_tm")
_LOCAL_AUTODETECT = ("ax25", "endurosat")

# The gr-satellites framing vocabulary (SatYAML ``framing:`` strings) reused via synthetic
# SatYAML — advertised here, decoded in the gr-satellites flowgraph. Representative of the ~50
# deframers (docs/08 §2); the backend surfaces these verbatim (scraped from gr-satellites).
GRSATELLITES_FRAMINGS = (
    "AX.25", "AX.25 G3RUH", "AX100 Mode 5", "AX100 Mode 6", "AX100 ASM+Golay",
    "USP", "Mobitex", "Mobitex-NX", "GEOSCAN", "AO-40 FEC", "AO-40 FEC short",
    "AO-40 uncoded", "CCSDS Concatenated", "CCSDS Reed-Solomon", "NGHam", "NGHam no Reed Solomon",
    "U482C", "SNET", "OpenLST", "Reaktor Hello World", "SMOG-P RA", "SMOG-P Signalling",
    "FX.25 NRZI", "TT-64", "SanoSat", "Grizu-263A", "AALTO-1",
)


def local_framings() -> tuple[str, ...]:
    """Framings deframed in-process by :func:`deframe` (numpy)."""
    return _LOCAL


def grsatellites_framings() -> tuple[str, ...]:
    """Framings deframed by reusing gr-satellites (via synthetic SatYAML) — advertised, not
    re-implemented here."""
    return GRSATELLITES_FRAMINGS


def known_framings() -> tuple[str, ...]:
    """Every framing the registry can deframe — local (numpy) + gr-satellites (upstream)."""
    return _LOCAL + GRSATELLITES_FRAMINGS


def deframe(bits, framing_name: str | None = None) -> tuple[list[bytes], str | None]:
    """Hard bits → ``(frames, matched_framing)`` via the LOCAL numpy deframers. ``framing_name``
    runs ONLY that link layer (the backend told us); ``None`` auto-detects across the FCS-gated
    local set (``ax25``/``endurosat``) and reports which matched, so the caller can lock onto it.

    A gr-satellites framing name (e.g. ``"USP"``) is not deframed here — it returns
    ``([], None)`` because that link layer is decoded upstream in the gr-satellites flowgraph
    (the synthetic-SatYAML path); the two run in parallel and the first valid frame wins."""
    from gfsk_ax25 import argos, ccsds, endurosat_link  # noqa: PLC0415
    from gfsk_ax25 import framing as ax25_framing  # noqa: PLC0415

    arr = np.asarray(bits, dtype=np.uint8)
    if not len(arr):
        return [], None
    order = [framing_name.strip().lower()] if framing_name else list(_LOCAL_AUTODETECT)
    for name in order:
        if name == "endurosat":
            frames = endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)
        elif name == "ax25":  # G3RUH-descrambled and plain — same framing, different descrambling
            frames = []
            for scramble in (True, False):
                frames.extend(ax25_framing.decode(arr, scramble=scramble, nrzi=True))
        elif name == "argos":  # explicit only — BCH-gated PTT/PMT-A3 (placeholder sync)
            frames = argos.deframe(arr)
        elif name in ("ccsds_tm", "ccsds"):  # explicit only — ASM+RS(255,223)+randomize+FECF chain
            frames = ccsds.deframe_tm(arr)
        else:
            continue  # gr-satellites framing (or unknown) → decoded upstream, not here
        if frames:
            return frames, name
    return [], None

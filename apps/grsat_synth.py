"""Synthetic gr-satellites SatYAML for NON-catalogued birds (docs/08 Phase 1).

``gr_satellites_flowgraph`` accepts ``file=<SatYAML path>`` (not just ``norad=``), and in
``grc_block=True`` mode it reads only the ``transmitters`` block (no ``data``/datasink needed).
So we can hand gr-satellites a SatYAML built from the backend's explicit
``(modulation, baud, framing)`` and **reuse its full demod + ~50-deframer library for ANY bird,
catalogued or not** — as long as the modulation is one gr-satellites demodulates.

FSK covers 2-FSK / GFSK / GMSK / MSK (gr-satellites' deviation-based ``fsk_demodulator``); BPSK
covers (D)BPSK; AFSK is Bell-202. QAM / APSK / OFDM / QPSK have no gr-satellites demod → they
return None and are handled by our own modem (docs/08 Tier 2). This module is numpy/PyYAML-only
(no GNU Radio), so it is fully unit-testable.
"""
from __future__ import annotations

import yaml

# Our modulation family → gr-satellites SatYAML ``modulation`` value.
_GRSAT_MOD = {
    "gfsk": "FSK", "gmsk": "FSK", "fsk": "FSK", "msk": "FSK",
    "bpsk": "BPSK", "dbpsk": "BPSK", "psk": "BPSK",
    "afsk": "AFSK",
}
# Modulation index for the FSK deviation default (peak deviation = mod_index * baud / 2).
_MOD_INDEX = {"gmsk": 0.5, "msk": 0.5}
_DEFAULT_MOD_INDEX = 0.5  # most cubesat GFSK / 2-FSK ~ h = 0.5


def can_synthesize(modulation, baud, framing) -> bool:
    """True when the gr-satellites synthetic-SatYAML path applies: gr-satellites can demodulate
    the modulation (FSK/BPSK/AFSK family), ``baud`` is present, and ``framing`` is gr-satellites
    vocabulary (validated — a local-only token like ``"ax25"`` is NOT synthesizable). Lets the
    composer report the path without writing a file. NOTE: the runtime write path stays
    slightly more permissive (any truthy framing) because ``gr_satellites_flowgraph``'s own
    constructor is the authoritative validator and attempts are cheap + guarded."""
    import framings  # noqa: PLC0415 — outbound framing normalization (single point)
    kind = str(modulation or "").strip().lower()
    return bool(_GRSAT_MOD.get(kind) and baud and framings.to_grsatellites_framing(framing) is not None)


def synthetic_satyaml(norad, modulation, baud, framing, frequency_hz, *, name=None):
    """Return gr-satellites SatYAML **text** for a non-catalogued bird, or ``None`` when
    gr-satellites has no demodulator for ``modulation`` (QAM/APSK/OFDM/QPSK → our own modem) or
    ``framing``/``baud`` is missing. The ``framing`` string must be gr-satellites' vocabulary
    (e.g. ``"AX.25 G3RUH"``, ``"USP"``, ``"AX100 ASM+Golay"``, ``"CCSDS Concatenated"``) — which
    is exactly what the backend surfaces (it is scraped from gr-satellites' own SatYAML)."""
    import framings  # noqa: PLC0415 — sibling registry; the single (outbound) normalization point
    kind = str(modulation or "").strip().lower()
    mod = _GRSAT_MOD.get(kind)
    # Translate to gr-satellites vocabulary (local 'ax25' -> 'AX.25'); None ⇒ local-only framing
    # (endurosat/ccsds_tm/kiss) that gr-satellites can't build → not synthesizable, decode locally.
    gr_framing = framings.to_grsatellites_framing(framing)
    if not mod or gr_framing is None or not baud:
        return None
    tx: dict = {
        "frequency": float(frequency_hz or 0.0),
        "modulation": mod,
        "baudrate": int(baud),
        "framing": gr_framing,
    }
    if mod == "FSK":
        tx["deviation"] = int(round(_MOD_INDEX.get(kind, _DEFAULT_MOD_INDEX) * baud / 2.0))
    elif mod == "AFSK":
        tx["af_carrier"] = 1700  # Bell-202 tone centre
        tx["deviation"] = 500    # tone half-spacing (mark 1200 / space 2200)
    doc = {
        "name": name or f"NORAD-{int(norad)}",
        "norad": int(norad),
        "transmitters": {"downlink": tx},
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def write_synthetic_satyaml(path, norad, modulation, baud, framing, frequency_hz, *, name=None):
    """Build + write a synthetic SatYAML to ``path``; return ``path`` (to pass as
    ``gr_satellites_flowgraph(file=...)``), or ``None`` if gr-satellites can't demodulate it."""
    text = synthetic_satyaml(norad, modulation, baud, framing, frequency_hz, name=name)
    if text is None:
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

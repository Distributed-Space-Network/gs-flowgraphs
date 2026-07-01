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


def synthetic_satyaml(norad, modulation, baud, framing, frequency_hz, *, name=None):
    """Return gr-satellites SatYAML **text** for a non-catalogued bird, or ``None`` when
    gr-satellites has no demodulator for ``modulation`` (QAM/APSK/OFDM/QPSK → our own modem) or
    ``framing``/``baud`` is missing. The ``framing`` string must be gr-satellites' vocabulary
    (e.g. ``"AX.25 G3RUH"``, ``"USP"``, ``"AX100 ASM+Golay"``, ``"CCSDS Concatenated"``) — which
    is exactly what the backend surfaces (it is scraped from gr-satellites' own SatYAML)."""
    kind = str(modulation or "").strip().lower()
    mod = _GRSAT_MOD.get(kind)
    if not mod or not framing or not baud:
        return None
    tx: dict = {
        "frequency": float(frequency_hz or 0.0),
        "modulation": mod,
        "baudrate": int(baud),
        "framing": str(framing),
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

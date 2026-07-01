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
_LOCAL = ("ax25", "endurosat", "argos", "ccsds_tm", "kiss", "slip")
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


def normalize_framing(label) -> str | None:
    """Map ANY framing label — a local token or a backend/SatYAML label ("AX.25 G3RUH",
    "KISS", "CCSDS Concatenated") — to the LOCAL deframer that handles it, or ``None`` when no
    local deframer applies (the label is gr-satellites vocabulary decoded upstream, or unknown).

    This is the SINGLE normalization point for the whole system (docs/10 P0-2): gs-client passes
    the backend framing verbatim; this function decides what our in-process engine runs. Rules,
    most-specific first: AX.100 is NOT AX.25 (GOMspace — upstream); FX.25 is AX.25-compatible
    only after its RS layer, so it stays upstream too; "CCSDS Concatenated"/"Reed-Solomon" use
    conv/dual-basis-RS coding our ``ccsds_tm`` chain doesn't implement — upstream (only a plain
    CCSDS TM label maps locally)."""
    s = str(label or "").strip().lower()
    if not s:
        return None
    if s in _LOCAL:  # already a local token
        return s
    if "ax100" in s or "ax.100" in s or "fx.25" in s or "fx25" in s:
        return None  # AX.100 (Golay/RS) and FX.25 (RS-wrapped) → gr-satellites
    if "ax.25" in s or "ax25" in s or s == "aprs":
        return "ax25"  # G3RUH-scrambled and plain both tried by the deframer
    if "endurosat" in s or "airmac" in s:
        return "endurosat"
    if "argos" in s or "pmt-a3" in s:
        return "argos"
    if "kiss" in s:
        return "kiss"
    if "slip" in s:
        return "slip"
    if s in ("ccsds_tm", "ccsds tm", "ccsds", "ccsds aos"):
        return "ccsds_tm"  # bare TM/AOS chain; coded CCSDS variants stay upstream
    return None


def _bits_to_bytes_any_phase(arr: np.ndarray, decode) -> list[bytes]:
    """Byte-oriented framings (KISS/SLIP) after a bit-level demod have an arbitrary bit phase —
    recover it by trying all 8 alignments. KISS/SLIP carry NO checksum, so "first phase that
    yields frames" would happily return garbage from a wrong phase (any two spurious 0xC0 bytes
    bracket a "frame"); instead pick the phase with the MOST FEND (0xC0) delimiter bytes — the
    correct alignment maximizes it (every real frame contributes two), a wrong one only gets
    chance hits."""
    candidates = [bytes(np.packbits(arr[off:])) for off in range(8)]
    best = max(candidates, key=lambda b: b.count(0xC0))
    if best.count(0xC0) == 0:
        return []
    return decode(best)


def deframe(bits, framing_name: str | None = None) -> tuple[list[bytes], str | None]:
    """Hard bits → ``(frames, matched_framing)`` via the LOCAL numpy deframers. ``framing_name``
    (any vocabulary — normalized via :func:`normalize_framing`) runs ONLY that link layer;
    ``None`` auto-detects across the FCS-gated local set (``ax25``/``endurosat``) and reports
    which matched, so the caller can lock onto it.

    A gr-satellites-only framing label (e.g. ``"USP"``, ``"AX100 ASM+Golay"``) returns
    ``([], None)`` — that link layer is decoded upstream in the gr-satellites flowgraph
    (the synthetic-SatYAML path); the two run in parallel and the first valid frame wins."""
    from gfsk_ax25 import argos, ccsds, endurosat_link, kiss  # noqa: PLC0415
    from gfsk_ax25 import framing as ax25_framing  # noqa: PLC0415

    arr = np.asarray(bits, dtype=np.uint8)
    if not len(arr):
        return [], None
    if framing_name:
        local = normalize_framing(framing_name)
        order = [local] if local else []  # non-local label → decoded upstream, not here
    else:
        order = list(_LOCAL_AUTODETECT)
    for name in order:
        if name == "endurosat":
            frames = endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)
        elif name == "ax25":  # G3RUH-descrambled and plain — same framing, different descrambling
            frames = []
            for scramble in (True, False):
                frames.extend(ax25_framing.decode(arr, scramble=scramble, nrzi=True))
        elif name == "argos":
            # Explicit only. The documented PTT-A2 sync is BENCH-CONFIRM (argos.py); until then
            # this registry path uses it deliberately — the module API requires it be spelled out.
            frames = argos.deframe(
                arr, sync=argos.ARGOS_PTT_A2_SYNC, sync_bits=argos.ARGOS_PTT_A2_SYNC_BITS)
        elif name == "ccsds_tm":  # explicit only — ASM+RS(255,223)+randomize+FECF chain
            frames = ccsds.deframe_tm(arr)
        elif name == "kiss":   # byte-oriented TNC framing — all 8 bit phases tried
            frames = _bits_to_bytes_any_phase(arr, kiss.kiss_decode)
        elif name == "slip":
            frames = _bits_to_bytes_any_phase(arr, kiss.slip_decode)
        else:
            continue
        if frames:
            return frames, name
    return [], None

"""Framing registry — framing name → deframe (docs/08 Tier 1).

Two classes of framing, so the registry is the single source of "what we can deframe":

  * **Local (numpy, here):** ``ax25`` (G3RUH-scrambled + plain), ``endurosat``/AirMAC,
    ``ccsds_tm``, and ``kiss``. (``argos`` and ``slip`` are deliberately NOT wired — see the
    ``_LOCAL`` comment below.)
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

# Link layers our own engine deframes in-process (numpy). ``ccsds_tm``/``kiss`` run ONLY when
# explicitly requested (CCSDS TM needs per-bird channel-coding params; KISS has no checksum) —
# kept out of autodetect to avoid spurious/mis-parametrized runs.
# ``argos`` is deliberately NOT here: its documented sync is an 8-bit PLACEHOLDER and the
# BCH(31,21) accepts ~48% of random words, so running it floods noise captures with false
# frames (which would even win the engine race and gate off gr-satellites). The module
# (gfsk_ax25.argos) stays, fully tested, for when the real full-length sync is bench-confirmed;
# until then an argos bird plans as record-only.
# ``slip`` is NOT here either: with no checksum AND no type byte, strict gating still passes
# ~30 garbage frames per noise drain — unusable on a demodulated bitstream. The codec
# (gfsk_ax25.kiss.slip_*) remains for its real use, byte-exact TNC/serial pipes (uplink/relay).
_LOCAL = ("ax25", "endurosat", "ccsds_tm", "kiss")
_LOCAL_AUTODETECT = ("ax25", "endurosat")

# Local framings whose deframer verifies a REAL integrity check (AX.25 FCS, EnduroSat CRC-16,
# CCSDS RS+FECF) before emitting a frame. Only these may declare a win in the engine race
# (docs/10 MED-1): ``kiss`` carries NO checksum and measurably passes ~2 chance "frames" per
# noise drain, so a KISS hit on the first drain must never gate off the real gr-satellites
# decoder for the whole pass. KISS frames are still emitted as products — they just can't gate.
_CRC_GATED = ("ax25", "endurosat", "ccsds_tm")

# docs/13 live-vs-post-pass split (single-sourced here so the RX engine and the post-pass decoder
# can never drift). LIVE = the LIGHT framings the RX engines run in real time, both at once (the
# CRC-gated autodetect set). POST_PASS = the OTHER CRC-gated local framings — swept AFTER LOS on
# the recorded .cf32, so they cost no pass-time CPU. Only CRC-gated framings are auto-swept: a
# blind whole-pass sweep with an unchecked framing (``kiss`` — no checksum) would emit ~noise
# "frames", so ``kiss`` is excluded from the default and decoded only on explicit request.
LIVE_FRAMINGS = _LOCAL_AUTODETECT  # ("ax25", "endurosat")
POST_PASS_FRAMINGS = tuple(f for f in _CRC_GATED if f not in _LOCAL_AUTODETECT)  # ("ccsds_tm",)

# The gr-satellites framing vocabulary (SatYAML ``framing:`` strings) reused via synthetic
# SatYAML — advertised here, decoded in the gr-satellites flowgraph. Representative of the ~50
# deframers (docs/08 §2); the backend surfaces these verbatim (scraped from gr-satellites).
GRSATELLITES_FRAMINGS = (
    "AX.25", "AX.25 G3RUH", "AX100 Mode 5", "AX100 Mode 6", "AX100 ASM+Golay",
    "USP", "Mobitex", "Mobitex-NX", "GEOSCAN", "AO-40 FEC", "AO-40 FEC short",
    "AO-40 uncoded", "CCSDS Concatenated", "CCSDS Reed-Solomon", "CCSDS Uncoded",
    "NGHam", "NGHam no Reed Solomon",
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


def autodetect_framings() -> tuple[str, ...]:
    """Local framings :func:`deframe` tries when NO framing label is given (all CRC-gated by
    construction — see the ``_LOCAL_AUTODETECT`` comment). The composer derives the
    absent-framing decode plan from this same tuple so plan and engine cannot drift
    (docs/J LOW-2)."""
    return _LOCAL_AUTODETECT


def crc_gated_framings() -> tuple[str, ...]:
    """Local framings whose deframer validates a real integrity check (FCS/CRC/RS) — the only
    ones allowed to declare an engine-race win (docs/10 MED-1)."""
    return _CRC_GATED


def is_crc_gated(label) -> bool:
    """True when ``label`` (ANY vocabulary — local token or backend/SatYAML label) normalizes to
    a local deframer that validates a real integrity check before emitting a frame. Frames from a
    non-gated framing (``kiss`` — no checksum) are still products, but they may NOT win the
    engine race and starve gr-satellites (docs/10 MED-1)."""
    return normalize_framing(label) in _CRC_GATED


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
    if "kiss" in s:
        return "kiss"
    # "slip": byte-pipe codec only — ungateable on a demodulated bitstream (no checksum, no
    # type byte); a SLIP label plans as record-only.
    # NOTE: the explicit local token "ccsds_tm" is handled by the _LOCAL early-return above.
    # A bare/coded "CCSDS" LABEL never maps locally: spec CCSDS uses dual-basis RS/concatenated
    # coding our local TM chain does not implement — those decode upstream (gr-satellites).
    # "argos"/PMT-A3: module exists but its sync is a placeholder — record-only until benched.
    return None


# OUTBOUND normalization (the reverse of :func:`normalize_framing`): a LOCAL token that
# gr-satellites can ALSO build → its SatYAML ``framing:`` label. Only ``ax25`` qualifies; the
# other local tokens are decoded ONLY in-process (``endurosat``/AirMAC is not gr-satellites
# vocabulary; ``ccsds_tm`` needs per-bird qualified coding gr-satellites can't infer; ``kiss`` is
# a TNC byte framing, not a SatYAML framing) → they must NOT be handed to gr-satellites.
# Bare ``ccsds`` is here too — NOT because it's local (it isn't; :func:`normalize_framing` returns
# None for it) but because it's UNBUILDABLE: gr-satellites has only QUALIFIED CCSDS framings
# ("CCSDS Concatenated"/"Reed-Solomon"/"Uncoded"); a bare "CCSDS" carries no FEC to pick one, so
# its synthetic constructor would throw. It plans + records as record-only, never synthesizable.
# The qualified labels have a distinct lowercase key ("ccsds concatenated" …) and pass verbatim.
_GRSAT_LABEL = {"ax25": "AX.25"}
_NO_GRSAT = frozenset({"endurosat", "airmac", "ccsds_tm", "ccsds", "kiss"})


def to_grsatellites_framing(label) -> str | None:
    """Map ANY framing label to a gr-satellites SatYAML ``framing:`` string for the synthetic
    SatYAML path, or ``None`` when gr-satellites has no synthetic deframer for it (a local-only
    token → decode in-process / record-only).

    This is the OUTBOUND half of the single normalization point (docs/10 P0-2). A recognized
    local token is translated (``ax25`` → ``AX.25``); a local-only token returns ``None``; every
    OTHER label passes through VERBATIM — the ``gr_satellites_flowgraph`` constructor is the
    authoritative validator and :data:`GRSATELLITES_FRAMINGS` is deliberately non-exhaustive, so
    real gr-satellites labels (``AX.25 G3RUH``, ``USP``, ``AX100 Mode 5``, …) must not be
    collapsed or dropped here."""
    s = str(label or "").strip()
    if not s:
        return None
    low = s.lower()
    if low in _NO_GRSAT:
        return None
    return _GRSAT_LABEL.get(low, s)


def _bits_to_bytes_any_phase(arr: np.ndarray, decode) -> list[bytes]:
    """Byte-oriented framings (KISS/SLIP) after a bit-level demod have an arbitrary bit phase —
    recover by decoding ALL 8 alignments in strict mode and returning the deduped UNION. Picking
    a single "best" phase (by frame count or FEND density) measurably LOSES the real frame ~10%
    of the time under noise: a wrong phase's chance frames can tie or beat the true phase's
    count. The union never loses the real frame (its phase is always included); the cost is that
    wrong phases can contribute a little extra chance garbage — KISS/SLIP carry NO checksum, so
    some noise acceptance is irreducible (quantified in gfsk_ax25.kiss and the regression test).
    """
    out: list[bytes] = []
    seen_prior_phases: set[bytes] = set()
    for off in range(8):
        seg = arr[off:]
        seg = seg[: seg.size - (seg.size % 8)]  # packbits ZERO-pads the tail — the pad can
        # fabricate a trailing 0xC0 (11000000) and promote an unterminated chunk to a
        # "bracketed" frame with a silently truncated payload
        phase_frames = decode(bytes(np.packbits(seg)), strict=True)
        for f in phase_frames:
            # Dedup ACROSS phases only: a real frame decodes at exactly ONE alignment, so a
            # cross-phase duplicate is chance garbage — but two identical frames at the SAME
            # phase are a genuine fast repeat beacon and must BOTH be kept. Known residual: a
            # mid-window demod clock SLIP puts genuinely-distinct identical beacons at two
            # phases; the second is deduped (one frame lost per slip — accepted, KISS has no
            # sequence numbers to distinguish them).
            if f not in seen_prior_phases:
                out.append(f)
        seen_prior_phases.update(phase_frames)
    return out


def grsat_deframer_plan(framing) -> list[tuple]:
    """PURE: framing label → the gr-satellites deframers to build for it, as ``(kind, *args)``
    tuples (``gnuradio_satellites.make_grsat_deframers`` builds the actual hier-blocks from this).

    Dots AND underscores are stripped first so the SatNOGS spelling ``"AX.100 Mode 5"`` matches the
    ``ax100`` checks — the builder used to test ``"ax100" in f`` and silently MISSED ``"ax.100"``,
    so every AX.100 bird built NO deframer even with GS_GRSAT_LIVE=1. Returns ``[]`` for a framing
    gr-satellites has no deframer for here (our numpy engine / record-only carries it)."""
    f = str(framing or "").strip().lower().replace(".", "").replace("_", " ")
    if not f:
        return []
    if "g3ruh" in f:  # AX.25, G3RUH-scrambled
        return [("ax25", True)]
    if f in ("ax25", "aprs"):  # both scramblings (mirrors :func:`deframe`)
        return [("ax25", False), ("ax25", True)]
    if "ax100" in f and ("5" in f or "asm" in f or "golay" in f):
        return [("ax100", "ASM")]
    if "ax100" in f and ("6" in f or "rs" in f or "reed" in f):
        return [("ax100", "RS")]
    if f == "usp":
        return [("usp",)]
    if f == "endurosat":
        return [("endurosat",)]
    return []


def _valid_ax25_address(body: bytes) -> bool:
    """Reject a CRC-16 FALSE POSITIVE. AX.25's FCS is only 16 bits, so over a noisy pass a random
    flag-delimited chunk passes the CRC ~1/65536 of the time and is emitted as a "frame" of garbage
    (seen on the bench: a frame whose address bytes decode to control chars, not callsigns). A REAL
    AX.25 frame's 14-byte address field decodes to shifted-ASCII callsigns — each char = ``byte>>1``
    ∈ ``A-Z`` / ``0-9`` / space. This costs nothing and rejects NO valid AX.25 frame (verified
    against real SatNOGS captures: ``CQ``/``WQ2XKO`` pass)."""
    if len(body) < 15:  # 14-byte address field + at least a control byte
        return False
    for base in (0, 7):  # destination + source callsigns (6 chars each; SSID byte 6/13 not checked)
        for i in range(6):
            c = body[base + i] >> 1
            if not (0x41 <= c <= 0x5A or 0x30 <= c <= 0x39 or c == 0x20):  # A-Z / 0-9 / space
                return False
    return True


# Public alias: the live RX engines (cubesat_gfsk_ax25_rx) apply this same CRC-16-false-positive
# guard on their AX.25 frames before emitting — their StreamDecoder / framing.decode path gates only
# on the 16-bit FCS, so without this a noise chunk that passes the FCS would be emitted as a frame.
valid_ax25_address = _valid_ax25_address


def deframe(bits, framing_name: str | None = None) -> tuple[list[bytes], str | None]:
    """Hard bits → ``(frames, matched_framing)`` via the LOCAL numpy deframers. ``framing_name``
    (any vocabulary — normalized via :func:`normalize_framing`) runs ONLY that link layer;
    ``None`` auto-detects across the FCS-gated local set (``ax25``/``endurosat``) and reports
    which matched, so the caller can lock onto it.

    A gr-satellites-only framing label (e.g. ``"USP"``, ``"AX100 ASM+Golay"``) returns
    ``([], None)`` — that link layer is decoded upstream in the gr-satellites flowgraph
    (the synthetic-SatYAML path); the two run in parallel and the first valid frame wins."""
    from gfsk_ax25 import ccsds, endurosat_link, kiss  # noqa: PLC0415
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
            # Reject CRC-16 false positives (noise that passed the 16-bit FCS) — a real frame has a
            # valid callsign address; garbage does not. Keeps decode output trustworthy (bench).
            frames = [f for f in frames if _valid_ax25_address(f)]
        elif name == "ccsds_tm":  # explicit only — ASM+RS(255,223)+randomize+FECF chain
            frames = ccsds.deframe_tm(arr)
        elif name == "kiss":   # byte-oriented TNC framing — all 8 bit phases tried
            frames = _bits_to_bytes_any_phase(arr, kiss.kiss_decode)
        else:
            continue
        if frames:
            return frames, name
    return [], None

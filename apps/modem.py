"""Modem registry — modulation family → GNU Radio demod / mod chain (docs/08 Tier 1).

The modulation↔chain dispatch that used to live inline in ``gnuradio_satellites`` lives here,
so a new modulation plugs in as an isolated registry entry instead of surgery on the engine.

Two layers, deliberately split so the taxonomy is testable without GNU Radio:
  * ``modulation_spec(kind)`` — a PURE classifier (family, PSK order, differential/offset,
    FSK-ness, Tier). numpy/stdlib-only, fully unit-tested.
  * ``build_demod`` / ``build_mod`` — construct the actual DSP chains; they import GNU Radio
    (via ``gnuradio_gfsk``) LAZILY so this module stays import-safe for the numpy engine + tests.

Tier 1 (this file): the full **FSK family** (2-FSK/GFSK/GMSK/MSK/CPFSK, + M-FSK build-pending) and
**PSK family** (BPSK/DBPSK, QPSK/DQPSK/OQPSK, 8-PSK) + AFSK, and their modulators. Tier 2 (QAM/
APSK/OFDM/DVB-S2) classify as ``tier=2`` and return no Tier-1 chain — they route to our own
higher-order modem / gr-dvbs2rx later.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Taxonomy (pure) ──────────────────────────────────────────────────────────────────────────
# 2-level FSK family — all share the deviation-based FSK demod (deviation from mod index).
# (G)MSK and CPFSK are h≈0.5 continuous-phase FSK; 2-FSK/GFSK differ only in the TX pulse shape,
# which the RX matched filter handles identically.
_FSK_2LEVEL = ("fsk", "2fsk", "gfsk", "gmsk", "msk", "cpfsk", "cpm", "ffsk")
# M-ary FSK — needs an M-level frequency slicer (build-pending on the bench).
_MFSK = {"4fsk": 4, "mfsk": 4, "gfsk4": 4, "8fsk": 8}
# PSK family → constellation order.
_PSK_ORDER = {
    "bpsk": 2, "dbpsk": 2, "psk": 2,
    "bpskmanchester": 2, "dbpskmanchester": 2,
    "qpsk": 4, "dqpsk": 4, "oqpsk": 4,
    "8psk": 8, "psk8": 8,
}
_DIFFERENTIAL = frozenset({"dbpsk", "dbpskmanchester", "dqpsk"})
_OFFSET = frozenset({"oqpsk"})                   # offset QPSK (I/Q half-symbol stagger)
_MANCHESTER = frozenset({"bpskmanchester", "dbpskmanchester"})
# Tier 2 — higher-order / multicarrier. Classified into distinct families with a constellation
# order; the demod/mod chains are GNU Radio (+ gr-dvbs2rx for DVB-S2), constructed on the bench.
_QAM_ORDER = {"qam": 16, "qam16": 16, "qam32": 32, "qam64": 64, "qam128": 128, "qam256": 256}
_APSK_ORDER = {"apsk": 16, "apsk16": 16, "apsk32": 32}
_OFDM = ("ofdm",)
_DVBS2 = ("dvbs2", "dvb-s2", "dvbs2x", "dvb-s2x")
# Tier 3 — long tail. OOK/ASK + CW/Morse are numpy codecs (offline, on a captured .cf32); the
# analog families (NBFM/WFM/AM) are GNU Radio. All classify tier=3.
_OOK = {"ook": 2, "ask": 2, "2ask": 2, "4ask": 4, "mask": 4}
_CW = ("cw", "morse")
_ANALOG = {"nbfm": "nbfm", "fm": "nbfm", "wfm": "wfm", "am": "am"}


@dataclass(frozen=True)
class ModSpec:
    """A pure classification of a modulation string (no GNU Radio)."""
    kind: str                 # normalized input
    family: str               # "fsk" | "mfsk" | "psk" | "afsk" | "tier2"
    order: int = 2            # constellation / FSK levels (2 = binary)
    differential: bool = False
    offset: bool = False      # OQPSK I/Q stagger
    manchester: bool = False  # two BPSK half-symbols per decoded symbol
    tier: int = 1             # 1 = built here; 2 = higher-order (routed elsewhere)

    @property
    def supported(self) -> bool:
        """True when Tier 1 can build a demod for it today (M-FSK/8-PSK are build-pending on
        the bench but still classify as Tier 1 — see ``build_demod`` for what constructs now)."""
        return self.tier == 1


def _norm(kind: str) -> str:
    return (kind or "").strip().lower().replace("_", "").replace(" ", "")


def modulation_spec(kind: str) -> ModSpec | None:
    """Classify a modulation string into a :class:`ModSpec`, or ``None`` if unrecognized.

    Recognizes the Tier-1 FSK/PSK/AFSK families (incl. differential/offset variants and M-FSK)
    and the Tier-2 higher-order families (QAM/APSK/OFDM/DVB-S2). Pure — no GNU Radio."""
    k = _norm(kind)
    if not k:
        return None
    if k in _FSK_2LEVEL:
        return ModSpec(k, "fsk", order=2)
    if k in _MFSK:
        return ModSpec(k, "mfsk", order=_MFSK[k])
    if k in _PSK_ORDER:
        return ModSpec(
            k, "psk", order=_PSK_ORDER[k],
            differential=k in _DIFFERENTIAL, offset=k in _OFFSET,
            manchester=k in _MANCHESTER,
        )
    if k == "afsk":
        return ModSpec(k, "afsk", order=2)
    if k in _QAM_ORDER:
        return ModSpec(k, "qam", order=_QAM_ORDER[k], tier=2)
    if k in _APSK_ORDER:
        return ModSpec(k, "apsk", order=_APSK_ORDER[k], tier=2)
    if k in _OFDM:
        return ModSpec(k, "ofdm", order=0, tier=2)
    if k in _DVBS2:
        return ModSpec(k, "dvbs2", order=0, tier=2)
    if k in _OOK:
        return ModSpec(k, "ook", order=_OOK[k], tier=3)
    if k in _CW:
        return ModSpec(k, "cw", order=2, tier=3)
    if k in _ANALOG:
        return ModSpec(k, _ANALOG[k], order=0, tier=3)
    return None


_TIER2_KEYS = set(_QAM_ORDER) | set(_APSK_ORDER) | set(_OFDM) | set(_DVBS2)
_TIER3_KEYS = set(_OOK) | set(_CW) | set(_ANALOG)


def demod_families() -> set[str]:
    """Every modulation key the modem recognizes for RX (Tiers 1–3). Tier-2 builds via GNU Radio /
    gr-dvbs2rx; Tier-3 OOK/CW are numpy codecs and the analog families are GNU Radio (bench)."""
    return (set(_FSK_2LEVEL) | set(_MFSK) | set(_PSK_ORDER) | {"afsk"}
            | _TIER2_KEYS | _TIER3_KEYS)


def mod_families() -> set[str]:
    """Every modulation key we can TX-modulate (Tiers 1–3)."""
    return (set(_FSK_2LEVEL) | set(_MFSK) | set(_PSK_ORDER) | {"afsk"}
            | _TIER2_KEYS | _TIER3_KEYS)


# ── Chain construction (GNU Radio, lazy) ─────────────────────────────────────────────────────
def build_demod(
    kind: str,
    tb,
    src,
    sample_rate: float,
    symbol_rate: float,
    *,
    differential: bool | None = None,
    channel_bw_hz: float | None = None,
    collect_hard: bool = True,
):
    """Build the GNU Radio demod chain for ``kind`` tapping ``src`` (already at the channel rate)
    and return ``(bit_sink, soft_tap)``: the hard-bit sink (``drain()`` → hard bits) and — for the
    FSK family only — the FLOAT soft-symbol tap that gr-satellites deframer components consume
    (``None`` for non-FSK). ``(None, None)`` when the modulation is unsupported / build-pending.

    FSK → tuned quadrature-demod chain (exposes the soft tap); PSK (2/4/8) → FLL+Costas chain
    (order from the spec); AFSK → FM-demod + tone xlate → FSK chain. Tier 2 (QAM/APSK/OFDM/DVB-S2)
    routes to the ``gnuradio_hirate`` bench constructors. M-FSK and offset/8-PSK are classified but
    their dedicated slicer/loop is bench build-pending (return ``(None, None)``, never crash).

    ``differential`` is the backend's per-bird DxPSK flag (rfLink → params ``differential``):
    True/False is honoured by the PSK chain; ``None`` (backend didn't say) keeps the robust
    differential-on default — most cubesat PSK downlinks are differentially encoded.

    ``collect_hard=False`` is for an FSK soft-only consumer such as native USP or a decoupled
    gr-satellites deframer. The slicer is terminated without constructing a Python queue. Other
    modem families have no soft tap, so they return no chain when hard symbols are not requested.
    """
    import logging  # noqa: PLC0415

    spec = modulation_spec(kind)
    if spec is None:
        return None, None  # unrecognized — return BEFORE importing GNU Radio (import-safe)
    if not collect_hard and spec.family != "fsk":
        return None, None
    if spec.tier == 2:
        return _build_tier2_demod(spec, tb, src, sample_rate, symbol_rate), None
    if spec.tier == 3:
        return _build_tier3_demod(spec, tb, src, sample_rate), None

    from gnuradio_gfsk import (  # noqa: PLC0415 — GNU Radio only; keeps this module import-safe
        connect_afsk_demod,
        connect_gfsk_demod,
        connect_psk_demod,
    )

    from gfsk_ax25 import endurosat  # noqa: PLC0415

    if spec.family == "fsk":
        mod_index = 0.5 if spec.kind in ("gmsk", "msk", "cpfsk", "cpm") \
            else endurosat.LinkProfile().mod_index
        profile = endurosat.LinkProfile(
            symbol_rate_hz=symbol_rate or 9600.0, mod_index=mod_index)
        # (bit_sink, soft_tap): the caller feeds bits to our numpy deframers and the soft tap to
        # the gr-satellites deframers (docs/12 §L.7 Phase 3).
        return connect_gfsk_demod(
            tb, src, sample_rate, profile, decimate=False, sdr_rate=sample_rate,
            channel_bw_hz=channel_bw_hz, collect_hard=collect_hard)
    if spec.family == "psk":
        if spec.manchester:
            logging.getLogger("modem").info(
                "modem: %s Manchester recovery is post-pass-only", kind
            )
            return None, None
        # RX differential precedence: the backend's per-bird rfLink flag (``differential`` param)
        # when supplied wins; otherwise the robust default True — most cubesat PSK downlinks are
        # differentially encoded, a bare "bpsk" name doesn't say, and a "dbpsk" name agrees with
        # the default anyway. order/offset route 8-PSK/OQPSK (bench-pending loop tuning). No float
        # soft tap — PSK deframers take complex symbols, not the FSK float tap.
        rx_diff = differential if differential is not None else True
        return connect_psk_demod(
            tb, src, sample_rate, symbol_rate or 1200.0,
            order=spec.order, differential=rx_diff, offset=spec.offset), None
    if spec.family == "afsk":
        return connect_afsk_demod(tb, src, sample_rate, baud=symbol_rate or 1200.0), None
    # mfsk — dedicated M-level slicer not built yet on the bench.
    logging.getLogger("modem").info("modem: %s (%s) build-pending; skipping", kind, spec.family)
    return None, None


def _build_tier2_demod(spec, tb, src, sample_rate: float, symbol_rate: float):
    """Route a Tier-2 spec to the ``gnuradio_hirate`` bench constructor, guarded so a box without
    GNU Radio / gr-dvbs2rx gets ``None`` (import-safe) rather than an ImportError."""
    import logging  # noqa: PLC0415

    try:
        import gnuradio_hirate  # noqa: PLC0415 — GNU Radio (+ gr-dvbs2rx) bench module
    except Exception as e:  # noqa: BLE001 — no GNU Radio here; caller treats as build-pending
        logging.getLogger("modem").info("modem: %s demod needs GNU Radio (%s)", spec.kind, e)
        return None
    if spec.family == "qam":
        return gnuradio_hirate.connect_qam_demod(
            tb, src, sample_rate, symbol_rate, order=spec.order)
    if spec.family == "apsk":
        return gnuradio_hirate.connect_apsk_demod(
            tb, src, sample_rate, symbol_rate, order=spec.order)
    if spec.family == "ofdm":
        return gnuradio_hirate.connect_ofdm_demod(tb, src, sample_rate)
    if spec.family == "dvbs2":
        return gnuradio_hirate.connect_dvbs2_demod(tb, src, sample_rate, symbol_rate)
    return None


def _build_tier3_demod(spec, tb, src, sample_rate: float):
    """Tier 3. The analog families (NBFM/WFM/AM) build a GNU Radio analog demod (bench, guarded).
    OOK/ASK and CW/Morse are offline numpy codecs (:mod:`gfsk_ax25.ook` / :mod:`gfsk_ax25.morse`)
    run on the captured .cf32 post-pass, not in-flowgraph bit-sink chains → return None with a
    pointer."""
    import logging  # noqa: PLC0415

    if spec.family in ("nbfm", "wfm", "am"):
        try:
            import gnuradio_hirate  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            logging.getLogger("modem").info("modem: %s demod needs GNU Radio (%s)", spec.kind, e)
            return None
        return gnuradio_hirate.connect_analog_demod(tb, src, sample_rate, spec.family)
    logging.getLogger("modem").info(
        "modem: %s is an offline numpy codec (gfsk_ax25.%s); run post-pass on the .cf32",
        spec.kind, "ook" if spec.family == "ook" else "morse")
    return None


def build_mod(kind: str, tb, src, sample_rate: float, symbol_rate: float):
    """Build the GNU Radio **modulator** chain for ``kind`` (TX): **packed bytes** in → complex
    IQ out, or ``None`` if unsupported / build-pending. Lazily imports ``gnuradio.digital``;
    bench-only.

    INPUT CONTRACT: GNU Radio's hierblock modulators (``gfsk_mod``/``psk_mod``/``generic_mod``)
    default to ``do_unpack=True`` — they expect PACKED bytes and unpack internally. Feed
    ``np.packbits(bits)``, never unpacked 0/1 bits (each bit would become 8 symbols).

    Maps the Tier-1 families to GNU Radio hierblocks: FSK→``gfsk_mod``/``gmsk_mod``, PSK→
    ``psk_mod``. Construction is confirmed on the bench (no GNU Radio in CI); the classification
    that selects the block is unit-tested via :func:`modulation_spec`."""
    import logging  # noqa: PLC0415

    spec = modulation_spec(kind)
    if spec is None:
        return None
    if spec.tier == 2:  # QAM/APSK/OFDM/DVB-S2 modulators live in the hirate bench module
        try:
            import gnuradio_hirate  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001 — no GNU Radio here
            logging.getLogger("modem").info("modem: %s TX needs GNU Radio (%s)", kind, e)
            return None
        return gnuradio_hirate.build_tier2_mod(spec, tb, src, sample_rate, symbol_rate)
    if spec.tier == 3:  # OOK/CW TX are numpy codecs; analog TX uses the FM apps → not a GR block
        logging.getLogger("modem").info(
            "modem: %s TX is a numpy codec / analog app, not a GR modulator block", kind)
        return None
    try:
        from gnuradio import digital  # noqa: PLC0415 — bench-only
    except Exception:  # noqa: BLE001 — no GNU Radio here; caller treats as build-pending
        logging.getLogger("modem").info("modem: gnuradio.digital unavailable; %s TX pending", kind)
        return None
    sps = max(2, int(round(sample_rate / max(symbol_rate, 1.0))))
    if spec.family == "fsk":
        # h=0.5 across the whole FSK family: it is what our RX chain and grsat_synth's
        # deviation default assume — a TX/RX modulation-index mismatch would put a 2x
        # deviation error on our own loopback.
        h = 0.5
        return digital.gfsk_mod(samples_per_symbol=sps, sensitivity=(3.14159 * h) / sps)
    if spec.family == "psk":
        return digital.psk_mod(
            constellation_points=spec.order, differential=spec.differential, samples_per_symbol=sps)
    logging.getLogger("modem").info("modem: %s TX build-pending; skipping", kind)
    return None

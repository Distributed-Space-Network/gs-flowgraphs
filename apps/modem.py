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
    "qpsk": 4, "dqpsk": 4, "oqpsk": 4,
    "8psk": 8, "psk8": 8,
}
_DIFFERENTIAL = frozenset({"dbpsk", "dqpsk"})   # differential encoding (DxPSK)
_OFFSET = frozenset({"oqpsk"})                   # offset QPSK (I/Q half-symbol stagger)
# Tier 2 — higher-order / multicarrier. Classified into distinct families with a constellation
# order; the demod/mod chains are GNU Radio (+ gr-dvbs2rx for DVB-S2), constructed on the bench.
_QAM_ORDER = {"qam": 16, "qam16": 16, "qam32": 32, "qam64": 64, "qam128": 128, "qam256": 256}
_APSK_ORDER = {"apsk": 16, "apsk16": 16, "apsk32": 32}
_OFDM = ("ofdm",)
_DVBS2 = ("dvbs2", "dvb-s2", "dvbs2x", "dvb-s2x")


@dataclass(frozen=True)
class ModSpec:
    """A pure classification of a modulation string (no GNU Radio)."""
    kind: str                 # normalized input
    family: str               # "fsk" | "mfsk" | "psk" | "afsk" | "tier2"
    order: int = 2            # constellation / FSK levels (2 = binary)
    differential: bool = False
    offset: bool = False      # OQPSK I/Q stagger
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
    return None


_TIER2_KEYS = set(_QAM_ORDER) | set(_APSK_ORDER) | set(_OFDM) | set(_DVBS2)


def demod_families() -> set[str]:
    """Every modulation key the modem recognizes for RX (Tier 1 + Tier 2). Tier-2 keys build via
    GNU Radio / gr-dvbs2rx on the bench. Used by callers to decide whether to attempt a build."""
    return set(_FSK_2LEVEL) | set(_MFSK) | set(_PSK_ORDER) | {"afsk"} | _TIER2_KEYS


def mod_families() -> set[str]:
    """Every modulation key we can TX-modulate (Tier 1 + Tier 2)."""
    return set(_FSK_2LEVEL) | set(_MFSK) | set(_PSK_ORDER) | {"afsk"} | _TIER2_KEYS


# ── Chain construction (GNU Radio, lazy) ─────────────────────────────────────────────────────
def build_demod(kind: str, tb, src, sample_rate: float, symbol_rate: float):
    """Build the GNU Radio demod chain for ``kind`` tapping ``src`` (already at the channel rate)
    and return its bit sink (``drain()`` → hard bits), or ``None`` if unsupported / build-pending.

    FSK → tuned quadrature-demod chain; PSK (2/4/8) → FLL+Costas chain (order from the spec);
    AFSK → FM-demod + tone xlate → FSK chain. Tier 2 (QAM/APSK/OFDM/DVB-S2) routes to the
    ``gnuradio_hirate`` bench constructors. M-FSK and offset/8-PSK are classified but their
    dedicated slicer/loop is bench build-pending (return ``None`` with a log, never crash)."""
    import logging  # noqa: PLC0415

    spec = modulation_spec(kind)
    if spec is None:
        return None  # unrecognized — return BEFORE importing GNU Radio (import-safe)
    if spec.tier == 2:
        return _build_tier2_demod(spec, tb, src, sample_rate, symbol_rate)

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
        return connect_gfsk_demod(
            tb, src, sample_rate, profile, decimate=False, sdr_rate=sample_rate)
    if spec.family == "psk":
        # RX: differential decode is the robust default for the fallback race — most cubesat PSK
        # downlinks are differentially encoded, and the modulation NAME ("bpsk") doesn't tell us
        # (gr-satellites carries that in a separate SatYAML flag). A per-bird `differential`
        # rfLink field should drive this later (backend follow-up); spec.differential (name-
        # derived) stays authoritative for TX. order/offset route 8-PSK/OQPSK (bench-pending loop).
        return connect_psk_demod(
            tb, src, sample_rate, symbol_rate or 1200.0,
            order=spec.order, differential=True, offset=spec.offset)
    if spec.family == "afsk":
        return connect_afsk_demod(tb, src, sample_rate, baud=symbol_rate or 1200.0)
    # mfsk — dedicated M-level slicer not built yet on the bench.
    logging.getLogger("modem").info("modem: %s (%s) build-pending; skipping", kind, spec.family)
    return None


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


def build_mod(kind: str, tb, src, sample_rate: float, symbol_rate: float):
    """Build the GNU Radio **modulator** chain for ``kind`` (TX): bytes/bits in → complex IQ out,
    or ``None`` if unsupported / build-pending. Lazily imports ``gnuradio.digital``; bench-only.

    Maps the Tier-1 families to GNU Radio hierblocks: FSK→``gfsk_mod``/``gmsk_mod``, PSK→
    ``psk_mod``/``constellation_modulator``. Construction is confirmed on the bench (no GNU Radio
    in CI); the classification that selects the block is unit-tested via :func:`modulation_spec`."""
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
    try:
        from gnuradio import digital  # noqa: PLC0415 — bench-only
    except Exception:  # noqa: BLE001 — no GNU Radio here; caller treats as build-pending
        logging.getLogger("modem").info("modem: gnuradio.digital unavailable; %s TX pending", kind)
        return None
    sps = max(2, int(round(sample_rate / max(symbol_rate, 1.0))))
    if spec.family == "fsk":
        h = 0.5 if spec.kind in ("gmsk", "msk", "cpfsk", "cpm") else 1.0
        return digital.gfsk_mod(samples_per_symbol=sps, sensitivity=(3.14159 * h) / sps)
    if spec.family == "psk":
        return digital.psk_mod(
            constellation_points=spec.order, differential=spec.differential, samples_per_symbol=sps)
    logging.getLogger("modem").info("modem: %s TX build-pending; skipping", kind)
    return None

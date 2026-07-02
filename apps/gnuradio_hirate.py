"""High-rate / higher-order modem chains — QAM, APSK, OFDM, DVB-S2 (docs/08 Tier 2, bench).

Imported ONLY when the modem registry routes a Tier-2 modulation (via ``modem._build_tier2_demod``
/ ``build_mod``), so it may import ``gnuradio`` (and, for DVB-S2, the ``dvbs2rx`` OOT) at load.
It assembles the standard GNU Radio hierblocks; the coherent front end (AGC → matched filter →
symbol sync → carrier recovery → constellation decode) mirrors ``gnuradio_gfsk.connect_psk_demod``.

Status: BENCH-VALIDATION-PENDING — there is no GNU Radio in CI, so these chains are wired +
syntax-checked here and tuned/confirmed on the bench (loop gains and the OFDM/DVB-S2 packet-sink
wiring especially). DVB-S2/S2X requires the ``gr-dvbs2rx`` dependency (GPLv3); if it is absent the
constructor returns ``None`` and the caller treats the modulation as build-pending.

License: GPLv3 (see ../COPYING).
"""
from __future__ import annotations

import logging

from gnuradio import blocks, digital
from gnuradio import filter as gr_filter
from gnuradio_gfsk import _BitSink, _RmsAgc  # reuse the shared bit sink + RMS AGC

_log = logging.getLogger("gnuradio_hirate")


def _coherent_bit_chain(tb, src, sample_rate, symbol_rate, constel, *, excess_bw=0.35):
    """Shared coherent linear-modulation RX: LPF → RMS AGC → FLL → RRC matched filter → symbol
    sync → constellation receiver → unpack. Returns the bit sink. Same structure as the PSK chain,
    generalized to any ``constellation`` (QAM/APSK/PSK)."""
    sps = sample_rate / symbol_rate
    lpf = None
    if 2.0 * symbol_rate < sample_rate / 2.0:
        lpf = gr_filter.fir_filter_ccf(
            1, gr_filter.firdes.low_pass(1.0, sample_rate, 2.0 * symbol_rate, 0.2 * symbol_rate))
    agc = _RmsAgc(2e-2 / sps, 1.0, cplx=True)
    fll = digital.fll_band_edge_cc(sps, excess_bw, 100, 6.283185 * 25.0 / sample_rate)
    ntaps = int(11 * sps) | 1
    rrc = gr_filter.fir_filter_ccf(
        1, gr_filter.firdes.root_raised_cosine(1.0, sample_rate, symbol_rate, excess_bw, ntaps))
    # Gardner, not M&M: timing recovery sits BEFORE the carrier loop here, and M&M is
    # decision-directed (degrades under residual rotation); Gardner is rotation-invariant.
    sync = digital.symbol_sync_cc(
        digital.TED_GARDNER, sps, 0.045, 1.0, 1.0, 0.05, 1,
        constel, digital.IR_MMSE_8TAP, 128, [])
    # constellation_receiver_cb does joint carrier (2nd-order loop) + constellation decode.
    receiver = digital.constellation_receiver_cb(constel, 0.06, -0.25, 0.25)
    unpack = blocks.unpack_k_bits_bb(constel.bits_per_symbol())
    sink = _BitSink()
    chain = [src]
    if lpf is not None:
        chain.append(lpf)
    chain += [agc, fll, rrc, sync, receiver]
    if constel.apply_pre_diff_code():
        # RX applies the INVERSE of the pre-diff code to the decoded symbol indices — GR's
        # generic_demod does map_bb(mod_codes.invert_code(...)); the forward code is TX-side.
        # Dead today (apply_pre_diff_code() is False for the constellations used) but must be
        # the inverse if a pre-diff-coded constellation lands.
        chain.append(digital.map_bb(digital.mod_codes.invert_code(constel.pre_diff_code())))
    chain += [unpack, sink]
    tb.connect(*chain)
    return sink


def connect_qam_demod(tb, src, sample_rate, symbol_rate, *, order=16):
    """M-QAM demod (order 16/64/256 — square constellations only; GR 3.10's gray-coded
    constructor RAISES for the cross constellations 32/128, which the caller's build guard
    treats as build-failed/record-only). ``mod_code=GRAY_CODE`` is explicit — the constructor
    default is NO_CODE (natural binary), which would scramble the bit assignment."""
    constel = digital.qam.qam_constellation(
        order, differential=False, mod_code=digital.mod_codes.GRAY_CODE).base()
    return _coherent_bit_chain(tb, src, sample_rate, symbol_rate, constel)


def connect_apsk_demod(tb, src, sample_rate, symbol_rate, *, order=16):
    """M-APSK demod (16/32) — the DVB-S2 amplitude-phase constellation. Uses GNU Radio's APSK
    constellation points; for full DVB-S2 framing use :func:`connect_dvbs2_demod` instead."""
    ctor = getattr(digital, f"constellation_{order}apsk", None)
    if ctor is None:
        _log.info("hirate: no constellation_%dapsk in this GNU Radio; APSK build-pending", order)
        return None
    return _coherent_bit_chain(tb, src, sample_rate, symbol_rate, ctor().base())


def connect_ofdm_demod(tb, src, sample_rate):
    """OFDM RX — BUILD-PENDING. ``digital.ofdm_rx`` outputs a tagged BYTE STREAM (not a PDU),
    which the ``_BitSink``-based engine contract cannot drain yet; returning the raw block would
    leave an unconnected stream port and abort ``tb.start()`` (costing the recording). Until the
    byte-stream→frame adapter is built and bench-validated, report unsupported."""
    _log.info("hirate: OFDM RX adapter build-pending; skipping (record-only for this pass)")
    return None


def connect_dvbs2_demod(tb, src, sample_rate, symbol_rate):
    """DVB-S2/S2X RX via the ``gr-dvbs2rx`` OOT (GPLv3). Returns ``None`` if the dep is absent."""
    try:
        import dvbs2rx  # noqa: PLC0415 — optional OOT dependency
    except Exception as e:  # noqa: BLE001
        _log.info("hirate: gr-dvbs2rx not installed; DVB-S2 build-pending (%s)", e)
        return None
    _log.info("hirate: gr-dvbs2rx present; construct dvbs2_rx on the bench (params per bird): %r",
              dvbs2rx)
    return None  # the dvbs2rx receive chain is assembled per-stream on the bench (MODCOD/rolloff)


def connect_analog_demod(tb, src, sample_rate, kind):
    """Tier-3 analog voice demod — BUILD-PENDING as an engine path. The analog blocks
    (``nbfm_rx``/``wfm_rcv``/``am_demod_cf``) emit an audio-rate FLOAT stream, not the hard bits
    the ``_BitSink`` engine contract drains; returning such a block with its output unconnected
    would abort ``tb.start()`` and cost the recording. Analog voice already has its dedicated
    spawnable app (``amateur_fm_narrowband_rx.py``); route analog missions there. Until an
    audio-sink adapter is built for THIS engine, report unsupported (record-only)."""
    _log.info("hirate: analog %s demod in the satellites engine is build-pending; "
              "use the dedicated FM app (record-only for this pass)", kind)
    return None


def build_tier2_mod(spec, tb, src, sample_rate, symbol_rate):
    """Tier-2 TX modulator: QAM/APSK via ``digital.generic_mod`` (``constellation_modulator`` is
    a GRC-only wrapper, NOT a Python symbol), OFDM via ``ofdm_tx``, DVB-S2 via ``gr-dvbs2rx``.
    Input contract: PACKED bytes (generic_mod unpacks internally). Bench-pending like the demods."""
    sps = max(2, int(round(sample_rate / max(symbol_rate, 1.0))))
    if spec.family == "qam":
        constel = digital.qam.qam_constellation(
            spec.order, differential=False, mod_code=digital.mod_codes.GRAY_CODE).base()
        return digital.generic_mod(
            constellation=constel, differential=False, samples_per_symbol=sps)
    if spec.family == "ofdm":
        return digital.ofdm_tx(fft_len=64, cp_len=16, packet_length_tag_key="len")
    if spec.family == "apsk":
        ctor = getattr(digital, f"constellation_{spec.order}apsk", None)
        return None if ctor is None else digital.generic_mod(
            constellation=ctor().base(), differential=False, samples_per_symbol=sps)
    _log.info("hirate: %s TX build-pending (gr-dvbs2rx for DVB-S2)", spec.kind)
    return None

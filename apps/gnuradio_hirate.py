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

from gnuradio import analog, blocks, digital
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
    sync = digital.symbol_sync_cc(
        digital.TED_MUELLER_AND_MULLER, sps, 0.045, 1.0, 1.0, 0.05, 1,
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
        chain.append(digital.map_bb(constel.pre_diff_code()))
    chain += [unpack, sink]
    tb.connect(*chain)
    return sink


def connect_qam_demod(tb, src, sample_rate, symbol_rate, *, order=16):
    """M-QAM demod (order 16/32/64/128/256) via the gray-coded rectangular QAM constellation."""
    constel = digital.qam.qam_constellation(order, differential=False).base()
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
    """OFDM RX via ``digital.ofdm_rx`` (default 64-carrier / 16-CP profile). The packet payload is
    emitted as a message PDU; wiring that to the bit/frame sink is confirmed on the bench."""
    rx = digital.ofdm_rx(
        fft_len=64, cp_len=16, packet_length_tag_key="len",
        occupied_carriers=(list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0))
                           + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),),
        pilot_carriers=((-21, -7, 7, 21),), pilot_symbols=((1, 1, 1, -1),),
        sync_word1=None, sync_word2=None, bps_header=1, bps_payload=1)
    tb.connect(src, rx)
    _log.info("hirate: OFDM ofdm_rx built; PDU→sink wiring bench-pending")
    return rx  # PDU-producing block; caller taps its message port on the bench


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
    """Tier-3 analog voice/data demod → real float output. NBFM via ``analog.nbfm_rx`` (or a
    quadrature demod), WFM via ``analog.wfm_rcv``, AM via ``analog.am_demod_cf``. Returns the
    terminal block (an audio-rate float stream); the caller records/decodes it on the bench."""
    if kind == "am":
        demod = analog.am_demod_cf(channel_rate=int(sample_rate), audio_decim=1,
                                   audio_pass=5000, audio_stop=5500)
        tb.connect(src, demod)
        return demod
    if kind == "wfm":
        demod = analog.wfm_rcv(quad_rate=int(sample_rate), audio_decimation=1)
        tb.connect(src, demod)
        return demod
    quad = analog.quadrature_demod_cf(sample_rate / (2.0 * 3.141592653589793 * 5000.0))
    tb.connect(src, quad)  # NBFM: quadrature discriminator (audio filtering added on the bench)
    _log.info("hirate: NBFM quadrature demod built; audio LPF/de-emphasis bench-tuned")
    return quad


def build_tier2_mod(spec, tb, src, sample_rate, symbol_rate):
    """Tier-2 TX modulator: QAM via ``constellation_modulator``, OFDM via ``ofdm_tx``, DVB-S2 via
    ``gr-dvbs2rx``. Bench-pending like the demods."""
    sps = max(2, int(round(sample_rate / max(symbol_rate, 1.0))))
    if spec.family == "qam":
        constel = digital.qam.qam_constellation(spec.order, differential=False).base()
        return digital.constellation_modulator(constel, differential=False, samples_per_symbol=sps)
    if spec.family == "ofdm":
        return digital.ofdm_tx(fft_len=64, cp_len=16, packet_length_tag_key="len")
    if spec.family == "apsk":
        ctor = getattr(digital, f"constellation_{spec.order}apsk", None)
        return None if ctor is None else digital.constellation_modulator(
            ctor().base(), differential=False, samples_per_symbol=sps)
    _log.info("hirate: %s TX build-pending (gr-dvbs2rx for DVB-S2)", spec.kind)
    return None

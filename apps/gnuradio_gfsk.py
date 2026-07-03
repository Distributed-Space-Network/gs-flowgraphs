"""GNU Radio front-end for the 2-GFSK / AX.25 flowgraphs (bench engine).

This module is imported ONLY when ``--engine gnuradio`` is selected, so it may
import ``gnuradio`` at module load (the ``dsp`` engine and the pytest suite never
touch it). It implements just the IQ<->bits physical front-end in GNU Radio; the
scrambler/NRZI/HDLC/AX.25 protocol layer stays in the shared, unit-tested
``gfsk_ax25`` library, so both engines decode identically.

RX:  SoapySDR source -> quadrature demod -> Gardner symbol sync -> binary
     slicer -> bit sink (drained by the app and fed to ``framing.decode``).
TX:  ``framing.encode`` bits -> GFSK mod -> SoapySDR sink.

Status: bench-pending. Verify on a Linux box with ``gnuradio`` + ``gr-soapy``
(SoapyLoopback for hardware-free dev), as in the README's NBFM recipe. The loop
gains / sensitivity below are starting points to tune against real captures.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import logging
import math
import queue

import numpy as np
from _recorder import PassRecorder
from _soapy import (
    apply_corrections,
    auto_lo_offset,
    capture_plan,
    configure_soapy_source,
    make_decimator,
    make_sink,
    make_source,
    merge_sdr_params,
    retune_source,
    sdr_env,
    tune_source,
)
from gnuradio import analog, blocks, digital, gr
from gnuradio import filter as gr_filter

from gfsk_ax25 import ax25, endurosat, framing

_log = logging.getLogger("gnuradio_gfsk")


class _BitSink(gr.sync_block):
    """Collects unpacked hard bits (one 0/1 per byte) into a thread-safe queue."""

    def __init__(self) -> None:
        gr.sync_block.__init__(self, name="bit_sink", in_sig=[np.uint8], out_sig=None)
        self._q: queue.Queue[np.ndarray] = queue.Queue()

    def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
        self._q.put(np.array(input_items[0], dtype=np.uint8))
        return len(input_items[0])

    def drain(self) -> np.ndarray:
        out: list[np.ndarray] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(out) if out else np.empty(0, dtype=np.uint8)


class _RmsAgc(gr.hier_block2):
    """Divide a signal by its running RMS (normalize to ``reference``). Faithful port of
    gr-satellites' ``rms_agc`` (GPLv3) — an RMS-normalizing AGC, not a peak/feedback one, so
    it does not ring on a fading pass. ``cplx`` selects complex (PSK) vs float (FSK) I/O."""

    def __init__(self, alpha: float, reference: float = 1.0, *, cplx: bool) -> None:
        size = gr.sizeof_gr_complex if cplx else gr.sizeof_float
        gr.hier_block2.__init__(
            self, "rms_agc", gr.io_signature(1, 1, size), gr.io_signature(1, 1, size))
        rms = blocks.rms_cf(alpha) if cplx else blocks.rms_ff(alpha)
        scale = blocks.multiply_const_ff(1.0 / reference)
        floor = blocks.add_const_ff(1e-19)  # avoid divide-by-zero on silence
        div = blocks.divide_cc(1) if cplx else blocks.divide_ff(1)
        if cplx:
            self.connect(self, rms, scale, floor, blocks.float_to_complex(1), (div, 1))
        else:
            self.connect(self, rms, scale, floor, (div, 1))
        self.connect(self, (div, 0))
        self.connect(div, self)


class _RxContext:
    def __init__(
        self,
        tb: gr.top_block,
        src,
        sink: _BitSink,
        center_hz: float,
        recorder=None,
        lo_offset_hz: float = 0.0,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._recorder = recorder

    def start(self) -> None:
        self.tb.start()

    def stop(self) -> None:
        # Just stop the graph; views are derived post-pass by gs-client (iq_views on the
        # on-disk cf32), so a slow/hung gr-soapy teardown can't cost the recording/views.
        self.tb.stop()
        self.tb.wait()

    def wait(self) -> None:
        self.tb.wait()

    def drain_bits(self) -> np.ndarray:
        return self._sink.drain()

    def set_doppler(self, offset_hz: float) -> None:
        retune_source(self.src, self._center, self._lo_offset, offset_hz)


def connect_gfsk_demod(
    tb, src, sample_rate: float, profile: endurosat.LinkProfile, *,
    decimate: bool, sdr_rate: float, dc_block: bool = True, channel_bw_hz: float | None = None,
) -> tuple[_BitSink, object]:
    """Connect a 2-GFSK/FSK/GMSK/MSK demod chain onto ``src`` and return ``(bit_sink, soft_tap)``:
    the hard-bit sink (``drain()`` → bits, fed to our numpy deframers) and the FLOAT soft-symbol
    tap (post clock-recovery, PRE-slicer) that gr-satellites deframer components consume.

    Stock-GNU-Radio port of SatNOGS' proven ``satnogs_fsk.py`` (GPL-3.0) demod tail — NEVER the
    AGPL gr-satnogs hier-blocks, and never the monolithic ``gr_satellites_flowgraph`` (which fed
    raw IQ, did demod+deframe+resample internally, and buffer-deadlocked while starving the
    recorder — docs/12). Chain:

        [SDR→channel decimator] → FLL band-edge → LPF(0.625·baud, decimate to ~2 sps)
        → quad demod(1.2) → [dc_blocker_ff(1024)] → M&M clock recovery(2 sps) ──soft──►
        → binary slicer → _BitSink.

    Doppler is applied UPSTREAM at the source (the LO-offset rotator, gs-orbitd), NOT here. The
    bit sink is a terminal queue and the deframers on the soft tap emit messages, so nothing here
    can backpressure the recorder that taps the same channel stream. ``channel_bw_hz`` is accepted
    for call-compat but IGNORED — SatNOGS sizes the pre-discriminator LPF from the baud, never a
    provisioned width. ``dc_block`` gates the (SatNOGS-default-on) discriminator dc_blocker_ff."""
    _ = channel_bw_hz  # accepted for call-compat; SatNOGS sizes the LPF from baud, not bandwidth
    baud = float(profile.symbol_rate_hz)
    sps_in = sample_rate / baud                        # samples/symbol at the channel rate
    # Decimate the demod stream to ~2 sps in the LPF, exactly like SatNOGS' fir_filter_ccf with
    # decimation = (baud·decim)//2. FLOOR (not round): flooring the decimation guarantees
    # out_sps = sps_in/floor(sps_in/2) ≥ 2.0 for every baud, so M&M never runs sub-Nyquist. (For
    # the common 48 kHz bauds — 40/20/10/5 sps — floor and round agree; they diverge only for odd
    # sps_in, where round-half-even could pick a decim that drops out_sps below 2.) Clamp ≥1.
    lpf_decim = max(1, int(sps_in / 2.0))
    out_sps = sps_in / lpf_decim                        # ≥ 2.0

    # Construct EVERY block first, connect LAST — a constructor raise leaves the graph untouched.
    blocks_list: list = []
    if decimate:                                        # SDR capture-rate → channel-rate first
        blocks_list.append(make_decimator(sdr_rate, float(sample_rate)))
    # 1) FLL band-edge: coarse carrier-frequency lock (SatNOGS: sps, rolloff 0.5, size 2·sps+1,
    #    bw = 2π/sps/100). Residual offset after the upstream Doppler/LO rotator is small.
    fll_size = int(sps_in * 2.0 + 1.0) | 1             # odd
    blocks_list.append(digital.fll_band_edge_cc(
        sps_in, 0.5, fll_size, 2.0 * math.pi / sps_in / 100.0))
    # 2) Pre-discriminator LPF sized from the BAUD (0.625·baud cutoff, baud/8 transition),
    #    DECIMATING to ~2 sps. firdes default window = Hamming (== SatNOGS' WIN_HAMMING); omit the
    #    window arg so the taps match without depending on the 3.8-vs-3.10 window-enum module path.
    blocks_list.append(gr_filter.fir_filter_ccf(
        lpf_decim, gr_filter.firdes.low_pass(1.0, sample_rate, 0.625 * baud, baud / 8.0)))
    # 3) Quadrature (frequency) discriminator — SatNOGS' fixed empirical gain 1.2.
    blocks_list.append(analog.quadrature_demod_cf(1.2))
    # 4) DC blocker on the discriminator output — removes the residual carrier bias a bare slicer
    #    mis-slices. SatNOGS uses a fixed 1024-tap long-form blocker; skipped (dc_block=False) for
    #    short cubesat bursts where its settle would eat the frame start.
    if dc_block:
        blocks_list.append(gr_filter.dc_blocker_ff(1024, True))
    # 5) Mueller & Müller symbol timing recovery at ~2 sps — SatNOGS' EXACT constants
    #    (satnogs_fsk.py: clock_recovery_mm_ff(2, 2π/100, 0.5, 0.5/8, 0.01)). This float output is
    #    the SOFT-SYMBOL TAP the gr-satellites deframers consume (fanned out by the caller).
    soft = digital.clock_recovery_mm_ff(
        out_sps, 2.0 * math.pi / 100.0, 0.5, 0.5 / 8.0, 0.01)
    # 6) Slice to hard bits (one 0/1 per byte) for our numpy deframers.
    slicer = digital.binary_slicer_fb()
    sink = _BitSink()
    tb.connect(src, *blocks_list, soft, slicer, sink)  # single connect; a raise above leaves clean
    return sink, soft


def connect_psk_demod(
    tb, src, sample_rate: float, symbol_rate: float, *, order: int = 2,
    differential: bool = True, offset: bool = False, excess_bw: float = 0.35,
) -> _BitSink:
    """Connect a coherent PSK demod chain onto ``src`` (already at the channel rate) and
    return the bit sink. ``order`` selects the constellation: 2 = BPSK, 4 = QPSK, 8 = 8-PSK.

    AGC → RRC matched filter → Gardner symbol sync → Costas carrier recovery →
    constellation decode → [differential decode] → (pre-diff Gray map) → unpack to hard
    bits. Gardner (not Mueller&Müller) because timing recovery runs BEFORE carrier recovery
    here: M&M is decision-directed and degrades under the residual rotation the FLL leaves;
    Gardner is rotation-invariant (same choice as gr-satellites' bpsk_demodulator).

    ``differential`` toggles the DxPSK differential decoder. For order 2 it resolves the
    Costas phase ambiguity (correct for DBPSK; pass False for true coherent BPSK). For
    order > 2 the index-domain diff decode does NOT implement the DQPSK phase mapping
    (gray-baked indices are not circular phase positions), so it is SKIPPED with a warning
    until the psk_demod-equivalent mapping is bench-validated. ``offset`` marks OQPSK —
    the half-symbol I/Q stagger demod is bench build-pending, so it is currently treated
    as QPSK (logged); confirm against a real OQPSK bird before relying on it."""
    sps = sample_rate / symbol_rate
    if order == 8:
        constel = digital.constellation_8psk().base()
    elif order == 4:
        constel = digital.constellation_qpsk().base()
    else:
        constel = digital.constellation_bpsk().base()
    if offset:  # OQPSK: dedicated staggered-demod not built yet — best-effort as QPSK.
        _log.info("connect_psk_demod: OQPSK offset handling bench-pending; treating as QPSK")
    # Front LPF (~2x baud) limits noise into the loops (mirrors gr-satellites' xlating filter).
    lpf = None
    lpf_cut = 2.0 * symbol_rate
    if lpf_cut < sample_rate / 2.0:
        lpf = gr_filter.fir_filter_ccf(
            1, gr_filter.firdes.low_pass(1.0, sample_rate, lpf_cut, 0.2 * symbol_rate))
    agc = _RmsAgc(2e-2 / sps, 1.0, cplx=True)  # RMS AGC, ~50-symbol time constant (gr-satellites)
    # FLL band-edge: COARSE carrier-frequency recovery. The Costas loop tracks phase + only a
    # small frequency offset; the FLL acquires the residual ~kHz offset first (mirrors
    # gr-satellites' bpsk_demodulator — the PSK analog of the FSK DC-blocker fix).
    fll = digital.fll_band_edge_cc(sps, excess_bw, 100, 2.0 * math.pi * 25.0 / sample_rate)
    ntaps = int(11 * sps) | 1  # odd
    rrc = gr_filter.fir_filter_ccf(
        1, gr_filter.firdes.root_raised_cosine(1.0, sample_rate, symbol_rate, excess_bw, ntaps)
    )
    sync = digital.symbol_sync_cc(
        digital.TED_GARDNER, sps, 0.045, 1.0, 1.0, 0.05, 1,
        constel, digital.IR_MMSE_8TAP, 128, [],
    )
    costas = digital.costas_loop_cc(0.04, order)
    decoder = digital.constellation_decoder_cb(constel)
    unpack = blocks.unpack_k_bits_bb(constel.bits_per_symbol())  # symbol index → hard bits
    sink = _BitSink()
    # constellation_decoder emits symbol INDICES; the inverted pre-diff code (below) undoes
    # the constellation's Gray/bit assignment before unpacking (generic_demod does the same).
    chain = [src]
    if lpf is not None:
        chain.append(lpf)
    chain += [agc, fll, rrc, sync, costas, decoder]
    if differential and order == 2:  # DBPSK: resolves the Costas 180° ambiguity
        chain.append(digital.diff_decoder_bb(order))
    elif differential:
        # Index-domain diff decode is NOT the DQPSK/D8PSK phase mapping (see docstring);
        # decode coherently and let the deframer's own polarity tolerance do what it can.
        _log.warning("connect_psk_demod: differential decode for order %d bench-pending; "
                     "decoding coherently", order)
    if constel.apply_pre_diff_code():
        # RX applies the INVERSE of the constellation's pre-diff code to the decoded symbol
        # indices — GR's generic_demod does map_bb(mod_codes.invert_code(...)); the forward
        # code is TX-side (generic_mod). Dead today (apply_pre_diff_code() is False for every
        # constellation used here) but must be the inverse if a pre-diff-coded one lands.
        chain.append(digital.map_bb(digital.mod_codes.invert_code(constel.pre_diff_code())))
    chain += [unpack, sink]
    tb.connect(*chain)
    return sink


def connect_afsk_demod(
    tb, src, sample_rate: float, *, baud: float = 1200.0,
    mark_hz: float = 1200.0, space_hz: float = 2200.0,
) -> _BitSink:
    """Connect a Bell-202 AFSK demod (mark/space audio tones FM-carried) onto ``src`` and
    return the bit sink. FM-demod the IQ to audio, frequency-shift the audio tone-centre to
    baseband, then run the SatNOGS FSK demod on it — so AFSK inherits the FLL / baud-LPF /
    dc-blocker / M&M chain for free. NRZI/HDLC is handled downstream by the deframer. The
    float soft-symbol tap is discarded here (AFSK gr-satellites deframers are bench-pending —
    AFSK is rare on 401 MHz UHF)."""
    af_carrier = (mark_hz + space_hz) / 2.0      # tone centre (Bell-202: 1700 Hz)
    af_dev = abs(space_hz - mark_hz) / 2.0       # tone half-spacing (500 Hz)
    # 1) FM-demod the IQ to real audio (scaling handled by the xlating + FSK demod).
    audio = analog.quadrature_demod_cf(1.0)
    # 2) Shift the tone-centre to baseband (real → complex) and limit to ~2x the deviation.
    xlate = gr_filter.freq_xlating_fir_filter_fcf(
        1, gr_filter.firdes.low_pass(1.0, sample_rate, 2.0 * af_dev, 0.1 * af_dev),
        af_carrier, sample_rate)
    # 3) Reuse the SatNOGS FSK demod on the complex baseband (its dc_blocker_ff absorbs any
    #    residual tone-centre error; the LPF is sized from the baud, so mod_index is unused here).
    profile = endurosat.LinkProfile(symbol_rate_hz=baud, mod_index=2.0 * af_dev / baud)
    # Construct + wire the FSK chain BEFORE connecting anything to ``src`` (null_sink rule,
    # docs/10 A4/A5 class): if an FSK-chain constructor raises (absurd baud → sps < 1), the
    # caller catches it — but an already-connected src→audio→xlate tap would leave xlate's
    # output dangling, abort tb.start(), and cost the IQ RECORDING. connect_gfsk_demod itself
    # builds every block first and connects last, so a raise leaves the graph untouched.
    sink, _soft = connect_gfsk_demod(
        tb, xlate, sample_rate, profile, decimate=False, sdr_rate=sample_rate)
    tb.connect(src, audio, xlate)
    return sink


def build_rx_top_block(
    args, profile: endurosat.LinkProfile, sample_rate: float, params: dict | None = None
) -> _RxContext:
    env = sdr_env()  # station-wide GS_SDR_* (antenna/gain/lo-offset/ppm/dc-removal/rate)
    # Capture at the SDR's supported rate (XTRX floor ~2.1 Msps) and decimate to the
    # channel rate; the demod chain runs at ``sample_rate``.
    sdr_rate, decimate = capture_plan(env["capture_rate_hz"], float(sample_rate))
    # AUTO LO offset → DC spike dodged off the bird (no per-pass config); see _soapy.
    lo = auto_lo_offset(sdr_rate, float(sample_rate), env["lo_offset_hz"])
    tb = gr.top_block("cubesat_gfsk_ax25_rx_gr")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo)  # LO offset → DC spike off-signal
    configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])
    # Decimate to the channel rate ONCE; the demod chain and the recorder both tap it, so
    # the capture is the narrow channel (~MB/min), not the multi-GB wideband SDR stream.
    chan = src
    if decimate:
        chan = make_decimator(sdr_rate, float(sample_rate))
        tb.connect(src, chan)
    # dc_block=False: EnduroSat frames are short bursts; the DC blocker's settling would eat the
    # start of a frame. (The generic satellites fallback keeps it on by default.) The soft tap is
    # discarded — the EnduroSat path deframes AirMAC via our numpy codec off the hard-bit sink.
    sink, _soft = connect_gfsk_demod(
        tb, chan, float(sample_rate), profile,
        decimate=False, sdr_rate=float(sample_rate), dc_block=False,
    )
    recorder = PassRecorder.maybe_start(args, tb, chan, sample_rate_hz=float(sample_rate))
    return _RxContext(tb, src, sink, float(args.center_freq_hz), recorder, lo_offset_hz=lo)


def transmit_gnuradio(args, params: dict[str, object], profile: endurosat.LinkProfile) -> None:
    """Build the AX.25 frame, GFSK-modulate via GNU Radio, and key it out the SDR."""
    import base64

    sample_rate = float(args.sample_rate or 96_000)
    sps = int(round(sample_rate / profile.symbol_rate_hz))
    payload = b""
    b64 = params.get("uplink_b64")
    if isinstance(b64, str) and b64:
        payload = base64.b64decode(b64)
    body = ax25.encode_ui(
        dest=str(params.get("dest", "CQ")),
        src=str(params.get("src", "DSN")),
        info=payload[: endurosat.AX25_INFO_MAX_BYTES],
    )
    bits = framing.encode(body, scramble=profile.scramble, nrzi=profile.nrzi)

    sensitivity = math.pi * profile.mod_index / sps  # rad/sample at full deflection
    tb = gr.top_block("cubesat_gfsk_ax25_tx_gr")
    # gfsk_mod defaults to do_unpack=True: it expects PACKED bytes and unpacks internally.
    # Feeding unpacked 0/1 bits would make each bit 8 symbols (8x-slow, garbled waveform).
    # np.packbits pads the tail with zero bits — harmless after the closing HDLC flags.
    src = blocks.vector_source_b(np.packbits(bits.astype(np.uint8)).tolist(), repeat=False)
    mod = digital.gfsk_mod(samples_per_symbol=sps, sensitivity=sensitivity, bt=profile.bt)
    sink = make_sink(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    sink.set_sample_rate(0, sample_rate)
    sink.set_frequency(0, float(args.center_freq_hz))  # TX: no LO offset (mod at baseband 0)
    configure_soapy_source(sink, merge_sdr_params(params))  # TX antenna + gain (PA drive)
    # Analog TX filter ≈ the SDR sample rate, NOT the narrow channel width — the latter is
    # below the device filter floor (~0.8 MHz on the XTRX) and would break the path (same
    # treatment as the FM TX app). TX levels are BENCH-PENDING.
    sink.set_bandwidth(0, sample_rate)
    tb.connect(src, mod, sink)
    tb.run()


__all__ = ["build_rx_top_block", "transmit_gnuradio"]

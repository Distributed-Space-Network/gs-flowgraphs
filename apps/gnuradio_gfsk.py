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

import numpy as np
from _recorder import PassRecorder
from _soapy import (
    DEFAULT_LO_OFFSET_HZ,
    apply_corrections,
    auto_lo_offset,
    capture_plan,
    configure_soapy_source,
    lo_phase_inc,
    make_decimator,
    make_lo_rotator,
    make_source,
    merge_sdr_params,
    open_analog_bandwidth,
    sdr_env,
    tune_below,
)
from gnuradio import analog, blocks, digital, gr
from gnuradio import filter as gr_filter
from native_framing.runtime_queue import BoundedQueue, QueueStats, require_lossless

from gfsk_ax25 import endurosat

_log = logging.getLogger("gnuradio_gfsk")

_SYMBOL_QUEUE_CAPACITY_CHUNKS = 256
_SYMBOL_QUEUE_CAPACITY_SYMBOLS = 1 << 20


def _drain_symbol_queue(
    handoff: BoundedQueue[np.ndarray], *, dtype: type[np.generic], label: str
) -> np.ndarray:
    stats = handoff.stats()
    require_lossless(stats, label=label, unit_name="symbols")
    chunks = handoff.drain()
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)


class _BitSink(gr.sync_block):
    """Collects unpacked hard bits (one 0/1 per byte) into a thread-safe queue."""

    def __init__(self) -> None:
        gr.sync_block.__init__(self, name="bit_sink", in_sig=[np.uint8], out_sig=None)
        self._q = BoundedQueue[np.ndarray](
            capacity_items=_SYMBOL_QUEUE_CAPACITY_CHUNKS,
            capacity_units=_SYMBOL_QUEUE_CAPACITY_SYMBOLS,
        )

    def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
        chunk = np.array(input_items[0], dtype=np.uint8)
        self._q.offer(chunk, units=int(chunk.size))
        return len(input_items[0])

    def drain(self) -> np.ndarray:
        return _drain_symbol_queue(self._q, dtype=np.uint8, label="hard-bit")

    @property
    def queue_stats(self) -> QueueStats:
        return self._q.stats()


class SoftSymbolSink(gr.sync_block):
    """Collect post-timing-recovery float symbols for native soft-input deframers."""

    def __init__(self) -> None:
        gr.sync_block.__init__(self, name="soft_symbol_sink", in_sig=[np.float32], out_sig=None)
        self._q = BoundedQueue[np.ndarray](
            capacity_items=_SYMBOL_QUEUE_CAPACITY_CHUNKS,
            capacity_units=_SYMBOL_QUEUE_CAPACITY_SYMBOLS,
        )

    def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
        chunk = np.array(input_items[0], dtype=np.float64)
        self._q.offer(chunk, units=int(chunk.size))
        return len(input_items[0])

    def drain(self) -> np.ndarray:
        return _drain_symbol_queue(self._q, dtype=np.float64, label="soft-symbol")

    @property
    def queue_stats(self) -> QueueStats:
        return self._q.stats()


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
        *,
        rotator=None,
        sdr_rate_hz: float = 0.0,
        sdr_applied: dict | None = None,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._rotator = rotator        # software LO+Doppler NCO (Phase 1); None ⇒ no retune
        self._sdr_rate = sdr_rate_hz
        self.recorder = recorder       # public: the app's R-11 first-sample probe reads it
        self.sdr_applied = dict(sdr_applied or {})  # R-21: what configure/corrections applied

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
        # Software NCO retune (gs-orbitd ephemeris → rotator), NOT a hardware LO retune (docs/12
        # Phase 1): shifts the +lo_offset+doppler carrier to DC, no PLL settle glitch.
        if self._rotator is not None and self._sdr_rate:
            self._rotator.set_phase_inc(
                lo_phase_inc(self._sdr_rate, self._lo_offset, offset_hz))


def connect_gfsk_demod(
    tb, src, sample_rate: float, profile: endurosat.LinkProfile, *,
    decimate: bool, sdr_rate: float, dc_block: bool = True, channel_bw_hz: float | None = None,
    collect_hard: bool = True,
) -> tuple[_BitSink | None, object]:
    """Connect a 2-GFSK/FSK/GMSK/MSK demod chain onto ``src`` and return ``(bit_sink, soft_tap)``:
    the hard-bit sink (``drain()`` → bits, fed to our numpy deframers) and the FLOAT soft-symbol
    tap (post clock-recovery, PRE-slicer) that gr-satellites deframer components consume.

    Uses the pinned gr-satellites ``fsk_demodulator`` component (GPL-3.0) directly — never the
    monolithic ``gr_satellites_flowgraph`` (which fed raw IQ, did demod+deframe+resample internally,
    and buffer-deadlocked while starving the recorder — docs/12). Its chain is:

        [SDR→channel decimator] → Carson-width IQ LPF → deviation-normalized quad demod
        → one-symbol square-pulse filter/decimator → [32-symbol DC blocker]
        → Gardner symbol sync ──soft──► binary slicer → _BitSink.

    Doppler is applied UPSTREAM at the source (the LO-offset rotator, gs-orbitd), NOT here. The
    bit sink is a terminal queue and the deframers on the soft tap emit messages, so nothing here
    can backpressure the recorder that taps the same channel stream. ``channel_bw_hz`` is accepted
    for call compatibility but does not override the modem's deviation-derived Carson width.
    ``dc_block`` gates the pinned 32-symbol discriminator DC blocker.
    When ``collect_hard`` is false, the slicer terminates in a GNU Radio null sink instead of an
    undrained Python queue; the soft tap remains available to its actual consumer.
    """
    _ = channel_bw_hz
    baud = float(profile.symbol_rate_hz)
    deviation = float(profile.mod_index) * baud / 2.0
    if not math.isfinite(deviation) or deviation <= 0.0:
        raise ValueError("FSK deviation must be a finite positive number")
    if sample_rate / baud <= 1.0:
        raise ValueError("FSK timing recovery requires more than one sample per symbol")

    # Import the proven modem component, not a local approximation. The containing application
    # already requires the pinned gr-satellites runtime; native USP and the upstream USP deframer
    # therefore receive the same recovered symbols as gr-satellites itself.
    from satellites.components.demodulators import (  # noqa: PLC0415 - station dependency
        fsk_demodulator,
    )
    soft = fsk_demodulator(
        baud,
        float(sample_rate),
        iq=True,
        deviation=deviation,
        dc_block=dc_block,
        options=None,
    )

    # Slice to hard bits (one 0/1 per byte) only for local hard-input deframers.
    slicer = digital.binary_slicer_fb()
    hard_consumer = _BitSink() if collect_hard else blocks.null_sink(gr.sizeof_char)
    prefix = [make_decimator(sdr_rate, float(sample_rate))] if decimate else []
    # Construct every block before connecting so an import/API/constructor failure leaves the
    # graph untouched and the caller can retain recorder-only operation safely.
    tb.connect(src, *prefix, soft, slicer, hard_consumer)
    return (hard_consumer if collect_hard else None), soft


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
    baseband, then run the pinned gr-satellites FSK demodulator on it — so AFSK inherits its
    deviation-aware filter/discriminator, square-pulse filter, DC block, and Gardner timing.
    NRZI/HDLC is handled downstream by the deframer. The
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
    # 3) Reuse the pinned gr-satellites FSK demod on the complex baseband (its dc_blocker_ff
    #    absorbs residual tone-centre error; mod_index declares the Bell-202 tone deviation).
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
    # AUTO LO offset → DC spike dodged off the bird in SOFTWARE (rotator); see _soapy Phase 1.
    lo = auto_lo_offset(sdr_rate, float(sample_rate), env["lo_offset_hz"],
                        default_offset_hz=DEFAULT_LO_OFFSET_HZ)
    tb = gr.top_block("cubesat_gfsk_ax25_rx_gr")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    open_analog_bandwidth(src, sdr_rate)  # widen analog BW so the +lo_offset carrier survives
    tune_below(src, float(args.center_freq_hz), lo)  # LO to center-lo_offset (plain; no BB CORDIC)
    applied = configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    applied.update(apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"]))
    # Software LO+Doppler rotator right after the source (brings the +lo_offset carrier to DC and
    # is the mid-pass Doppler NCO). Then decimate to the channel rate ONCE; the demod chain and
    # the recorder both tap it, so the capture is the narrow channel (~MB/min), not the wideband.
    rotator = make_lo_rotator(sdr_rate, lo, 0.0)
    tb.connect(src, rotator)
    chan = rotator
    if decimate:
        chan = make_decimator(sdr_rate, float(sample_rate))
        tb.connect(rotator, chan)
    # dc_block=False: EnduroSat frames are short bursts; the DC blocker's settling would eat the
    # start of a frame. (The generic satellites fallback keeps it on by default.) The soft tap is
    # discarded — the EnduroSat path deframes AirMAC via our numpy codec off the hard-bit sink.
    sink, _soft = connect_gfsk_demod(
        tb, chan, float(sample_rate), profile,
        decimate=False, sdr_rate=float(sample_rate), dc_block=False,
    )
    recorder = PassRecorder.maybe_start(args, tb, chan, sample_rate_hz=float(sample_rate))
    return _RxContext(
        tb, src, sink, float(args.center_freq_hz), recorder,
        lo_offset_hz=lo, rotator=rotator, sdr_rate_hz=sdr_rate, sdr_applied=applied)


def modulate_gnuradio(frame):
    """GFSK-modulate an ALREADY-FRAMED uplink with GNU Radio and RETURN the baseband IQ.

    It does NOT touch the SDR, and — since R2-43 — it does not decide WHAT to transmit either.

    It used to. It resolved the payload from ``uplink_b64`` and nowhere else (so a file-sourced
    uplink silently became an EMPTY frame), and it always built an AX.25 UI frame — even when the
    pass asked for ``framing=endurosat``, a pair the waveform schema ADVERTISES. Flying that pair
    keyed the PA and radiated a well-formed, correctly-modulated AX.25 frame at a satellite that
    speaks EnduroSat chip packets. It reported success.

    Now it receives an :class:`_uplink_frame.UplinkFrame` — bits, plus the symbol rate / modulation
    index / BT that the chosen framing implies — built by the same code the dsp engine uses. The two
    engines cannot disagree about the frame, because only one of them knows what a frame is.

    (The earlier ``transmit_gnuradio`` also opened its own SDR sink, which made it a second transmit
    path that never emitted ``transmit_started`` — safety stayed in KEYED_READY, the orchestrator's
    immediate de-key only fires from KEYED, and the PA stayed energized until LOS. That is why this
    is a modulator and the caller owns the sink.)
    """
    sps = int(round(frame.sps))
    if abs(frame.sps - sps) > 1e-9:
        msg = (
            f"sample_rate/symbol_rate must be an integer (got {frame.sps:.4f} sps for "
            f"{frame.sample_rate_hz:.0f}/{frame.symbol_rate_hz:.0f}) — refusing to key the PA with "
            f"a resampled frame"
        )
        raise ValueError(msg)
    sensitivity = math.pi * frame.mod_index / sps  # rad/sample at full deflection
    tb = gr.top_block("cubesat_gfsk_mod_gr")
    # gfsk_mod defaults to do_unpack=True: it expects PACKED bytes and unpacks internally.
    # Feeding unpacked 0/1 bits would make each bit 8 symbols (8x-slow, garbled waveform).
    # Round 7: feed UNPACKED bits with do_unpack=False.
    #
    # gfsk_mod defaults to do_unpack=True, which expects PACKED bytes — and np.packbits() pads the
    # last byte with ZERO BITS, so the modulator radiated up to seven invented symbols on the end of
    # a frame whose FCS did not cover them. Round 6 "fixed" that by padding the bitstream with flag
    # bits at the framing layer, but that padding was appended AFTER scrambling and NRZI, so it was
    # not encoded idle fill at all — just raw bits wearing a flag's clothes.
    #
    # The real answer is not to pack at all. One byte per bit, do_unpack=False, exact bit count, no
    # padding anywhere, and the two engines modulate the identical bitstream.
    bits = np.asarray(frame.bits, dtype=np.uint8)
    if bits.size == 0:
        msg = "GNU Radio engine: empty frame — refusing to key the PA"
        raise ValueError(msg)
    src = blocks.vector_source_b(bits.tolist(), repeat=False)
    mod = digital.gfsk_mod(
        samples_per_symbol=sps, sensitivity=sensitivity, bt=frame.bt, do_unpack=False
    )
    snk = blocks.vector_sink_c()
    tb.connect(src, mod, snk)
    tb.run()
    iq = np.asarray(snk.data(), dtype=np.complex64)
    if iq.size == 0:
        msg = "GNU Radio GFSK modulator produced ZERO samples — refusing to key the PA"
        raise RuntimeError(msg)
    return iq


__all__ = ["build_rx_top_block", "modulate_gnuradio"]

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
    tb, src, sample_rate: float, profile: endurosat.LinkProfile, *, decimate: bool, sdr_rate: float
) -> _BitSink:
    """Connect a 2-GFSK demod chain (quad-demod → Gardner symbol sync → binary slicer →
    bit sink) onto ``src``, decimating ``sdr_rate``→``sample_rate`` first when needed.
    Returns the bit sink (``drain()`` → hard bits). Shared by the cubesat GFSK engine
    and the gr-satellites engine's generic fallback for birds gr-satellites can't decode."""
    sps = sample_rate / profile.symbol_rate_hz
    deviation = profile.mod_index * profile.symbol_rate_hz / 2.0
    # Quadrature demod: instantaneous frequency scaled so +/- deviation → ~+/-1.
    quad = analog.quadrature_demod_cf(sample_rate / (2.0 * math.pi * deviation))
    # Gardner symbol timing recovery at the channel symbol rate.
    ted = digital.symbol_sync_ff(
        digital.TED_GARDNER, sps, 0.045, 1.0, 1.0, 0.05, 1,
        digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [],
    )
    slicer = digital.binary_slicer_fb()  # float -> 0/1 bytes
    sink = _BitSink()
    if decimate:  # SDR at capture rate → resample down to the channel rate, then demod
        tb.connect(src, make_decimator(sdr_rate, float(sample_rate)), quad, ted, slicer, sink)
    else:
        tb.connect(src, quad, ted, slicer, sink)
    return sink


def connect_psk_demod(
    tb, src, sample_rate: float, symbol_rate: float, *, order: int = 2, excess_bw: float = 0.35
) -> _BitSink:
    """Connect a coherent PSK demod chain onto ``src`` (already at the channel rate) and
    return the bit sink. ``order`` selects the constellation: 2 = BPSK, 4 = QPSK.

    AGC → RRC matched filter → Mueller&Müller symbol sync → Costas carrier recovery →
    constellation decode → differential decode → (pre-diff Gray map) → unpack to hard
    bits. Mirrors GNU Radio's own ``digital.psk.psk_demod`` so the symbol→bit mapping is
    correct for QPSK (a raw bit-unpack of the symbol index would scramble the Gray code).
    Differential is the safe default for an unknown bird — most cubesat PSK downlinks are
    differentially encoded, so the Costas phase ambiguity doesn't flip the data."""
    sps = sample_rate / symbol_rate
    constel = (digital.constellation_qpsk() if order == 4 else digital.constellation_bpsk()).base()
    agc = analog.agc_cc(1e-3, 1.0, 1.0)
    ntaps = int(11 * sps) | 1  # odd
    rrc = gr_filter.fir_filter_ccf(
        1, gr_filter.firdes.root_raised_cosine(1.0, sample_rate, symbol_rate, excess_bw, ntaps)
    )
    sync = digital.symbol_sync_cc(
        digital.TED_MUELLER_AND_MULLER, sps, 0.045, 1.0, 1.0, 0.05, 1,
        constel, digital.IR_MMSE_8TAP, 128, [],
    )
    costas = digital.costas_loop_cc(0.04, order)
    decoder = digital.constellation_decoder_cb(constel)
    diff = digital.diff_decoder_bb(order)
    unpack = blocks.unpack_k_bits_bb(constel.bits_per_symbol())  # symbol index → hard bits
    sink = _BitSink()
    # constellation_decoder emits symbol INDICES; map_bb(pre_diff_code) applies the
    # constellation's Gray/bit assignment before unpacking (psk_demod does the same).
    chain = [src, agc, rrc, sync, costas, decoder, diff]
    if constel.apply_pre_diff_code():
        chain.append(digital.map_bb(constel.pre_diff_code()))
    chain += [unpack, sink]
    tb.connect(*chain)
    return sink


def connect_afsk_demod(tb, src, sample_rate: float, *, baud: float = 1200.0) -> _BitSink:
    """Connect a Bell-202 AFSK demod (1200/2200 Hz tones FM-carried, ``baud`` symbols)
    onto ``src`` and return the bit sink. FM-demod to audio, then a delay-and-multiply
    discriminator (classic soft-TNC method) turns the mark/space tones into a bipolar
    stream → low-pass → Gardner symbol sync → slicer. NRZI/HDLC is handled downstream by
    the deframer. Bench-pending — validate the discriminator delay against a real
    AFSK capture (AFSK is rare on 401 MHz UHF; included for completeness)."""
    sps = sample_rate / baud
    audio = analog.quadrature_demod_cf(sample_rate / (2.0 * math.pi * 3000.0))  # NBFM → tones
    delay_n = max(1, int(round(sample_rate / 1700.0)))  # ~half period of the tone centre
    delayed = blocks.delay(gr.sizeof_float, delay_n)
    mult = blocks.multiply_ff()
    lpf = gr_filter.fir_filter_fff(1, gr_filter.firdes.low_pass(1.0, sample_rate, baud, baud / 2.0))
    sync = digital.symbol_sync_ff(
        digital.TED_GARDNER, sps, 0.045, 1.0, 1.0, 0.05, 1,
        digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [],
    )
    slicer = digital.binary_slicer_fb()
    sink = _BitSink()
    tb.connect(src, audio)
    tb.connect(audio, (mult, 0))
    tb.connect(audio, delayed, (mult, 1))
    tb.connect(mult, lpf, sync, slicer, sink)
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
    sink = connect_gfsk_demod(
        tb, chan, float(sample_rate), profile, decimate=False, sdr_rate=float(sample_rate)
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
    src = blocks.vector_source_b(bits.astype(np.uint8).tolist(), repeat=False)
    mod = digital.gfsk_mod(samples_per_symbol=sps, sensitivity=sensitivity, bt=profile.bt)
    sink = make_sink(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    sink.set_sample_rate(0, sample_rate)
    sink.set_frequency(0, float(args.center_freq_hz))  # TX: no LO offset (mod at baseband 0)
    configure_soapy_source(sink, merge_sdr_params(params))  # TX antenna + gain (PA drive)
    tb.connect(src, mod, sink)
    tb.run()


__all__ = ["build_rx_top_block", "transmit_gnuradio"]

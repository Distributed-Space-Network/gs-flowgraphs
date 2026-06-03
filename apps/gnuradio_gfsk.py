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
from gnuradio import analog, blocks, digital, gr, soapy

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
    def __init__(self, tb: gr.top_block, src, sink: _BitSink, center_hz: float) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz

    def start(self) -> None:
        self.tb.start()

    def stop(self) -> None:
        self.tb.stop()

    def wait(self) -> None:
        self.tb.wait()

    def drain_bits(self) -> np.ndarray:
        return self._sink.drain()

    def set_doppler(self, offset_hz: float) -> None:
        self.src.set_frequency(0, self._center + offset_hz)


def build_rx_top_block(args, profile: endurosat.LinkProfile, sample_rate: float) -> _RxContext:
    sps = sample_rate / profile.symbol_rate_hz
    deviation = profile.mod_index * profile.symbol_rate_hz / 2.0

    tb = gr.top_block("cubesat_gfsk_ax25_rx_gr")
    src = soapy.source(args.sdr_args, "fc32", 1, "", [""], [""], [""], [""])
    src.set_sample_rate(0, float(sample_rate))
    src.set_frequency(0, float(args.center_freq_hz))

    # Quadrature demod: output is instantaneous frequency scaled so +/- deviation
    # maps to ~+/-1 (gain = fs / (2*pi*deviation)).
    quad = analog.quadrature_demod_cf(sample_rate / (2.0 * math.pi * deviation))

    # Gardner symbol timing recovery at the channel symbol rate.
    ted = digital.symbol_sync_ff(
        digital.TED_GARDNER,
        sps,
        0.045,  # loop bandwidth — tune on the bench
        1.0,
        1.0,
        0.05,
        1,
        digital.constellation_bpsk().base(),
        digital.IR_MMSE_8TAP,
        128,
        [],
    )
    slicer = digital.binary_slicer_fb()  # float -> 0/1 bytes
    sink = _BitSink()
    tb.connect(src, quad, ted, slicer, sink)
    return _RxContext(tb, src, sink, float(args.center_freq_hz))


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
    sink = soapy.sink(args.sdr_args, "fc32", 1, "", [""], [""], [""], [""])
    sink.set_sample_rate(0, sample_rate)
    sink.set_frequency(0, float(args.center_freq_hz))
    tb.connect(src, mod, sink)
    tb.run()


__all__ = ["build_rx_top_block", "transmit_gnuradio"]

"""Multi-mission decode bridge: gr-satellites -> the spawn contract (bench engine).

Makes the ground station "prepared for everything" by delegating to **gr-satellites**
(GPLv3) — the canonical library of public satellite framers/deframers (AX.25,
AX.100, Mobitex, CCSDS, GOMspace, EnduroSat, …) across AFSK/FSK/GFSK/BPSK/GMSK/…,
SatYAML-driven. We integrate it rather than reimplement it (see Document F).

This module is imported ONLY when a pass selects a gr-satellites `satellite`, so it
may import ``satellites`` (gr-satellites) at load. It builds a GNU Radio flowgraph
that demodulates + deframes the selected satellite and forwards each decoded frame
(PDU) to the data/status sockets, exactly like ``cubesat_gfsk_ax25_rx.py`` emits
``frame_received``.

Licensing: gr-satellites is GPLv3 (compatible with this repo). Do NOT pull in
gr-satnogs / satnogs-client (AGPLv3). Depend on gr-satellites (apt/pip, pinned);
do not vendor its source.

Status: BENCH-PENDING — not runnable on the dev box (needs GNU Radio + gr-satellites
+ gr-soapy). The gr-satellites embedding API has shifted across versions; the
construction call below is the documented shape and must be confirmed against the
installed version on the bench (as with ``gnuradio_gfsk.py``).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import queue

from gnuradio import gr, soapy

# gr-satellites flowgraph component. Import name/shape may vary by version
# (e.g. ``satellites.core.gr_satellites_flowgraph``); confirm on the bench.
from satellites.core import gr_satellites_flowgraph


class _FrameSink(gr.basic_block):
    """Collects decoded frame PDUs (gr-satellites' message output) into a queue."""

    def __init__(self) -> None:
        gr.basic_block.__init__(self, name="frame_sink", in_sig=None, out_sig=None)
        self._q: queue.Queue[bytes] = queue.Queue()
        self.message_port_register_in(gr.pmt.intern("in"))
        self.set_msg_handler(gr.pmt.intern("in"), self._on_msg)

    def _on_msg(self, msg) -> None:  # type: ignore[no-untyped-def]
        # gr-satellites emits a PDU: (metadata, u8-vector). Extract the bytes.
        payload = gr.pmt.cdr(msg)
        data = bytes(gr.pmt.u8vector_elements(payload))
        self._q.put(data)

    def drain(self) -> list[bytes]:
        out: list[bytes] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class _SatContext:
    def __init__(self, tb: gr.top_block, src, sink: _FrameSink, center_hz: float) -> None:
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

    def drain_frames(self) -> list[bytes]:
        return self._sink.drain()

    def set_doppler(self, offset_hz: float) -> None:
        self.src.set_frequency(0, self._center + offset_hz)


def build_satellites_rx(args, satellite: str, sample_rate: float) -> _SatContext:
    """Build a gr-satellites RX flowgraph for ``satellite`` (a SatYAML name/id).

    BENCH-PENDING: confirm the gr_satellites_flowgraph constructor signature and
    the decoded-frame message port name against the installed gr-satellites
    version. The shape below follows the documented embedding pattern.
    """
    tb = gr.top_block("gr_satellites_rx")
    src = soapy.source(args.sdr_args, "fc32", 1, "", [""], [""], [""], [""])
    src.set_sample_rate(0, float(sample_rate))
    src.set_frequency(0, float(args.center_freq_hz))

    flowgraph = gr_satellites_flowgraph.make(
        name=satellite,
        samp_rate=float(sample_rate),
        iq=True,
        grc_block=True,  # construct as an embeddable hier block
    )
    sink = _FrameSink()
    tb.connect(src, flowgraph)
    # gr-satellites exposes decoded frames on a message output port; forward them.
    tb.msg_connect(flowgraph, "out", sink, "in")
    return _SatContext(tb, src, sink, float(args.center_freq_hz))


__all__ = ["build_satellites_rx"]

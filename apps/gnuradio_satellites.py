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

import logging
import queue

from _recorder import PassRecorder
from _soapy import (
    apply_corrections,
    capture_plan,
    configure_soapy_source,
    make_source,
    merge_sdr_params,
    retune_source,
    sdr_env,
    tune_source,
)
from gnuradio import gr

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
    def __init__(
        self,
        tb: gr.top_block,
        src,
        sink: _FrameSink,
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
        self.tb.stop()
        self.tb.wait()  # let the SDF sink flush + close before we derive CSV/PNG
        if self._recorder is not None:
            self._recorder.finalize()

    def wait(self) -> None:
        self.tb.wait()

    def drain_frames(self) -> list[bytes]:
        return self._sink.drain()

    def set_doppler(self, offset_hz: float) -> None:
        retune_source(self.src, self._center, self._lo_offset, offset_hz)


def build_satellites_rx(
    args, satellite: str, sample_rate: float, params: dict | None = None
) -> _SatContext:
    """Build a gr-satellites RX flowgraph for ``satellite``.

    ``satellite`` may be a NORAD id (all digits) or a SatYAML name. The orchestrator
    passes the pass's NORAD id (from the command's ``satellite.noradId``), so the
    common path is NORAD selection — gr-satellites resolves the bird from its catalog
    by NORAD.

    BENCH-PENDING: confirm the gr_satellites_flowgraph constructor signature and
    the decoded-frame message port name against the installed gr-satellites
    version. The shape below follows the documented embedding pattern.
    """
    env = sdr_env()  # station-wide GS_SDR_* (antenna/gain/lo-offset/ppm/dc-removal/rate)
    lo = env["lo_offset_hz"]
    # The SDR samples at the capture rate (XTRX can't stream the narrow channel rate);
    # gr-satellites resamples internally, so we feed it the actual SDR rate.
    sdr_rate, _ = capture_plan(env["capture_rate_hz"], float(sample_rate))
    tb = gr.top_block("gr_satellites_rx")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo)  # LO offset → DC spike off-signal
    configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])

    # Pre-demod IQ capture FIRST (the priority): it taps the SDR source independently
    # of the decoder, so a gr-satellites build problem never costs us the recording.
    # Recorded at the SDR (capture) rate → a proper wideband IQ file.
    recorder = PassRecorder.maybe_start(args, tb, src, sample_rate_hz=sdr_rate)

    # gr-satellites decoder (best-effort). NORAD id (all digits) -> norad=; else a
    # SatYAML name. ``gr_satellites_flowgraph`` is instantiated DIRECTLY (there is no
    # ``.make()`` classmethod) as an embeddable hier block exposing an 'out' PDU port.
    sink = _FrameSink()
    try:
        sat_sel = {"norad": int(satellite)} if str(satellite).isdigit() else {"name": satellite}
        flowgraph = gr_satellites_flowgraph(
            samp_rate=sdr_rate,  # gr-satellites decimates to the bird's rate internally
            iq=True,
            grc_block=True,
            **sat_sel,
        )
        tb.connect(src, flowgraph)
        tb.msg_connect(flowgraph, "out", sink, "in")
    except Exception:
        # A decoder build failure (wrong satellite, gr-satellites API drift, missing
        # SatYAML) must NOT kill the IQ capture — log it and record IQ only.
        logging.getLogger("gr_satellites_rx").exception(
            "gr-satellites decoder build failed (satellite=%s); recording IQ only", satellite
        )
    return _SatContext(tb, src, sink, float(args.center_freq_hz), recorder, lo_offset_hz=lo)


__all__ = ["build_satellites_rx"]

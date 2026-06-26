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
import os
import queue
import re

from _recorder import PassRecorder
from _soapy import (
    apply_corrections,
    capture_plan,
    configure_soapy_source,
    make_decimator,
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

_log = logging.getLogger("gr_satellites_rx")

# Fallback demods run in parallel (all tap one decimator) when gr-satellites has no
# decoder for a bird — frames come from whichever locks. Covers the frame-producing
# modulations of 401 MHz LEO cubesats: GFSK/FSK/GMSK, BPSK/QPSK/PSK, AFSK. Override per
# station with GS_FALLBACK_DEMODS (comma list of "<kind><rate>"); trim it if the SoC is
# CPU-bound. (FM here would be voice/APT — no frames — so the IQ recording covers it.)
_DEFAULT_FALLBACK_DEMODS = (
    "gfsk9600,gfsk4800,gmsk9600,gmsk4800,bpsk1200,bpsk9600,qpsk9600,afsk1200"
)
# Channel rate the fallback demods run at (the SDR rate is decimated to this). ~5 sps
# for 9k6, ~10 for 4k8; ±25 kHz covers a typical narrowband cubesat downlink.
_FALLBACK_CHANNEL_RATE = 50_000


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
        *,
        fallbacks=None,
    ) -> None:
        self.tb = tb
        self.src = src
        self._sink = sink
        self._center = center_hz
        self._lo_offset = lo_offset_hz
        self._recorder = recorder
        # When gr-satellites has no decoder for this bird, ``fallbacks`` is a list of
        # _FallbackDemod tapping the source in parallel; frames come from any of them.
        self._fallbacks = list(fallbacks or [])

    @property
    def framing(self) -> str:
        return "fallback" if self._fallbacks else "grsatellites"

    def start(self) -> None:
        self.tb.start()

    def stop(self) -> None:
        self.tb.stop()
        self.tb.wait()  # let the IQ sink flush before finalize
        if self._recorder is not None:
            self._recorder.finalize()

    def wait(self) -> None:
        self.tb.wait()

    def drain_frames(self) -> list[bytes]:
        if not self._fallbacks:
            return self._sink.drain()  # gr-satellites decoded PDUs
        frames: list[bytes] = []
        for fb in self._fallbacks:  # try every configured fallback demod
            frames.extend(fb.drain_frames())
        return frames

    def set_doppler(self, offset_hz: float) -> None:
        retune_source(self.src, self._center, self._lo_offset, offset_hz)


class _FallbackDemod:
    """One fallback demod tapping the SDR source: a demodulator chain + a deframer.
    ``drain_frames`` returns the bytes of any frames recovered since the last call."""

    def __init__(self, name: str, sink, deframe) -> None:
        self.name = name
        self._sink = sink
        self._deframe = deframe

    def drain_frames(self) -> list[bytes]:
        return self._deframe(self._sink)


def _bits_deframe(bit_sink) -> list[bytes]:
    """Modulation-agnostic deframer: turn a demod's hard bits into frames. Tries AX.25
    both descrambled (G3RUH) and plain, then the EnduroSat chip packet (both bit
    polarities). The HDLC flag + FCS/CRC gate false positives, so trying everything is
    safe — only CRC-valid frames come back. Used by every fallback demod."""
    import numpy as np  # noqa: PLC0415

    from gfsk_ax25 import endurosat_link, framing  # noqa: PLC0415

    bits = bit_sink.drain()
    if not len(bits):
        return []
    frames: list[bytes] = []
    for scramble in (True, False):  # G3RUH-scrambled AX.25, then plain AX.25
        frames.extend(framing.decode(bits, scramble=scramble, nrzi=True))
    if frames:
        return frames
    arr = np.asarray(bits, dtype=np.uint8)
    return endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)


# Modulation kind -> demod builder. ``_build_fallbacks`` parses "<kind><rate>"
# (e.g. gfsk9600, gmsk4800, bpsk1200, qpsk9600, afsk1200) and dispatches here.
_PSK_ORDER = {"bpsk": 2, "psk": 2, "qpsk": 4}


def _build_fallbacks(tb, demod_src, sample_rate: float) -> list[_FallbackDemod]:
    """Build every configured fallback demod, all tapping ``demod_src`` (already at the
    channel rate, so they fan out from a single decimator). Covers the frame-producing
    modulations of 401 MHz LEO cubesats: GFSK / FSK / GMSK, BPSK / QPSK / PSK, AFSK.
    All share the modulation-agnostic ``_bits_deframe``."""
    from gnuradio_gfsk import (  # noqa: PLC0415
        connect_afsk_demod,
        connect_gfsk_demod,
        connect_psk_demod,
    )

    from gfsk_ax25 import endurosat  # noqa: PLC0415

    modes = (os.environ.get("GS_FALLBACK_DEMODS", "") or _DEFAULT_FALLBACK_DEMODS).split(",")
    out: list[_FallbackDemod] = []
    for raw in modes:
        mode = raw.strip().lower()
        if not mode:
            continue
        m = re.match(r"([a-z]+)(\d*)", mode)
        kind = m.group(1) if m else ""
        rate = float(m.group(2)) if (m and m.group(2)) else 0.0
        if kind in ("gfsk", "fsk", "gmsk"):  # 2-FSK family (GMSK = h≈0.5)
            mod_index = 0.5 if kind == "gmsk" else endurosat.LinkProfile().mod_index
            profile = endurosat.LinkProfile(symbol_rate_hz=rate or 9600.0, mod_index=mod_index)
            sink = connect_gfsk_demod(
                tb, demod_src, sample_rate, profile, decimate=False, sdr_rate=sample_rate
            )
        elif kind in _PSK_ORDER:  # BPSK / QPSK / PSK (coherent, differential)
            sink = connect_psk_demod(
                tb, demod_src, sample_rate, rate or 1200.0, order=_PSK_ORDER[kind]
            )
        elif kind == "afsk":  # Bell-202 1200/2200 Hz
            sink = connect_afsk_demod(tb, demod_src, sample_rate, baud=rate or 1200.0)
        else:
            _log.warning("fallback demod %r not implemented; skipping", mode)
            continue
        out.append(_FallbackDemod(mode, sink, _bits_deframe))
    return out


def _gr_satellites_selector(satellite) -> dict | None:
    """The gr-satellites SatYAML key for ``satellite`` — a NORAD id (canonical,
    unambiguous) when it is purely numeric, else a name. Returns None for an empty /
    non-numeric-garbage id so we never hand gr-satellites a bogus string."""
    s = str(satellite or "").strip()
    if not s:
        return None
    if s.isdigit():
        return {"norad": int(s)}
    return {"name": s}


def build_satellites_rx(
    args, satellite: str, sample_rate: float, params: dict | None = None
) -> _SatContext:
    """Build an RX flowgraph for ``satellite``: gr-satellites if it has a SatYAML
    decoder for the bird, otherwise the configured fallback demods. Either way the
    wideband IQ is recorded (the priority), so an unknown bird still yields a capture.

    ``satellite`` is normally the pass's NORAD id (``satellite.noradId``); we pass it to
    gr-satellites as a clean ``norad=`` int (or ``name=`` for a non-numeric id) — never
    a bogus string. If gr-satellites raises (not in its catalog / API drift), we switch
    to the fallback demods (GS_FALLBACK_DEMODS; default GFSK 9k6 + 4k8).

    BENCH-PENDING: confirm the gr_satellites_flowgraph constructor signature and the
    decoded-frame message port name against the installed gr-satellites version.
    """
    env = sdr_env()  # station-wide GS_SDR_* (antenna/gain/lo-offset/ppm/dc-removal/rate)
    lo = env["lo_offset_hz"]
    # The SDR samples at the capture rate (XTRX can't stream the narrow channel rate).
    # gr-satellites resamples internally (fed the SDR rate); the fallback demods get a
    # single shared decimator down to the channel rate.
    sdr_rate, _ = capture_plan(env["capture_rate_hz"], float(sample_rate))
    tb = gr.top_block("gr_satellites_rx")
    src = make_source(args.sdr_args)  # centralized gr-soapy signature (see _soapy)
    src.set_sample_rate(0, sdr_rate)
    tune_source(src, float(args.center_freq_hz), lo)  # LO offset → DC spike off-signal
    configure_soapy_source(src, merge_sdr_params(params))  # antenna + gain (else deaf)
    apply_corrections(src, ppm=env["ppm"], dc_removal=env["dc_removal"])

    # Pre-demod IQ capture FIRST (the priority): it taps the SDR source independently of
    # the decoder, so a decoder problem never costs us the recording. At the SDR rate.
    recorder = PassRecorder.maybe_start(args, tb, src, sample_rate_hz=sdr_rate)

    # gr-satellites decoder if the bird is in its catalog; else fall back. ``flowgraph``
    # is instantiated DIRECTLY (no ``.make()``) as a hier block exposing an 'out' PDU.
    sink = _FrameSink()
    fallbacks: list[_FallbackDemod] = []
    selector = _gr_satellites_selector(satellite)
    try:
        if selector is None:
            raise ValueError(f"no usable satellite id {satellite!r}")
        flowgraph = gr_satellites_flowgraph(
            samp_rate=sdr_rate,  # gr-satellites decimates to the bird's rate internally
            iq=True,
            grc_block=True,
            **selector,
        )
        tb.connect(src, flowgraph)
        tb.msg_connect(flowgraph, "out", sink, "in")
        _log.info("gr-satellites: decoding %s (%r)", satellite, selector)
    except Exception as e:  # noqa: BLE001 — any build failure → fall back, keep the IQ
        # Not in gr-satellites' catalog (or API drift): switch to the fallback demods.
        # gr-satellites' samp_rate is wideband (~2 MHz); the GFSK fallback needs a narrow
        # CHANNEL rate (few sps), so decimate the SDR rate to _FALLBACK_CHANNEL_RATE once
        # and fan all fallbacks out from it.
        _log.warning("gr-satellites: no decoder for %s (%s) → fallback demods", satellite, e)
        channel = make_decimator(sdr_rate, float(_FALLBACK_CHANNEL_RATE))
        tb.connect(src, channel)
        fallbacks = _build_fallbacks(tb, channel, float(_FALLBACK_CHANNEL_RATE))
        _log.info(
            "fallback demods @ %d Hz: %s",
            _FALLBACK_CHANNEL_RATE,
            ", ".join(f.name for f in fallbacks) or "(none)",
        )
    return _SatContext(
        tb, src, sink, float(args.center_freq_hz), recorder, lo_offset_hz=lo, fallbacks=fallbacks
    )


__all__ = ["build_satellites_rx"]

#!/usr/bin/env python3
"""Cubesat 2-GFSK / AX.25 receive flowgraph (EnduroSat-class UHF, 9k6).

Decodes a 2-GFSK, G3RUH-scrambled, AX.25-framed downlink (beacon + packets) and
emits one ``frame_received`` status event per valid frame plus the raw frame
bytes on the data socket (``data_format = "raw_bits"`` -> RAW_BITS artifact).

Two interchangeable engines, selected by ``--engine`` / ``GS_FLOWGRAPH_ENGINE``
env / params-file ``engine`` key (default ``dsp``):

* ``dsp``      — pure numpy/scipy demod (``gfsk_ax25`` library). Runs anywhere,
                 fully unit-tested. IQ from SoapySDR (``--sdr-args driver=...``)
                 or a ``cf32`` file (``--sdr-args file:/path.cf32``) for bench
                 use without hardware.
* ``gnuradio`` — GNU Radio front-end (IQ -> bits) for the bench, handing the
                 bitstream to the SAME tested ``framing.decode`` protocol layer.

Both engines share the scrambler/NRZI/HDLC/AX.25 code, so only the IQ->bits
front-end differs. The ``gnuradio`` path is validated on the Linux bench (it
imports ``gnuradio``); the ``dsp`` path is exercised by the repo's pytest suite.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys

import numpy as np
from _recorder import StreamRecorder
from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)

from gfsk_ax25 import ax25, endurosat, endurosat_link

VERSION = "0.1.0"

_DEFAULT_SAMPLE_RATE = 96_000  # >= 5x the 18.7 kHz channel; integer-ish sps
_DECODE_PERIOD_S = 2.0  # how often the dsp engine re-decodes the capture
_READ_CHUNK = 4096


def _select_engine(args, params: dict[str, object]) -> str:
    explicit = getattr(args, "engine", "") or ""
    env = os.environ.get("GS_FLOWGRAPH_ENGINE", "")
    from_params = str(params.get("engine", "")) if isinstance(params, dict) else ""
    engine = (explicit or env or from_params or "dsp").lower()
    if engine not in ("dsp", "gnuradio"):
        logging.getLogger("cubesat_gfsk_ax25_rx").warning(
            "unknown engine %r; falling back to dsp", engine
        )
        engine = "dsp"
    return engine


def _select_framing(params: dict[str, object]) -> str:
    """ax25 (default, compat) | endurosat (chip-packet, the real Gen-2 link).

    Via GS_FLOWGRAPH_FRAMING env or a params ``framing`` key. EnduroSat payloads
    are the opaque (encrypted) AirMAC frames — the orchestrator handles those.
    """
    framing = (
        os.environ.get("GS_FLOWGRAPH_FRAMING", "")
        or (str(params.get("framing", "")) if isinstance(params, dict) else "")
        or "ax25"
    ).lower()
    return framing if framing in ("ax25", "endurosat") else "ax25"


def _profile_from_params(params: dict[str, object]) -> endurosat.LinkProfile:
    return endurosat.LinkProfile(
        scramble=bool(params.get("scramble", True)),
        nrzi=bool(params.get("nrzi", True)),
        mod_index=float(params.get("mod_index", endurosat.MOD_INDEX)),  # type: ignore[arg-type]
        bt=float(params.get("bt", endurosat.BT)),  # type: ignore[arg-type]
        symbol_rate_hz=float(params.get("symbol_rate_hz", endurosat.SYMBOL_RATE_HZ)),  # type: ignore[arg-type]
    )


async def _emit_frame(sockets, body: bytes, *, framing: str = "ax25") -> None:
    """Send a frame_received status event + raw frame bytes on the data socket.

    For ``endurosat`` framing the body is the opaque (encrypted AirMAC) payload,
    so we do not attempt an AX.25 parse — the orchestrator decodes it.
    """
    ui = ax25.decode_ui(body) if framing == "ax25" else None
    event: dict[str, object] = {
        "event": "frame_received",
        "framing": framing,
        "frame": {
            "bytes_b64": base64.b64encode(body).decode("ascii"),
            "len": len(body),
            "crc_ok": True,
        },
    }
    if ui is not None:
        event["frame"].update({"dest": ui.dest, "src": ui.src, "info_len": len(ui.info)})  # type: ignore[union-attr]
    await send_event(sockets.status_writer, event)
    try:
        sockets.data_writer.write(body)
        await sockets.data_writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        logging.getLogger("cubesat_gfsk_ax25_rx").warning("data socket closed; frame not stored")


# ----------------------------------------------------------------------
# dsp engine: SoapySDR / file IQ source -> numpy demod
# ----------------------------------------------------------------------


def _open_iq_source(args):
    """Return a blocking iterator of complex64 chunks (SoapySDR or cf32 file)."""
    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        path = sdr_args[len("file:") :]
        return _file_iq_chunks(path)
    return _soapy_iq_chunks(args)  # pragma: no cover (needs hardware/SoapySDR)


def _file_iq_chunks(path: str):
    with open(path, "rb") as f:
        while True:
            raw = f.read(_READ_CHUNK * 8)  # complex64 = 8 bytes
            if not raw:
                return
            yield np.frombuffer(raw, dtype=np.complex64)


def _soapy_iq_chunks(args):  # pragma: no cover (needs hardware/SoapySDR)
    import SoapySDR  # lazy: only the dsp+hardware path needs it
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

    dev = SoapySDR.Device(args.sdr_args)
    dev.setSampleRate(SOAPY_SDR_RX, 0, float(args.sample_rate or _DEFAULT_SAMPLE_RATE))
    dev.setFrequency(SOAPY_SDR_RX, 0, float(args.center_freq_hz))
    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    dev.activateStream(stream)
    buff = np.empty(_READ_CHUNK, dtype=np.complex64)
    try:
        while True:
            sr = dev.readStream(stream, [buff], len(buff))
            if sr.ret > 0:
                yield buff[: sr.ret].copy()
    finally:
        dev.deactivateStream(stream)
        dev.closeStream(stream)


async def _run_dsp_engine(args, sockets, params, started, stop_requested, profile, doppler) -> None:
    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    framing = _select_framing(params)
    if framing == "endurosat":
        # The EnduroSat chip link is 9600 sym/s (endurosat_link defaults), not the
        # 12480 the AX.25 LinkProfile assumes; honour params overrides if present.
        sym_hz = float(params.get("symbol_rate_hz", endurosat_link.DEFAULT_SYMBOL_RATE_HZ))
        decoder: object = endurosat_link.StreamDecoder(
            sample_rate,
            symbol_rate_hz=sym_hz,
            mod_index=float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
            bt=float(params.get("bt", endurosat_link.DEFAULT_BT)),
        )
    else:
        decoder = endurosat.StreamDecoder(sample_rate, profile=profile, recover_timing=True)

    # Digital Doppler NCO. ``doppler`` is shared with the command handler, so a
    # set_doppler mid-pass is picked up here on the next chunk; nco_phase keeps
    # the correction phase-continuous across chunks.
    nco_phase = 0.0

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "raw_bits",
            "sample_rate": int(sample_rate),
            "symbol_rate": int(profile.symbol_rate_hz),
            "engine": "dsp",
            "framing": framing,
            "flowgraph_version": VERSION,
        },
    )
    await started.wait()

    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_running_loop()
    # Pre-demod IQ capture: record the RAW chunk (before the Doppler NCO / demod).
    recorder = StreamRecorder.maybe_start(args, sample_rate_hz=sample_rate)

    def _reader() -> None:
        try:
            for chunk in _open_iq_source(args):
                if stop_requested.is_set():
                    break
                arr = np.asarray(chunk, dtype=np.complex64)
                if recorder is not None:
                    recorder.write(arr)
                loop.call_soon_threadsafe(queue.put_nowait, arr)
        except Exception:
            log.exception("IQ source error")
        finally:
            if recorder is not None:
                recorder.finalize()  # close SDF + derive CSV/PNG (numpy; in this worker thread)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    reader_task = loop.run_in_executor(None, _reader)

    async def _decode_loop() -> None:
        while not stop_requested.is_set():
            await asyncio.sleep(_DECODE_PERIOD_S)
            for body in decoder.decode_new():
                await _emit_frame(sockets, body, framing=framing)

    decode_task = asyncio.create_task(_decode_loop(), name="decode-loop")
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            off = doppler["hz"]
            if off:
                n = np.arange(len(chunk))
                ph = nco_phase - 2.0 * np.pi * off * n / sample_rate
                chunk = (chunk * np.exp(1j * ph)).astype(np.complex64)
                nco_phase = float(ph[-1]) if len(ph) else nco_phase
            decoder.push(chunk)
    finally:
        stop_requested.set()
        decode_task.cancel()
        for body in decoder.flush():
            await _emit_frame(sockets, body, framing=framing)
        await asyncio.gather(reader_task, decode_task, return_exceptions=True)


# ----------------------------------------------------------------------
# gnuradio engine: GR front-end -> shared framing.decode (bench)
# ----------------------------------------------------------------------


async def _run_gnuradio_engine(  # pragma: no cover (bench)
    args, sockets, params, started, stop_requested, profile, doppler
) -> None:
    """Bench engine: a GNU Radio top_block recovers the bitstream; the SAME
    ``framing.decode`` turns bits into AX.25 frames. Built lazily so this file
    imports on hosts without GNU Radio."""
    from gnuradio_gfsk import build_rx_top_block  # bench-only helper module

    from gfsk_ax25 import framing

    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    ctx = build_rx_top_block(args, profile, sample_rate, params)
    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "raw_bits",
            "sample_rate": int(sample_rate),
            "engine": "gnuradio",
            "flowgraph_version": VERSION,
        },
    )
    await started.wait()
    ctx.start()
    last_doppler = 0.0
    try:
        while not stop_requested.is_set():
            await asyncio.sleep(_DECODE_PERIOD_S)
            if doppler["hz"] != last_doppler:
                last_doppler = doppler["hz"]
                ctx.set_doppler(last_doppler)  # retune the SoapySDR source
            bits = ctx.drain_bits()  # np.uint8 hard bits recovered by GR
            for body in framing.decode(bits, scramble=profile.scramble, nrzi=profile.nrzi):
                await _emit_frame(sockets, body, framing="ax25")
    finally:
        ctx.stop()
        ctx.wait()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


async def amain(args) -> int:
    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    params = load_params(args)
    engine = _select_engine(args, params)
    profile = _profile_from_params(params)
    log.info("engine=%s profile=%s", engine, profile)

    sockets = await connect_spawn_sockets(args)
    started = asyncio.Event()
    stop_requested = asyncio.Event()
    doppler = {"hz": 0.0}

    async def _on_start(_cmd: dict[str, object]) -> None:
        started.set()
        await send_event(sockets.status_writer, {"event": "started"})

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        started.set()  # release any waiter
        await send_event(
            sockets.status_writer,
            {"event": "stopped", "reason": str(cmd.get("reason", "command"))},
        )

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        if isinstance(off, (int, float)):
            doppler["hz"] = float(off)  # shared with the running engine

    engine_fn = _run_dsp_engine if engine == "dsp" else _run_gnuradio_engine
    engine_task = asyncio.create_task(
        engine_fn(args, sockets, params, started, stop_requested, profile, doppler),
        name=f"engine-{engine}",
    )
    handlers = {"start": _on_start, "stop": _on_stop, "set_doppler": _on_set_doppler}
    try:
        await run_command_loop(sockets.control_reader, handlers)
    finally:
        stop_requested.set()
        started.set()
        await asyncio.gather(engine_task, return_exceptions=True)
        await sockets.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="cubesat_gfsk_ax25_rx",
        description="2-GFSK / AX.25 (9k6) cubesat receiver — dsp | gnuradio engines.",
    )
    parser.add_argument("--engine", default="", choices=["", "dsp", "gnuradio"])
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    logging.basicConfig(level=logging.INFO)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

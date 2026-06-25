#!/usr/bin/env python3
"""Cubesat 2-GFSK / AX.25 receive flowgraph (EnduroSat-class UHF, 9k6).

Decodes a 2-GFSK, G3RUH-scrambled, AX.25-framed downlink (beacon + packets) and
emits one ``frame_received`` status event per valid frame plus the raw frame
bytes on the data socket (``data_format = "raw_bits"`` -> RAW_BITS artifact).

Two interchangeable engines, selected by ``--engine`` / ``GS_FLOWGRAPH_ENGINE``
env / params-file ``engine`` key (default ``gnuradio``):

* ``gnuradio`` — GNU Radio front-end (IQ -> bits) using the gr-soapy source. The
                 default + production hardware path (SatNOGS-proven: gr-soapy owns
                 stream activation/MTU/timeouts). Hands the bitstream to the SAME
                 tested ``framing.decode`` protocol layer.
* ``dsp``      — pure numpy/scipy demod (``gfsk_ax25`` library). Backup engine:
                 runs anywhere, fully unit-tested, used for file-IQ bench work and
                 as a fallback. IQ from SoapySDR (``--sdr-args driver=...``) via a
                 hand-rolled read loop, or a ``cf32`` file (``--sdr-args
                 file:/path.cf32``).

Both engines share the scrambler/NRZI/HDLC/AX.25 code, so only the IQ->bits
front-end differs. The ``gnuradio`` path is validated on the Linux bench (it
imports ``gnuradio``); the ``dsp`` path is exercised by the repo's pytest suite.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import sys

import numpy as np
from _recorder import StreamRecorder
from _soapy import merge_sdr_params, sdr_env
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
_DEFAULT_SDR_GAIN_DB = 40.0  # manual RX gain when none is configured (0 dB = deaf)


def _select_engine(args, params: dict[str, object]) -> str:
    explicit = getattr(args, "engine", "") or ""
    env = os.environ.get("GS_FLOWGRAPH_ENGINE", "")
    from_params = str(params.get("engine", "")) if isinstance(params, dict) else ""
    # Default to the gr-soapy (gnuradio) front-end: it's the SatNOGS-proven path
    # (gr-soapy source handles stream activation/MTU/timeouts), whereas the dsp
    # engine hand-rolls SoapySDR.readStream and is kept as a backup (higher
    # frequencies, TX, file-IQ, the pytest suite).
    engine = (explicit or env or from_params or "gnuradio").lower()
    if engine not in ("dsp", "gnuradio"):
        logging.getLogger("cubesat_gfsk_ax25_rx").warning(
            "unknown engine %r; falling back to gnuradio", engine
        )
        engine = "gnuradio"
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


def _open_iq_source(args, params=None):
    """Return a blocking iterator of complex64 chunks (SoapySDR or cf32 file)."""
    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        path = sdr_args[len("file:") :]
        return _file_iq_chunks(path)
    return _soapy_iq_chunks(args, params or {})  # pragma: no cover (needs hardware)


def _file_iq_chunks(path: str):
    with open(path, "rb") as f:
        while True:
            raw = f.read(_READ_CHUNK * 8)  # complex64 = 8 bytes
            if not raw:
                return
            yield np.frombuffer(raw, dtype=np.complex64)


def _sdr_channel(args) -> int:
    """RX channel index from ``--sdr-port`` when it is a plain integer; else 0.
    A non-numeric port (e.g. ``RX1``) is treated as an antenna name, not a channel."""
    port = str(getattr(args, "sdr_port", "") or "").strip()
    return int(port) if port.isdigit() else 0


def _select_rx_antenna(dev, soapy_rx, ch, args, params, log) -> None:  # pragma: no cover
    """Select + set the RX antenna. Priority: ``params['sdr_antenna']`` → the
    ``--sdr-port`` label if the driver lists it → the first non-``NONE`` antenna.
    The available list + the chosen one are logged so a wrong physical port (the
    classic "0-byte capture / deaf radio") is obvious from the journal."""
    available = [str(a) for a in dev.listAntennas(soapy_rx, ch)]
    port = str(getattr(args, "sdr_port", "") or "").strip()
    chosen = None
    for cand in (params.get("sdr_antenna"), port):
        if isinstance(cand, str) and cand in available:
            chosen = cand
            break
    if chosen is None and available:
        chosen = next((a for a in available if a.upper() != "NONE"), available[0])
    if chosen is not None:
        dev.setAntenna(soapy_rx, ch, chosen)
    log.info("dsp SDR: RX antenna=%s (available=%s, requested port=%r)", chosen, available, port)


def _soapy_iq_chunks(args, params):  # pragma: no cover (needs hardware/SoapySDR)
    import SoapySDR  # lazy: only the dsp+hardware path needs it
    from SoapySDR import (
        SOAPY_SDR_CF32,
        SOAPY_SDR_OVERFLOW,
        SOAPY_SDR_RX,
        SOAPY_SDR_TIMEOUT,
    )

    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    ch = _sdr_channel(args)
    rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    freq = float(args.center_freq_hz)

    params = merge_sdr_params(params)  # station GS_SDR_* antenna/gain/agc defaults
    env = sdr_env()
    lo = env["lo_offset_hz"]

    dev = SoapySDR.Device(args.sdr_args)
    dev.setSampleRate(SOAPY_SDR_RX, ch, rate)
    # LO offset: tune the analog LO off-carrier (RF) + the baseband CORDIC back (BB)
    # so the DC spike sits at +lo, not on the signal. The dsp NCO handles Doppler at
    # baseband, so the LO is set once here.
    if lo:
        try:
            dev.setFrequency(SOAPY_SDR_RX, ch, "RF", freq + lo)
            dev.setFrequency(SOAPY_SDR_RX, ch, "BB", -lo)
        except Exception:  # noqa: BLE001 — driver without RF/BB split → direct tune
            dev.setFrequency(SOAPY_SDR_RX, ch, freq)
    else:
        dev.setFrequency(SOAPY_SDR_RX, ch, freq)
    _select_rx_antenna(dev, SOAPY_SDR_RX, ch, args, params, log)
    if env["ppm"]:
        with contextlib.suppress(Exception):
            dev.setFrequencyCorrection(SOAPY_SDR_RX, ch, env["ppm"])
    if env["dc_removal"]:
        with contextlib.suppress(Exception):
            dev.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

    # Front-end gain: GS_SDR_AGC enables hardware AGC; otherwise explicit
    # ``sdr_gain_db`` wins, else AGC off + a sane manual default (a 0 dB front-end is
    # effectively deaf and was never set before). Mirrors the gnuradio engines.
    gain_db = params.get("sdr_gain_db")
    if params.get("sdr_agc"):
        with contextlib.suppress(Exception):
            dev.setGainMode(SOAPY_SDR_RX, ch, True)  # hardware AGC
        gain_str = "AGC"
    elif isinstance(gain_db, (int, float)) and not isinstance(gain_db, bool):
        dev.setGain(SOAPY_SDR_RX, ch, float(gain_db))
        gain_str = f"{float(gain_db):.1f} dB"
    else:
        with contextlib.suppress(Exception):
            dev.setGainMode(SOAPY_SDR_RX, ch, False)  # disable AGC
        dev.setGain(SOAPY_SDR_RX, ch, float(_DEFAULT_SDR_GAIN_DB))
        gain_str = f"{_DEFAULT_SDR_GAIN_DB:.1f} dB (default)"
    log.info(
        "dsp SDR: %s ch=%d rate=%.0f freq=%.0f lo_offset=%.0f gain=%s ppm=%.2f "
        "dc_removal=%s — opening RX stream",
        args.sdr_args, ch, rate, freq, lo, gain_str, env["ppm"], env["dc_removal"],
    )

    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
    dev.activateStream(stream)
    buff = np.empty(_READ_CHUNK, dtype=np.complex64)
    total = reads = timeouts = overflows = errors = 0
    try:
        while True:
            # 1 s timeout so a stalled SDR surfaces as logged timeouts, not a silent
            # spin (the old loop swallowed every ret<=0 and recorded nothing).
            sr = dev.readStream(stream, [buff], len(buff), timeoutUs=1_000_000)
            if sr.ret > 0:
                total += sr.ret
                reads += 1
                if reads == 1:
                    log.info("dsp SDR: streaming — first %d samples received", sr.ret)
                elif reads % 5000 == 0:
                    log.info("dsp SDR: %.2f Msamples read so far", total / 1e6)
                yield buff[: sr.ret].copy()
            elif sr.ret == SOAPY_SDR_TIMEOUT:
                timeouts += 1
                if timeouts in (1, 10, 100) or timeouts % 1000 == 0:
                    log.warning(
                        "dsp SDR: readStream TIMEOUT x%d — no samples from %s "
                        "(check antenna/clock/driver)", timeouts, args.sdr_args,
                    )
            elif sr.ret == SOAPY_SDR_OVERFLOW:
                overflows += 1  # 'O' — host not draining fast enough; keep going
            else:
                errors += 1
                if errors in (1, 10) or errors % 1000 == 0:
                    log.warning("dsp SDR: readStream error ret=%d flags=%d", sr.ret, sr.flags)
    finally:
        log.info(
            "dsp SDR: closing stream — total=%.2f Ms reads=%d timeouts=%d overflows=%d errors=%d",
            total / 1e6, reads, timeouts, overflows, errors,
        )
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
            for chunk in _open_iq_source(args, params):
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

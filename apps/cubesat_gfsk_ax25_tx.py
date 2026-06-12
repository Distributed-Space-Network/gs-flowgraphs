#!/usr/bin/env python3
"""Cubesat 2-GFSK / AX.25 transmit flowgraph (EnduroSat-class UHF uplink, 9k6).

The mirror of ``cubesat_gfsk_ax25_rx.py``: builds an AX.25 UI frame from the
uplink payload, runs the shared ``gfsk_ax25`` transmit chain (HDLC -> NRZI ->
G3RUH -> 2-GFSK), and sinks the baseband IQ to the SDR (or a ``cf32`` file for
bench use). Same ``--engine {dsp,gnuradio}`` selection as the receiver.

Uplink payload sources, in order: params ``uplink_b64`` (base64), params
``uplink_file`` path, or ``<output-dir>/uplink.bin``. Callsigns come from params
``dest`` / ``src`` (default ``CQ`` / ``DSN``).

NOTE: keying the PA is the orchestrator's safety FSM responsibility (Document A
A.6); this flowgraph only produces/sends the modulated baseband and never
asserts PTT itself.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

import numpy as np
from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)

from gfsk_ax25 import ax25, endurosat, endurosat_link

VERSION = "0.1.0"
_DEFAULT_SAMPLE_RATE = 96_000


def _select_engine(args, params: dict[str, object]) -> str:
    engine = (
        (getattr(args, "engine", "") or "")
        or os.environ.get("GS_FLOWGRAPH_ENGINE", "")
        or (str(params.get("engine", "")) if isinstance(params, dict) else "")
        or "dsp"
    ).lower()
    return engine if engine in ("dsp", "gnuradio") else "dsp"


def _select_framing(params: dict[str, object]) -> str:
    """ax25 (default) | endurosat (chip-packet). For endurosat the uplink payload
    is the already-built (encrypted AirMAC) frame, sent verbatim in the packet."""
    framing = (
        os.environ.get("GS_FLOWGRAPH_FRAMING", "")
        or (str(params.get("framing", "")) if isinstance(params, dict) else "")
        or "ax25"
    ).lower()
    return framing if framing in ("ax25", "endurosat") else "ax25"


def _uplink_payload(args, params: dict[str, object]) -> bytes:
    b64 = params.get("uplink_b64")
    if isinstance(b64, str) and b64:
        return base64.b64decode(b64)
    for candidate in (params.get("uplink_file"), Path(args.output_dir or ".") / "uplink.bin"):
        if candidate and Path(candidate).exists():
            return Path(candidate).read_bytes()
    return b""


def _build_frame_iq(args, params: dict[str, object], profile) -> np.ndarray:
    payload = _uplink_payload(args, params)
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    if _select_framing(params) == "endurosat":
        # Uplink payload is the already-built (encrypted AirMAC) frame; wrap it in
        # the EnduroSat chip packet (preamble + sync + len + CRC-16) at 9600 sym/s
        # (endurosat_link defaults), honouring params overrides if present.
        sym_hz = float(params.get("symbol_rate_hz", endurosat_link.DEFAULT_SYMBOL_RATE_HZ))
        return endurosat_link.transmit(
            payload[: endurosat_link.MAX_PAYLOAD],
            sample_rate,
            symbol_rate_hz=sym_hz,
            mod_index=float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
            bt=float(params.get("bt", endurosat_link.DEFAULT_BT)),
        )
    body = ax25.encode_ui(
        dest=str(params.get("dest", "CQ")),
        src=str(params.get("src", "DSN")),
        info=payload[: endurosat.AX25_INFO_MAX_BYTES],
    )
    return endurosat.transmit(body, sample_rate, profile=profile)


def _sink_iq(args, iq: np.ndarray) -> None:
    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        Path(sdr_args[len("file:") :]).write_bytes(iq.astype(np.complex64).tobytes())
        return
    _soapy_sink(args, iq)  # pragma: no cover (needs hardware/SoapySDR)


def _soapy_sink(args, iq: np.ndarray) -> None:  # pragma: no cover (needs hardware)
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX

    dev = SoapySDR.Device(args.sdr_args)
    dev.setSampleRate(SOAPY_SDR_TX, 0, float(args.sample_rate or _DEFAULT_SAMPLE_RATE))
    dev.setFrequency(SOAPY_SDR_TX, 0, float(args.center_freq_hz))
    stream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    dev.activateStream(stream)
    try:
        buf = iq.astype(np.complex64)
        i = 0
        while i < len(buf):
            sr = dev.writeStream(stream, [buf[i : i + 4096]], len(buf[i : i + 4096]))
            i += sr.ret if sr.ret > 0 else 0
    finally:
        dev.deactivateStream(stream)
        dev.closeStream(stream)


async def amain(args) -> int:
    log = logging.getLogger("cubesat_gfsk_ax25_tx")
    params = load_params(args)
    engine = _select_engine(args, params)
    profile = endurosat.LinkProfile(
        scramble=bool(params.get("scramble", True)),
        nrzi=bool(params.get("nrzi", True)),
    )
    log.info("engine=%s", engine)

    sockets = await connect_spawn_sockets(args)
    stop_requested = asyncio.Event()

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "none",
            "engine": engine,
            "flowgraph_version": VERSION,
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        await send_event(sockets.status_writer, {"event": "started"})
        if engine == "gnuradio":  # pragma: no cover (bench)
            from gnuradio_gfsk import transmit_gnuradio

            await asyncio.to_thread(transmit_gnuradio, args, params, profile)
        else:
            iq = _build_frame_iq(args, params, profile)
            await asyncio.to_thread(_sink_iq, args, iq)
        await send_event(
            sockets.status_writer,
            {"event": "transmit_complete", "samples": 0},
        )

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        await send_event(
            sockets.status_writer,
            {"event": "stopped", "reason": str(cmd.get("reason", "command"))},
        )

    handlers = {"start": _on_start, "stop": _on_stop}
    try:
        await run_command_loop(sockets.control_reader, handlers)
    finally:
        stop_requested.set()
        await sockets.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="cubesat_gfsk_ax25_tx",
        description="2-GFSK / AX.25 (9k6) cubesat transmitter — dsp | gnuradio engines.",
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

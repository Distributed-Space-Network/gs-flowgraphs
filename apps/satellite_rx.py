#!/usr/bin/env python3
"""Generic multi-mission satellite receiver (gr-satellites bridge).

A spawn-contract flowgraph that decodes ANY gr-satellites-supported satellite —
selected by a ``satellite`` (SatYAML name/id) params key — and emits each decoded
frame over the status/data sockets. The demod + deframe is gr-satellites (GPLv3,
the canonical multi-mission decoder library); this app is just the spawn-contract
adapter (Document F). For the EnduroSat mission use the dedicated, tested
``cubesat_gfsk_ax25_rx.py`` (``--framing endurosat``) instead.

BENCH-PENDING: needs GNU Radio + gr-satellites + gr-soapy; not runnable on the dev
box (the gr-satellites pieces are imported lazily in the engine task).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys

from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)

VERSION = "0.1.0"
_DECODE_PERIOD_S = 2.0
_DEFAULT_SAMPLE_RATE = 2_000_000


async def _emit_frame(sockets, frame: bytes, satellite: str) -> None:
    await send_event(
        sockets.status_writer,
        {
            "event": "frame_received",
            "decoder": "gr-satellites",
            "satellite": satellite,
            "frame": {"bytes_b64": base64.b64encode(frame).decode("ascii"), "len": len(frame)},
        },
    )
    try:
        sockets.data_writer.write(frame)
        await sockets.data_writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        logging.getLogger("satellite_rx").warning("data socket closed; frame not stored")


async def amain(args) -> int:
    log = logging.getLogger("satellite_rx")
    params = load_params(args)
    satellite = str(params.get("satellite", "")) or getattr(args, "satellite", "")
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)

    sockets = await connect_spawn_sockets(args)
    started = asyncio.Event()
    stop_requested = asyncio.Event()
    doppler = {"hz": 0.0}

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "frames",
            "engine": "gnuradio",
            "decoder": "gr-satellites",
            "satellite": satellite,
            "flowgraph_version": VERSION,
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        started.set()
        await send_event(sockets.status_writer, {"event": "started"})

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        started.set()
        await send_event(
            sockets.status_writer,
            {"event": "stopped", "reason": str(cmd.get("reason", "command"))},
        )

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        if isinstance(off, (int, float)):
            doppler["hz"] = float(off)

    async def _engine() -> None:  # pragma: no cover (bench)
        from gnuradio_satellites import build_satellites_rx

        if not satellite:
            log.error("no 'satellite' selected (params 'satellite' / --satellite); nothing to do")
            return
        # Pre-demod IQ capture is wired inside build_satellites_rx (PassRecorder taps
        # the SDR source; ctx.stop() finalizes) — uniform with the other RX engines.
        ctx = build_satellites_rx(args, satellite, sample_rate, params)
        # RX: start streaming + recording at spawn (arm — before AOS) so the front-end
        # is warm and the whole window is captured. Do NOT gate on cmd:start (that is
        # for TX keying); cmd:start still fires the 'started' status event and cmd:stop
        # ends the pass.
        ctx.start()
        last_doppler = 0.0
        try:
            while not stop_requested.is_set():
                await asyncio.sleep(_DECODE_PERIOD_S)
                if doppler["hz"] != last_doppler:
                    last_doppler = doppler["hz"]
                    ctx.set_doppler(last_doppler)
                for frame in ctx.drain_frames():
                    await _emit_frame(sockets, frame, satellite)
        finally:
            ctx.stop()
            ctx.wait()

    engine_task = asyncio.create_task(_engine(), name="gr-satellites")
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
        prog="satellite_rx", description="Multi-mission gr-satellites receiver (bench)."
    )
    parser.add_argument("--satellite", default="", help="gr-satellites SatYAML name/id")
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

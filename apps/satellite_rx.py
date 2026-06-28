#!/usr/bin/env python3
"""Generic multi-mission satellite receiver (gr-satellites + fallback demods).

A spawn-contract flowgraph that records the wideband IQ of EVERY pass (the priority,
SatNOGS-style) and decodes the bird when it can: gr-satellites (GPLv3, the canonical
multi-mission decoder) when the ``satellite`` (NORAD id / SatYAML name) is in its
catalog, otherwise the configured fallback demods (GS_FALLBACK_DEMODS) run in parallel —
GFSK / FSK / GMSK, BPSK / QPSK / PSK, and AFSK — and frames come from whichever locks.
So an uncatalogued 401 MHz LEO bird still yields IQ, plus a best-effort frame decode. For
the EnduroSat mission use the dedicated, tested ``cubesat_gfsk_ax25_rx.py``
(``--framing endurosat``) instead.

BENCH-PENDING: needs GNU Radio + gr-satellites + gr-soapy; not runnable on the dev
box (the gr-satellites pieces are imported lazily in the engine task).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

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


def _append_frame_record(output_dir: str | None, frame: bytes, decoder: str) -> None:
    """Append a ``{raw, deframed}`` record to ``<output_dir>/frames.jsonl`` (lives in the
    pass dir → obeys the same IQ retention). For the multi-mission RX the decoded frame
    IS the deframed unit, so ``raw`` and ``deframed`` are the same on-wire bytes."""
    if not output_dir:
        return
    rec = {
        "ts": time.time(),
        "decoder": decoder,
        "len": len(frame),
        "raw_hex": frame.hex(),
        "deframed_hex": frame.hex(),
    }
    try:
        with (Path(output_dir) / "frames.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        logging.getLogger("satellite_rx").warning("could not append frames.jsonl: %s", e)


async def _emit_frame(
    sockets, frame: bytes, satellite: str, *, decoder: str = "gr-satellites", output_dir=None
) -> None:
    _append_frame_record(output_dir, frame, decoder)
    await send_event(
        sockets.status_writer,
        {
            "event": "frame_received",
            "decoder": decoder,
            "satellite": satellite,
            "frame": {"bytes_b64": base64.b64encode(frame).decode("ascii"), "len": len(frame)},
        },
    )
    # NOTE: we deliberately do NOT tee frames to the data socket. The decoded-frames product
    # is frames.jsonl (appended above; gs-client uploads it post-pass). Streaming frames to the
    # data socket would also tee them to raw_bits.bin, which then races frames.jsonl to the same
    # presigned upload URL. frames.jsonl is the single source of truth for decoded frames.


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
        if not satellite:
            log.error("no 'satellite' selected (params 'satellite' / --satellite); nothing to do")
            return
        try:
            # Import INSIDE the try: a missing dependency (e.g. an un-deployed helper
            # module) must surface in the journal, not get swallowed by the outer gather()
            # — which once left a dead pass with no capture and no error.
            from gnuradio_satellites import build_satellites_rx

            # Pre-demod IQ capture is wired inside build_satellites_rx (PassRecorder taps
            # the SDR source; ctx.stop() finalizes) — uniform with the other RX engines.
            ctx = build_satellites_rx(args, satellite, sample_rate, params)
            # RX: start streaming + recording at spawn (arm — before AOS) so the
            # front-end is warm and the whole window is captured. Do NOT gate on
            # cmd:start (that's for TX keying); cmd:start still fires the 'started'
            # status event and cmd:stop ends the pass.
            ctx.start()
            decoder = "gr-satellites" if ctx.framing == "grsatellites" else "fallback"
            out_dir = getattr(args, "output_dir", None)
            log.info(
                "satellite_rx: flowgraph started (sat=%s, decoder=%s); streaming+recording",
                satellite,
                decoder,
            )
        except Exception:
            # Build/start failures used to be swallowed by the outer gather() — log
            # them so a bad gr-satellites build / SDR error is visible at teardown.
            log.exception("gr-satellites: engine failed to start (satellite=%s)", satellite)
            stop_requested.set()
            return
        last_doppler = 0.0
        try:
            while not stop_requested.is_set():
                await asyncio.sleep(_DECODE_PERIOD_S)
                if doppler["hz"] != last_doppler:
                    last_doppler = doppler["hz"]
                    ctx.set_doppler(last_doppler)
                # Decode is LIVE: gr-satellites and/or our one demod (the bird's backend mode),
                # each frame tagged with the engine that produced it. Frames -> status events +
                # frames.jsonl, which gs-client uploads post-pass. No bank, no post-pass decode.
                for source, frame in ctx.drain_frames():
                    await _emit_frame(
                        sockets, frame, satellite, decoder=source, output_dir=out_dir
                    )
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

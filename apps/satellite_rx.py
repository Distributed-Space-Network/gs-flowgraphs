#!/usr/bin/env python3
"""Generic multi-mission satellite receiver (gr-satellites + the universal modem/framing).

A spawn-contract flowgraph that records the wideband IQ of EVERY pass (the priority,
SatNOGS-style) and decodes the bird when it can. Decode is fully BACKEND-DRIVEN (docs/08):
gr-satellites (GPLv3, the canonical multi-mission decoder) when the ``satellite`` (NORAD id /
SatYAML name) is in its catalog OR a synthetic SatYAML can be built from the backend's
``(modulation, framing, baud)``, racing the ONE backend-specified demod from the modem
registry (first CRC-valid frame wins). There is no brute-force demod bank
(``GS_FALLBACK_DEMODS`` is deprecated and ignored). So an uncatalogued 401 MHz LEO bird still
yields IQ, plus a best-effort frame decode. For the EnduroSat mission use the dedicated,
tested ``cubesat_gfsk_ax25_rx.py`` (``--framing endurosat``) instead.

BENCH-PENDING: needs GNU Radio + gr-satellites + gr-soapy; not runnable on the dev
box (the gr-satellites pieces are imported lazily in the engine task).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sys
import time
from pathlib import Path

from _doppler import NullDopplerSource, make_doppler_source, run_doppler_poll
from _recorder import first_sample_probe
from _soapy import merge_sdr_params, readback_soapy_settings, sdr_ready_fields
from _spawn_contract import (
    await_first_samples,
    build_argparser,
    connect_spawn_sockets,
    frame_received_event,
    load_params,
    run_command_loop,
    send_event,
    watch_engine_death,
)

VERSION = "0.1.0"
_DECODE_PERIOD_S = 2.0
_DEFAULT_SAMPLE_RATE = 2_000_000
# R-11: how long the started stream gets to deliver its FIRST samples before
# the pass fails closed (supervisor ready timeout is 30 s; leave room for the
# gr-soapy open itself).
_FIRST_SAMPLE_TIMEOUT_S = 15.0


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
    # R-18: the shared builder carries frame.id + crc_ok — without them the
    # orchestrator's parser defaulted crc_ok to False and every decoded frame
    # was counted INVALID (never downlink life). Both decode paths here are
    # validity-gated by construction (gr-satellites deframers CRC/FEC-check
    # their protocols; our fallback demods are CRC-gated).
    event = frame_received_event(frame, crc_ok=True)
    event["decoder"] = decoder
    event["satellite"] = satellite
    await send_event(sockets.status_writer, event)
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
    # Doppler v2 (docs/12): the flowgraph OWNS Doppler by POLLING gs-orbitd (``ephem_at``) at
    # --doppler-poll-hz and driving the rotator itself — instead of depending on the orchestrator
    # to push set_doppler over the control socket (a path coupled to orchestrator liveness that
    # broke repeatedly). When NO source resolves (NullDopplerSource) the
    # legacy control-socket push in the engine loop drives Doppler instead. When a source resolves
    # but then DIES mid-pass, run_doppler_poll's fallback_offset takes over the pushed value (still
    # streamed into ``doppler["hz"]`` below) so Doppler doesn't freeze at a stale offset.
    doppler_source = make_doppler_source(
        source=getattr(args, "doppler_source", "orbitd"),
        center_freq_hz=float(args.center_freq_hz or 0.0),
        orbitd_host=getattr(args, "orbitd_host", "127.0.0.1"),
        orbitd_port=int(getattr(args, "orbitd_port", 45400) or 45400),
        orbitd_handle=getattr(args, "orbitd_handle", "") or "",
    )
    doppler_poll = not isinstance(doppler_source, NullDopplerSource)
    poll_period_s = 1.0 / max(1.0, float(getattr(args, "doppler_poll_hz", 25.0) or 25.0))

    if not satellite:
        # R-11 fail-closed: a multi-mission RX with no target is a config
        # error — reporting ready would fake a live pass that captures nothing.
        log.error("no 'satellite' selected (params 'satellite' / --satellite); failing closed")
        await send_event(
            sockets.status_writer,
            {"event": "error", "code": "no-satellite", "detail": "no satellite selected"},
        )
        return 1

    # R-11: build + START the engine BEFORE declaring ready — 'ready' means the
    # SDR opened, the settings applied, the stream is ACTIVE and samples flow.
    # A build/open failure fails the pass here (error event + nonzero exit)
    # instead of being swallowed behind a live-looking command loop.
    try:
        # Import INSIDE the try: a missing dependency (e.g. an un-deployed helper
        # module) must surface as a failed pass, not a silent journal line —
        # which once left a dead pass with no capture and no error.
        from gnuradio_satellites import build_satellites_rx

        # Pre-demod IQ capture is wired inside build_satellites_rx (PassRecorder taps
        # the SDR source; ctx.stop() finalizes) — uniform with the other RX engines.
        ctx = build_satellites_rx(args, satellite, sample_rate, params)
        # RX: start streaming + recording at spawn (arm — before AOS) so the
        # front-end is warm and the whole window is captured. Do NOT gate on
        # cmd:start (that's for TX keying); cmd:start still fires the 'started'
        # status event and cmd:stop ends the pass.
        ctx.start()
    except Exception as e:
        log.exception("gr-satellites: engine failed to start (satellite=%s)", satellite)
        with contextlib.suppress(Exception):
            await send_event(
                sockets.status_writer,
                {"event": "error", "code": "engine-start-failed", "detail": repr(e)},
            )
        return 1
    # First-sample proof off the recorder's unbuffered cf32 (grows the moment
    # the SDR delivers). No recorder → proof unavailable, reported as such.
    probe = first_sample_probe(getattr(ctx, "recorder", None))
    first: bool | None = None
    if probe is not None:
        first = await await_first_samples(probe, timeout_s=_FIRST_SAMPLE_TIMEOUT_S)
        if not first:
            log.error("SDR stream active but delivered no samples — failing closed (R-11)")
            with contextlib.suppress(Exception):
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "error",
                        "code": "engine-no-samples",
                        "detail": f"no samples within {_FIRST_SAMPLE_TIMEOUT_S:.0f}s",
                    },
                )
            with contextlib.suppress(Exception):
                ctx.stop()
            return 1
    decoder = "gr-satellites" if ctx.framing == "grsatellites" else "fallback"
    out_dir = getattr(args, "output_dir", None)

    # R2-02: a recorder-only graph (no decoder could be built) must NOT be reported as an
    # ordinary decode pass. It still runs — the .cf32 is genuinely useful and can be decoded
    # offline — but `decode_built=False` + the reason ride the ready event so gs-client can
    # carry them into the pass result instead of returning a bare, green "completed".
    no_decode_reason = getattr(ctx, "no_decode_reason", "")
    if no_decode_reason:
        log.error("RECORDER-ONLY pass: %s", no_decode_reason)

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "decode_built": not no_decode_reason,
            "no_decode_reason": no_decode_reason,
            # "frames_jsonl" is an explicit key in gs-client's spec_for_data_format
            # map ("frames" was unmapped label drift — it silently fell back to the
            # RAW_BITS spec). Informational here: this app never writes the data
            # socket; its frames product ships via the gr-satellites path.
            "data_format": "frames_jsonl",
            "engine": "gnuradio",
            "decoder": decoder,
            "satellite": satellite,
            "flowgraph_version": VERSION,
            **sdr_ready_fields(
                device=str(args.sdr_args or ""),
                requested=merge_sdr_params(params),
                applied=getattr(ctx, "sdr_applied", None),
                actual=readback_soapy_settings(ctx.src),
                stream_active=True,
                first_samples=first,
            ),
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        started.set()
        await send_event(sockets.status_writer, {"event": "started"})

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        started.set()
        stop_reason["value"] = str(cmd.get("reason", "command"))

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        # Reject bool (isinstance(True, int)) and non-finite NaN/inf (json.loads accepts them) — a
        # bad pushed value must not become the fallback offset that drives the rotator (docs/12).
        if isinstance(off, (int, float)) and not isinstance(off, bool) and math.isfinite(off):
            doppler["hz"] = float(off)

    async def _engine() -> None:  # pragma: no cover (bench)
        # The engine context was built + started (and its stream proven) BEFORE
        # the ready event above (R-11); this task only runs the pass loop.
        log.info(
            "satellite_rx: flowgraph started (sat=%s, decoder=%s); streaming+recording",
            satellite,
            decoder,
        )
        last_doppler = 0.0
        # Flowgraph OWNS Doppler: poll the source and drive ctx.set_doppler at the poll rate,
        # decoupled from the control socket. Runs until stop_requested; a source outage just keeps
        # the last offset (run_doppler_poll never raises). When no source resolved, this is skipped
        # and the legacy control-socket push (below) applies instead. fallback_offset hands the
        # pushed offset (``doppler["hz"]``, kept fresh by _on_set_doppler) to the poll loop if the
        # source dies mid-pass, so a gs-orbitd outage/handle-eviction can't freeze Doppler.
        doppler_task = None
        if doppler_poll:
            doppler_task = asyncio.create_task(
                run_doppler_poll(doppler_source, ctx.set_doppler, stop_requested,
                                 period_s=poll_period_s,
                                 fallback_offset=lambda: doppler["hz"]),
                name="doppler-poll",
            )
            log.info("doppler: flowgraph-owned poll @ %.0f Hz", 1.0 / poll_period_s)
        try:
            while not stop_requested.is_set():
                await asyncio.sleep(_DECODE_PERIOD_S)
                # Legacy control-socket push — ONLY when no poll source resolved (else the poll owns
                # Doppler and this would fight it). Retained so a station without gs-orbitd (or a
                # handle-less pass) still gets orchestrator-pushed Doppler (backward compatible).
                if not doppler_poll and doppler["hz"] != last_doppler:
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
            if doppler_task is not None:
                doppler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await doppler_task
            ctx.stop()
            ctx.wait()

    engine_task = asyncio.create_task(_engine(), name="gr-satellites")
    # R-11: an engine loop that dies mid-pass fails the pass (error event +
    # nonzero exit), never idles behind a live command loop.
    watch_engine_death(engine_task, sockets.status_writer, sockets.control_reader, stop_requested)
    handlers = {"start": _on_start, "stop": _on_stop, "set_doppler": _on_set_doppler}

    async def _shutdown_engine() -> None:
        """Idempotent engine teardown — settle the gr-satellites task fully."""
        stop_requested.set()
        started.set()
        await asyncio.gather(engine_task, return_exceptions=True)

    try:
        reason = await run_command_loop(sockets.control_reader, handlers)
        # P0-08: engine teardown BEFORE the explicit stopped ack; then exit 0.
        # EOF is transport loss — no ack, exit nonzero.
        await _shutdown_engine()
        if reason == "stop":
            await send_event(
                sockets.status_writer,
                {"event": "stopped", "reason": stop_reason["value"]},
            )
            return 0
        log.warning("control EOF without stop — transport loss; exiting nonzero (P0-08)")
        return 1
    finally:
        await _shutdown_engine()
        await sockets.aclose()


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
    # Persist OUR (Python) logs — engine selection, decoded frames, errors — to a per-pass
    # file. stderr is a ring buffer the verbose xtrx/LMS7 C++ driver spam floods, so the
    # decode-relevant lines scroll out before teardown; this file keeps them (and the C++
    # spam never reaches it, so it stays clean + greppable).
    out_dir = getattr(args, "output_dir", "") or ""
    if out_dir:
        with contextlib.suppress(OSError):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(out_dir) / "flowgraph.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
            logging.getLogger().addHandler(fh)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

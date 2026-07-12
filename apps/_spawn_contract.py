"""Shared spawn-contract boilerplate for gs-flowgraphs apps.

Every flowgraph binary the orchestrator spawns must obey Document A
section A.7.2 (CLI flags) + A.7.3 (control / status NDJSON sockets) +
A.7.4 (data socket). This module factors out the boilerplate so the
per-waveform apps (``amateur_fm_narrowband_rx.py``, etc.) can focus on
DSP.

License: GPLv3 (see ``../COPYING``). Imports ``gnuradio`` are
delegated to the per-waveform apps so this helper stays import-safe
on hosts without GR.

The Phase 3 / Phase 5 Python stubs (``stub_rx.py``, ``stub_tx.py``)
have their own minimal copy of this protocol because they pre-date
this helper and the orchestrator's tests pin to their behaviour.
Real flowgraphs use this module.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import itertools
import json
import logging
import queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

CommandHandler = Callable[[dict[str, object]], Awaitable[None]]


class EngineFailure(RuntimeError):
    """An RX/TX engine could not start or prove its stream flows (R-11).

    Raised by app engines so the pass FAILS CLOSED — the death watch (or the
    app's own try) turns it into an ``error`` status event and a nonzero exit,
    instead of a live-looking process that captures nothing."""


# ----------------------------------------------------------------------
# argparse
# ----------------------------------------------------------------------


def build_argparser(*, prog: str, description: str) -> argparse.ArgumentParser:
    """Standard CLI per Document A A.7.2. Per-waveform apps may add
    their own flags after calling this; do NOT change any of the
    declared flags below — the orchestrator builds the argv exactly
    from these names."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--version", action="store_true")
    p.add_argument("--waveform-id", default="")
    p.add_argument("--direction", default="")
    p.add_argument("--center-freq-hz", type=int, default=0)
    p.add_argument("--bandwidth-hz", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=2_000_000)
    p.add_argument("--sdr-driver", default="soapy")
    p.add_argument("--sdr-args", default="")
    p.add_argument("--sdr-port", default="")
    p.add_argument("--control-socket", default="")
    p.add_argument("--status-socket", default="")
    p.add_argument("--data-socket", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--params-file", default="")
    # Pre-demod IQ capture (gs-client [recording]). When set, the engine taps the SDR
    # stream before demod and writes capture artifacts into --output-dir.
    p.add_argument("--record-iq", action="store_true")
    p.add_argument("--record-formats", default="")  # comma list: sdf,csv,png
    # Doppler v2 (docs/12): the flowgraph OWNS Doppler by POLLING gs-orbitd at a fixed cadence
    # (SatNOGS-style), not the orchestrator pushing set_doppler. All optional with safe defaults —
    # an old orchestrator that passes none of these leaves the source unresolved (record-only /
    # legacy control-socket push). --orbitd-handle is the gs-orbitd plan handle the orchestrator
    # materialized for this pass; with no handle the source is record-only.
    p.add_argument("--doppler-source", default="orbitd")  # orbitd | none
    p.add_argument("--orbitd-host", default="127.0.0.1")
    p.add_argument("--orbitd-port", type=int, default=45400)
    p.add_argument("--orbitd-handle", default="")
    p.add_argument("--doppler-poll-hz", type=float, default=25.0)
    return p


# ----------------------------------------------------------------------
# tcp:// URL parsing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TcpEndpoint:
    host: str
    port: int


def parse_tcp_url(url: str) -> TcpEndpoint:
    """Parse ``tcp://host:port`` into ``TcpEndpoint``. Raises on
    malformed input."""
    if not url.startswith("tcp://"):
        msg = f"only tcp:// URLs supported: got {url!r}"
        raise ValueError(msg)
    rest = url[len("tcp://") :]
    if ":" not in rest:
        msg = f"missing port in {url!r}"
        raise ValueError(msg)
    host, port_str = rest.rsplit(":", 1)
    return TcpEndpoint(host=host, port=int(port_str))


# ----------------------------------------------------------------------
# Event encoder + command decoder (NDJSON)
# ----------------------------------------------------------------------


async def send_event(writer: asyncio.StreamWriter, event: dict[str, object]) -> None:
    """Encode + send a single NDJSON event on the status socket. The
    keys are sorted so events are byte-stable for tests / replay
    debugging."""
    line = (json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    writer.write(line)
    await writer.drain()


async def read_command(reader: asyncio.StreamReader) -> dict[str, object] | None:
    """Read one NDJSON command from the control socket. Returns
    ``None`` on EOF; skips malformed lines (with a debug log) rather
    than raising — a single garbled byte shouldn't kill the
    flowgraph mid-pass."""
    while True:
        raw = await reader.readline()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("control socket: ignoring malformed line: %r", raw[:80])
            continue
        if isinstance(obj, dict):
            return obj
        log.debug("control socket: ignoring non-object: %r", obj)


# ----------------------------------------------------------------------
# Data-socket pump
# ----------------------------------------------------------------------


async def pump_data_queue(
    queue_: queue.Queue[bytes | None],
    writer: asyncio.StreamWriter,
) -> None:
    """Pull byte blobs from ``queue_`` and write them to the data
    socket. A ``None`` sentinel ends the pump cleanly.

    The queue is fed by a GR sink running on the GNU Radio worker
    thread; this coroutine runs on the asyncio loop so the socket
    write doesn't block GR's scheduler. Keep the queue bounded
    (constructed in the app) so RX runaway producers don't grow
    memory without limit; queue.Full is logged + dropped, which is
    acceptable for audio (gap shows in the saved file) but should
    be flagged in production.
    """
    loop = asyncio.get_running_loop()
    while True:
        # Run blocking queue.get on a thread so the asyncio loop
        # stays responsive.
        chunk = await loop.run_in_executor(None, queue_.get)
        if chunk is None:
            return
        try:
            writer.write(chunk)
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            log.warning("data socket: peer closed; ending pump")
            return


# ----------------------------------------------------------------------
# Open the three sockets per A.7.2
# ----------------------------------------------------------------------


@dataclass
class SpawnSockets:
    control_reader: asyncio.StreamReader
    control_writer: asyncio.StreamWriter
    status_writer: asyncio.StreamWriter
    data_writer: asyncio.StreamWriter

    async def aclose(self) -> None:
        for w in (self.control_writer, self.status_writer, self.data_writer):
            try:
                w.close()
            except Exception:
                pass


async def connect_spawn_sockets(args: argparse.Namespace) -> SpawnSockets:
    """Open the three sockets the orchestrator pre-opened (it's the
    server; we're the client). Returns on connect or raises ``OSError``
    on timeout."""
    ctrl = parse_tcp_url(args.control_socket)
    status = parse_tcp_url(args.status_socket)
    data = parse_tcp_url(args.data_socket)

    ctrl_reader, ctrl_writer = await asyncio.open_connection(ctrl.host, ctrl.port)
    _status_reader, status_writer = await asyncio.open_connection(status.host, status.port)
    _data_reader, data_writer = await asyncio.open_connection(data.host, data.port)

    return SpawnSockets(
        control_reader=ctrl_reader,
        control_writer=ctrl_writer,
        status_writer=status_writer,
        data_writer=data_writer,
    )


# ----------------------------------------------------------------------
# Command dispatch loop
# ----------------------------------------------------------------------


# Exit reason when a command handler raised: the app must exit NONZERO on this, so the
# supervisor sees a crash rather than a clean stop (audit).
_HANDLER_FAILED = "handler-failed"


async def run_command_loop(
    reader: asyncio.StreamReader,
    handlers: dict[str, CommandHandler],
    status_writer: asyncio.StreamWriter,
    *,
    on_unknown: CommandHandler | None = None,
    terminal_cmds: frozenset[str] = frozenset({"stop"}),
) -> str:
    """Dispatch loop. Each command's ``cmd`` field selects a handler in
    ``handlers``; unknown ``cmd`` values fall through to ``on_unknown`` (or are
    logged + ignored).

    P0-08 exit semantics — returns the exit reason:

    * a name in ``terminal_cmds`` (default ``"stop"``): accepting a stop ENDS
      command dispatch. The app then tears its engine down, emits the explicit
      ``stopped`` event AFTER cleanup, and exits 0 — the supervisor never has
      to force-terminate a healthy stop.
    * ``"eof"``: the control socket closed WITHOUT a stop — that is transport
      loss (a crashed/stopped supervisor), never a clean acknowledgement. The
      app must clean up and exit nonzero.
    """
    while True:
        cmd = await read_command(reader)
        if cmd is None:
            log.info("control socket EOF; exiting command loop (transport loss)")
            return "eof"
        name = cmd.get("cmd")
        if not isinstance(name, str):
            log.debug("control: ignoring command without 'cmd' string: %r", cmd)
            continue
        handler = handlers.get(name)
        if handler is None:
            if on_unknown is not None:
                await on_unknown(cmd)
            else:
                log.debug("control: no handler for cmd=%r; ignoring", name)
            continue
        try:
            await handler(cmd)
        except Exception as e:
            # Audit: this used to log and KEEP DISPATCHING, with no error event and no
            # effect on the exit code. The TX apps do ALL of their transmitting inside
            # these handlers, so a `transmit` that blew up was invisible to gs-client and
            # the pass completed as if it had radiated. The first fix made the error event
            # OPTIONAL (status_writer defaulted to None) and no app passed one — so it
            # stayed silent. It is REQUIRED now, and a failed handler ENDS the loop:
            # the app exits nonzero, the supervisor classifies a real crash, the pass
            # fails. An engine that cannot execute a command must not keep answering.
            log.exception("control: handler for cmd=%r raised", name)
            with contextlib.suppress(Exception):
                await send_event(
                    status_writer,
                    {
                        "event": "error",
                        "code": "handler-failed",
                        "cmd": name,
                        "detail": repr(e),
                    },
                )
            if name not in terminal_cmds:
                # A NON-terminal handler (start, transmit_*) failed: the engine could not
                # do what it was told. End dispatch and let the app exit nonzero so the
                # supervisor sees a crash. A raising STOP handler keeps the P0-08
                # semantics below — dispatch ends and the caller's cleanup owns the rest.
                return _HANDLER_FAILED
        if name in terminal_cmds:
            # Even a raising stop handler ends dispatch — the caller's cleanup
            # path owns the rest (P0-08).
            return name


# ----------------------------------------------------------------------
# R-18: the ONE frame_received event builder
# ----------------------------------------------------------------------

_frame_ids = itertools.count()


def frame_received_event(
    body: bytes,
    *,
    crc_ok: bool,
    framing: str = "",
    frame_id: str = "",
    extra_frame_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a ``frame_received`` status event with the fields the
    orchestrator's typed parser REQUIRES — ``frame.id`` and ``frame.crc_ok``.
    R-18: apps that hand-rolled the event omitted them, so gs-client's parser
    defaulted ``crc_ok`` to False and every valid frame was counted invalid
    (and never registered as downlink life). ``frame_id`` defaults to a
    process-unique sequential id."""
    frame: dict[str, object] = {
        "id": frame_id or f"frm-{next(_frame_ids)}",
        "bytes_b64": base64.b64encode(body).decode("ascii"),
        "len": len(body),
        "crc_ok": bool(crc_ok),
    }
    if extra_frame_fields:
        frame.update(extra_frame_fields)
    event: dict[str, object] = {"event": "frame_received", "frame": frame}
    if framing:
        event["framing"] = framing
    return event


# ----------------------------------------------------------------------
# R-11: first-sample proof + engine death watch
# ----------------------------------------------------------------------


async def await_first_samples(
    probe: Callable[[], int],
    *,
    timeout_s: float,
    poll_s: float = 0.1,
) -> bool:
    """Bounded wait for first-sample proof: poll ``probe`` (a count of
    samples/bytes seen so far — e.g. the recorder's on-disk cf32 size) until it
    goes positive. Returns False on timeout — the SDR opened and 'started' but
    delivered NOTHING, the classic deaf-radio 0-byte-capture failure that must
    fail the pass at spawn, not at LOS."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            n = probe()
        except Exception:  # noqa: BLE001 — a flaky probe reads as "no proof yet"
            n = 0
        if n > 0:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(poll_s)


def watch_engine_death(
    task: asyncio.Task,
    status_writer: asyncio.StreamWriter,
    control_reader: asyncio.StreamReader,
    stop_requested: asyncio.Event,
) -> None:
    """R-11: an engine task that dies mid-pass must FAIL the pass, not linger
    behind a live command loop that keeps answering the orchestrator. On an
    unexpected exception (not a clean return, not during a requested stop) this
    emits an ``error`` status event and feeds EOF to the control reader — the
    command loop exits on the P0-08 transport-loss path and the process exits
    nonzero, so the supervisor classifies a real crash."""

    def _on_done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        # AUDIT ROUND 4: an EngineFailure is the engine DYING — never a clean teardown. It
        # must be reported even when stop_requested is set, because the engine's own
        # teardown sets that flag before it raises (which is exactly how this guard came to
        # swallow a dead SDR: exception raised, flag set, nothing reported).
        if exc is None or (stop_requested.is_set() and not isinstance(exc, EngineFailure)):
            return
        log.error("engine task died: %r — failing the pass (R-11)", exc)

        async def _fail() -> None:
            with contextlib.suppress(Exception):
                await send_event(
                    status_writer,
                    {"event": "error", "code": "engine-died", "detail": repr(exc)},
                )
            control_reader.feed_eof()

        t.get_loop().create_task(_fail())

    task.add_done_callback(_on_done)


# ----------------------------------------------------------------------
# Per-pass parameters (Document C C.5.5: PassDirective RfLink
# waveform_parameters Struct)
# ----------------------------------------------------------------------


def load_params(args: argparse.Namespace) -> dict[str, object]:
    """Load the per-pass parameters file the orchestrator wrote (if
    any), returning the parsed JSON object as a plain dict.

    The file contains the directive's ``RfLink.waveform_parameters``
    Struct serialized via protobuf's ``MessageToDict``. Each waveform
    declares its own schema in ``WaveformEntry.parameters_schema`` —
    this loader is intentionally schema-agnostic so different
    flowgraphs can interpret their own keys (baud rate, deviation,
    decoder choice, etc.) without coordinating with the orchestrator.

    Returns ``{}`` if ``--params-file`` was not passed, the file is
    missing, or the file is malformed (rather than raising — a bad
    params file should let the flowgraph fall back to defaults, not
    crash the pass). Logs a warning on the missing-or-malformed path.
    """
    if not args.params_file:
        return {}
    try:
        with open(args.params_file, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("params file %r not loadable: %s", args.params_file, e)
        return {}
    if not isinstance(obj, dict):
        log.warning(
            "params file %r is not a JSON object (got %s); ignoring",
            args.params_file, type(obj).__name__,
        )
        return {}
    return obj


__all__ = [
    "CommandHandler",
    "EngineFailure",
    "SpawnSockets",
    "TcpEndpoint",
    "await_first_samples",
    "build_argparser",
    "connect_spawn_sockets",
    "frame_received_event",
    "load_params",
    "parse_tcp_url",
    "pump_data_queue",
    "read_command",
    "run_command_loop",
    "send_event",
    "watch_engine_death",
]

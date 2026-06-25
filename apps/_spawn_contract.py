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
import json
import logging
import queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

CommandHandler = Callable[[dict[str, object]], Awaitable[None]]


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


async def run_command_loop(
    reader: asyncio.StreamReader,
    handlers: dict[str, CommandHandler],
    *,
    on_unknown: CommandHandler | None = None,
) -> None:
    """Dispatch loop. Runs until the control socket closes (EOF).
    Each command's ``cmd`` field selects a handler in ``handlers``;
    unknown ``cmd`` values fall through to ``on_unknown`` (or are
    logged + ignored)."""
    while True:
        cmd = await read_command(reader)
        if cmd is None:
            log.info("control socket EOF; exiting command loop")
            return
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
        except Exception:
            log.exception("control: handler for cmd=%r raised", name)


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
    "SpawnSockets",
    "TcpEndpoint",
    "build_argparser",
    "connect_spawn_sockets",
    "load_params",
    "parse_tcp_url",
    "pump_data_queue",
    "read_command",
    "run_command_loop",
    "send_event",
]

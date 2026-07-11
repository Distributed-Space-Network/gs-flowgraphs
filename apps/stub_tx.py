#!/usr/bin/env python3
"""Stub TX flowgraph (Phase 5 placeholder).

Mirror of ``stub_rx.py`` for the TX direction. Honours the Document A
A.7.2 spawn contract + the A.7.3 socket protocol so the orchestrator's
TX path (spawn -> ready -> start -> transmit_started -> stop -> stopped)
can be exercised against a real subprocess without GNU Radio.

Real GR TX flowgraphs (e.g. amateur.fm.narrowband driven by SoapySDR
loopback) land in the same repo; they obey the same CLI contract so the
orchestrator does not change when they replace this stub.

Usage (Document A A.7.2)::

    stub_tx.py --waveform-id amateur.fm.narrowband \\
        --direction tx --center-freq-hz 437500000 --bandwidth-hz 25000 \\
        --sample-rate 48000 --sdr-driver soapy --sdr-args "" --sdr-port TX1 \\
        --control-socket tcp://127.0.0.1:1234 \\
        --status-socket tcp://127.0.0.1:1235 \\
        --data-socket tcp://127.0.0.1:1236 \\
        --output-dir /var/lib/gs/passes/XXX/

Special flags:

  --version   print the stub version and exit 0.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


def _b64_first_bytes(path: Path, *, n: int) -> str:
    """Read up to ``n`` bytes from ``path`` and return base64. Used by
    the stub to leave a tamper-evident summary of what it read for
    tests, without writing arbitrarily large blobs to disk."""
    try:
        with path.open("rb") as f:
            head = f.read(n)
    except OSError:
        return ""
    return base64.b64encode(head).decode("ascii")

VERSION = "0.0.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stub_tx",
        description="Stub TX flowgraph (Phase 5 placeholder)",
    )
    p.add_argument("--version", action="store_true")
    p.add_argument("--waveform-id", default="")
    p.add_argument("--direction", default="tx")
    p.add_argument("--center-freq-hz", type=int, default=0)
    p.add_argument("--bandwidth-hz", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--sdr-driver", default="soapy")
    p.add_argument("--sdr-args", default="")
    p.add_argument("--sdr-port", default="TX1")
    p.add_argument("--control-socket", default="")
    p.add_argument("--status-socket", default="")
    p.add_argument("--data-socket", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--params-file", default="")
    # Stub-only knobs to script the canned event sequence.
    p.add_argument("--frames-to-tx", type=int, default=2)
    # Opt-in periodic emission for whole-pass simulations. 0 disables.
    p.add_argument("--periodic-frame-interval-s", type=float, default=0.0)
    return p.parse_args(argv)


@dataclass
class _Endpoint:
    host: str
    port: int


def parse_tcp_url(url: str) -> _Endpoint:
    """Parse ``tcp://host:port`` into ``_Endpoint``. Raises on malformed input."""
    if not url.startswith("tcp://"):
        msg = f"only tcp:// URLs supported by this stub: got {url!r}"
        raise ValueError(msg)
    rest = url[len("tcp://") :]
    if ":" not in rest:
        msg = f"missing port in {url!r}"
        raise ValueError(msg)
    host, port_str = rest.rsplit(":", 1)
    return _Endpoint(host=host, port=int(port_str))


async def _send_line(writer: asyncio.StreamWriter, obj: dict[str, object]) -> None:
    line = (json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    writer.write(line)
    await writer.drain()


async def _read_command(reader: asyncio.StreamReader) -> object:
    while True:
        raw = await reader.readline()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj


async def amain(args: argparse.Namespace) -> int:
    log = logging.getLogger("stub_tx")
    ctrl = parse_tcp_url(args.control_socket)
    status = parse_tcp_url(args.status_socket)
    data = parse_tcp_url(args.data_socket)

    # Honour --params-file (Document C C.5.5.2). Same diagnostic-echo
    # pattern as stub_rx: write params_received.json into the pass dir.
    received_params: dict[str, object] = {}
    if args.params_file:
        try:
            with open(args.params_file, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                received_params = obj
        except (OSError, json.JSONDecodeError):
            log.exception("could not read params file %r", args.params_file)
    if received_params and args.output_dir:
        out = Path(args.output_dir) / "params_received.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(received_params, f, separators=(",", ":"), sort_keys=True)
        except OSError:
            log.exception("could not write params_received.json to %r", out)

    ctrl_reader, _ctrl_writer = await asyncio.open_connection(ctrl.host, ctrl.port)
    _status_reader, status_writer = await asyncio.open_connection(status.host, status.port)
    _data_reader, data_writer = await asyncio.open_connection(data.host, data.port)

    # Announce ready. TX flowgraphs declare their data_format too even
    # though no audio is captured on the data socket — Phase 7 may use
    # the data socket for a TX-recording / loopback if hardware supports.
    await _send_line(
        status_writer,
        {
            "event": "ready",
            "data_format": "raw_bytes",
            "sample_rate": args.sample_rate,
            "stub_version": VERSION,
            "params_loaded": sorted(received_params),
        },
    )

    started = False
    transmit_announced = False
    bg_tasks: list[asyncio.Task[None]] = []

    async def _periodic_frame() -> None:
        i = 0
        while True:
            await asyncio.sleep(args.periodic_frame_interval_s)
            await _send_line(
                status_writer,
                {
                    "event": "transmit_complete",
                    "frame_id": f"periodic-tx-{i}",
                },
            )
            i += 1

    async def _cancel_bg() -> None:
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        bg_tasks.clear()

    try:
        while True:
            cmd = await _read_command(ctrl_reader)
            if cmd is None:
                break
            assert isinstance(cmd, dict)
            name = cmd.get("cmd")
            if name == "start" and not started:
                started = True
                await _send_line(status_writer, {"event": "started"})
                # Announce that we are actively radiating. The orchestrator
                # uses this to flip the safety FSM from KEYED_READY to KEYED.
                await _send_line(status_writer, {"event": "transmit_started"})
                transmit_announced = True
                # Emit a few transmit_complete events to mimic a real
                # flowgraph cycling through frames.
                for i in range(args.frames_to_tx):
                    await asyncio.sleep(0.02)
                    await _send_line(
                        status_writer,
                        {
                            "event": "transmit_complete",
                            "frame_id": f"stub-tx-{i}",
                        },
                    )
                # Whole-pass periodic emission (opt-in).
                if args.periodic_frame_interval_s > 0:
                    bg_tasks.append(
                        asyncio.create_task(
                            _periodic_frame(), name="stub_tx-periodic-frame",
                        ),
                    )
            elif name == "transmit_frame":
                # Two variants per Document A A.7.3:
                #   - ``bytes_b64``: inline payload (small frames)
                #   - ``payload_file``: absolute path to a pre-staged file
                #                       (used for object-storage-fetched uplink
                #                        payloads — Document C C.5.5.3)
                # The stub doesn't modulate, it just acks with byte count so
                # tests can verify the file actually reached the flowgraph.
                fid_obj = cmd.get("frame_id", "")
                fid = fid_obj if isinstance(fid_obj, str) else ""
                # R-16 contract: a per-burst transmit announces acceptance
                # FIRST (the orchestrator flips KEYED_READY -> KEYED on it),
                # then completes with the accepted count + explicit outcome.
                await _send_line(status_writer, {"event": "transmit_started"})
                bytes_transmitted = 0
                payload_file_obj = cmd.get("payload_file")
                if isinstance(payload_file_obj, str) and payload_file_obj:
                    payload_path = Path(payload_file_obj)
                    try:
                        bytes_transmitted = payload_path.stat().st_size
                        # Echo the path so the test can inspect what we read.
                        # Also write a sidecar so tests can verify the file's
                        # contents without snooping the status stream.
                        sidecar = payload_path.with_name(
                            payload_path.stem + "_tx_seen.json",
                        )
                        sidecar.write_text(
                            json.dumps(
                                {
                                    "frame_id": fid,
                                    "payload_file": str(payload_path),
                                    "bytes_transmitted": bytes_transmitted,
                                    "first_bytes_b64": _b64_first_bytes(
                                        payload_path, n=64,
                                    ),
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    except OSError as e:
                        log.warning(
                            "transmit_frame: payload_file %r unreadable: %s",
                            payload_file_obj, e,
                        )
                await _send_line(
                    status_writer,
                    {
                        "event": "transmit_complete",
                        "frame_id": fid,
                        "bytes_transmitted": bytes_transmitted,
                        "samples": max(1, bytes_transmitted),
                        "outcome": "complete",
                    },
                )
            elif name == "stop":
                await _cancel_bg()
                await _send_line(status_writer, {"event": "stopped", "reason": "command"})
                return 0
            else:
                log.debug("ignoring command: %s", name)
    except (ConnectionError, OSError):
        return 0
    finally:
        # Drop the unused flag rather than emitting a synthetic stopped
        # event — the orchestrator already times out cleanly on ctrl
        # socket close.
        del transmit_announced
        await _cancel_bg()
        for w in (status_writer, data_writer):
            try:
                w.close()
            except Exception:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())

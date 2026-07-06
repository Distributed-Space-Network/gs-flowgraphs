#!/usr/bin/env python3
"""Stub RX flowgraph (Phase 3 placeholder).

This is the first ``--waveform-id`` binary the orchestrator can spawn end to
end. It is NOT a real GNU Radio flowgraph — it does no DSP, has no SDR
connection, and emits canned events. Its sole purpose is to honour the
Document A A.7.2 spawn contract and the A.7.3 socket protocol so that the
orchestrator's flowgraph supervisor, storage tee, and PassResult flow can
be exercised against a real subprocess.

Real GR flowgraphs land in Phase 5. They will be in the same repo and obey
the same CLI contract; the orchestrator will not need to change.

Usage (Document A A.7.2)::

    stub_rx.py --waveform-id <id> \\
        --direction rx --center-freq-hz 437800000 --bandwidth-hz 25000 \\
        --sample-rate 48000 --sdr-driver soapy --sdr-args "" --sdr-port RX1 \\
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


VERSION = "0.0.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stub_rx", description="Stub RX flowgraph (Phase 3 placeholder)")
    p.add_argument("--version", action="store_true")
    p.add_argument("--waveform-id", default="")
    p.add_argument("--direction", default="rx")
    p.add_argument("--center-freq-hz", type=int, default=0)
    p.add_argument("--bandwidth-hz", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--sdr-driver", default="soapy")
    p.add_argument("--sdr-args", default="")
    p.add_argument("--sdr-port", default="RX1")
    p.add_argument("--control-socket", default="")
    p.add_argument("--status-socket", default="")
    p.add_argument("--data-socket", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--params-file", default="")
    p.add_argument("--record-iq", action="store_true")
    p.add_argument("--record-formats", default="")
    p.add_argument("--signal-events", type=int, default=3)
    p.add_argument("--audio-chunks", type=int, default=4)
    p.add_argument("--audio-chunk-bytes", type=int, default=512)
    # Opt-in periodic emission for whole-pass simulations. 0 disables.
    # Existing tests leave these at zero and observe the legacy fixed
    # burst above; long-running E2E tests set non-zero intervals so the
    # stub keeps producing telemetry, frames, and audio for the full
    # pass duration until ``stop`` arrives.
    p.add_argument("--periodic-signal-interval-s", type=float, default=0.0)
    p.add_argument("--periodic-frame-interval-s", type=float, default=0.0)
    p.add_argument("--periodic-audio-interval-s", type=float, default=0.0)
    # TOLERATE unknown spawn-contract flags: this stub hand-mirrors the A.7.2 flag list rather
    # than importing build_argparser, so a NEW orchestrator flag (e.g. Doppler v2's
    # --doppler-source / --orbitd-*) would otherwise argparse-crash the stub (rc=2)
    # and fail every e2e pass. A stub need not model every flag — accept and ignore the extras.
    ns, extra = p.parse_known_args(argv)
    if extra:
        logging.getLogger("stub_rx").debug("stub_rx: ignoring unmodeled spawn flags: %r", extra)
    return ns


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


async def _read_commands(reader: asyncio.StreamReader) -> object:
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


def _write_stub_capture(args: argparse.Namespace) -> None:
    """Synthetic pre-demod cf32 capture so ``--record-iq`` exercises the record path
    end-to-end without hardware (the stub has no real SDR). The view artifacts
    (SDF/CSV/PNG) are derived post-pass by iq_views, like the real engines. Lazy imports
    keep numpy off the default stub path; the seed makes it deterministic."""
    from _recorder import StreamRecorder  # noqa: PLC0415 — lazy: only when capturing
    import numpy as np  # noqa: PLC0415

    rec = StreamRecorder.maybe_start(args, sample_rate_hz=float(args.sample_rate or 48000))
    if rec is None:
        return
    fs = float(args.sample_rate or 48000)
    n = int(fs * 0.5)  # ~0.5 s of synthetic capture
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    iq = (
        0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        + 0.3 * np.exp(2j * np.pi * 1200.0 * t)
    ).astype(np.complex64)
    rec.write(iq)
    rec.close()


async def amain(args: argparse.Namespace) -> int:
    log = logging.getLogger("stub_rx")
    ctrl = parse_tcp_url(args.control_socket)
    status = parse_tcp_url(args.status_socket)
    data = parse_tcp_url(args.data_socket)

    # Honour --params-file (Document C C.5.5.2). The stub does no DSP
    # so it can't *act* on the params, but it writes a copy of the
    # parsed dict into the pass output dir so E2E tests can verify the
    # orchestrator → supervisor → flowgraph plumbing end-to-end.
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

    # Pre-demod IQ capture (uniform with the real RX engines). Synthetic, since the
    # stub has no SDR — exercises the SDF/CSV/PNG path for orchestrator E2E.
    if args.record_iq:
        try:
            _write_stub_capture(args)
        except Exception:
            log.exception("stub capture failed")

    ctrl_reader, _ctrl_writer = await asyncio.open_connection(ctrl.host, ctrl.port)
    status_reader_unused, status_writer = await asyncio.open_connection(status.host, status.port)
    data_reader_unused, data_writer = await asyncio.open_connection(data.host, data.port)
    del status_reader_unused, data_reader_unused

    await _send_line(
        status_writer,
        {
            "event": "ready",
            "data_format": "audio_ogg",
            "sample_rate": args.sample_rate,
            "stub_version": VERSION,
            "params_loaded": sorted(received_params),
        },
    )

    started = False
    bg_tasks: list[asyncio.Task[None]] = []
    audio_chunk_payload = bytes([0x55, 0xAA] * (args.audio_chunk_bytes // 2))

    async def _periodic_signal() -> None:
        i = 0
        while True:
            await asyncio.sleep(args.periodic_signal_interval_s)
            await _send_line(
                status_writer,
                {
                    "event": "signal",
                    # Walk the RSSI/SNR a bit so the time-series varies
                    # rather than emitting identical samples.
                    "rssi_dbm": -82.0 + float(i % 7),
                    "snr_db": 8.0 + float(i % 5),
                    "lock": True,
                },
            )
            i += 1

    async def _periodic_frame() -> None:
        i = 0
        while True:
            await asyncio.sleep(args.periodic_frame_interval_s)
            await _send_line(
                status_writer,
                {
                    "event": "frame_received",
                    "frame": {
                        "id": f"periodic-{i}",
                        "bytes_b64": base64.b64encode(
                            f"periodic-frame-{i}".encode(),
                        ).decode("ascii"),
                        "crc_ok": True,
                    },
                },
            )
            i += 1

    async def _periodic_audio() -> None:
        while True:
            await asyncio.sleep(args.periodic_audio_interval_s)
            data_writer.write(audio_chunk_payload)
            await data_writer.drain()

    async def _cancel_bg() -> None:
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        bg_tasks.clear()

    try:
        while True:
            cmd = await _read_commands(ctrl_reader)
            if cmd is None:
                break
            assert isinstance(cmd, dict)
            name = cmd.get("cmd")
            if name == "start" and not started:
                started = True
                await _send_line(status_writer, {"event": "started"})

                for i in range(args.signal_events):
                    await asyncio.sleep(0.05)
                    await _send_line(
                        status_writer,
                        {
                            "event": "signal",
                            "rssi_dbm": -80.0 + float(i),
                            "snr_db": 10.0 + float(i),
                            "lock": True,
                        },
                    )

                # Write canned audio bytes through the data socket so the
                # orchestrator's storage tee has something to record.
                for _ in range(args.audio_chunks):
                    data_writer.write(audio_chunk_payload)
                    await data_writer.drain()
                    await asyncio.sleep(0.01)

                # Emit one decoded frame so DECODE telemetry has content.
                await _send_line(
                    status_writer,
                    {
                        "event": "frame_received",
                        "frame": {
                            "id": "stub-1",
                            "bytes_b64": base64.b64encode(b"hello").decode("ascii"),
                            "crc_ok": True,
                        },
                    },
                )

                # Whole-pass periodic emission: started after the initial
                # burst, runs until ``stop`` cancels them. Backwards-
                # compatible — intervals default to 0.0 (disabled).
                if args.periodic_signal_interval_s > 0:
                    bg_tasks.append(
                        asyncio.create_task(
                            _periodic_signal(), name="stub_rx-periodic-signal",
                        ),
                    )
                if args.periodic_frame_interval_s > 0:
                    bg_tasks.append(
                        asyncio.create_task(
                            _periodic_frame(), name="stub_rx-periodic-frame",
                        ),
                    )
                if args.periodic_audio_interval_s > 0:
                    bg_tasks.append(
                        asyncio.create_task(
                            _periodic_audio(), name="stub_rx-periodic-audio",
                        ),
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
        # Print on stdout per the orchestrator's --version probe.
        print(VERSION)
        return 0
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())

"""P0-08 regressions: accepting stop ENDS command dispatch; EOF is transport
loss, never a clean acknowledgement."""

from __future__ import annotations

import asyncio
import json

from _spawn_contract import run_command_loop


def _reader_with(*commands: dict, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    for cmd in commands:
        reader.feed_data(json.dumps(cmd).encode() + b"\n")
    if eof:
        reader.feed_eof()
    return reader


def test_stop_ends_dispatch_even_with_more_commands_queued():
    async def run() -> tuple[str, list[str]]:
        seen: list[str] = []

        async def handler(cmd: dict) -> None:
            seen.append(str(cmd.get("cmd")))

        reader = _reader_with(
            {"cmd": "start"},
            {"cmd": "stop"},
            {"cmd": "start"},  # must never be dispatched (P0-08)
        )
        reason = await run_command_loop(
            reader, {"start": handler, "stop": handler}
        )
        return reason, seen

    reason, seen = asyncio.run(run())
    assert reason == "stop"
    assert seen == ["start", "stop"], "dispatch continued past the accepted stop"


def test_eof_without_stop_reports_transport_loss():
    async def run() -> str:
        return await run_command_loop(_reader_with({"cmd": "start"}), {})

    assert asyncio.run(run()) == "eof"


def test_raising_stop_handler_still_ends_dispatch():
    async def run() -> str:
        async def boom(_cmd: dict) -> None:
            raise RuntimeError("stop handler died")

        return await run_command_loop(_reader_with({"cmd": "stop"}), {"stop": boom})

    assert asyncio.run(run()) == "stop"

#!/usr/bin/env python3
"""Derive the bounded operator-preview telemetry sidecar from frames.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from native_telemetry.output import derive_preview

log = logging.getLogger("telemetry_preview")


def _norad(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("NORAD id must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("NORAD id must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="immutable frames.jsonl input")
    parser.add_argument("--output", type=Path, help="default: telemetry_preview.jsonl beside input")
    parser.add_argument("--norad-id", type=_norad)
    parser.add_argument("--framing", default="", help="pass framing when a frame omits it")
    args = parser.parse_args(argv)
    output = args.output or args.frames.with_name("telemetry_preview.jsonl")
    try:
        summary = derive_preview(
            args.frames,
            output,
            norad_id=args.norad_id,
            pass_framing=args.framing,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        log.exception("telemetry preview derivation failed")
        return 1
    print(json.dumps(summary.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

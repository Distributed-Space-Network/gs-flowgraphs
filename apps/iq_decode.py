#!/usr/bin/env python3
"""Post-pass decoder: run the full demod bank over a recorded ``.cf32`` and write frames.

This is the exhaustive-decode counterpart to ``iq_views`` (which derives PNG/CSV/SDF). It
runs AFTER the pass, off the recorded IQ, so the broad multi-modulation search never has to
keep up with the SDR in real time — which on a constrained SoC overran the RX DMA (BUF_OVF)
and starved Doppler retuning. The real-time graph now runs only the backend's targeted mode;
everything else happens here, with no deadline.

For each CRC-valid frame recovered it appends a record to ``<dir>/frames.jsonl`` carrying the
**raw bytes** (always) plus a best-effort **upper-layer** summary (AX.25 addresses) — the
shape the uplink/uplink-response bucket will need (raw + decoded-if-possible).

The DSP (``gnuradio_satellites.decode_file``) needs GNU Radio + gr-satellites, so it is
imported lazily inside ``main`` — keeping the record/parse helpers importable (and unit
tested) off the bench, like ``satellite_rx`` does with ``build_satellites_rx``.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

VERSION = "0.1.0"


def read_sidecar_rate(cf32_path: Path, fallback_hz: float) -> float:
    """The true sample rate of ``cf32_path`` from its ``<file>.cf32.json`` sidecar (a
    high-baud bird widens the channel, so the recorded rate may differ from the orchestrator
    ``--sample-rate``). Falls back to ``fallback_hz`` if the sidecar is missing/unreadable."""
    sidecar = Path(str(cf32_path) + ".json")
    with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        rate = float(meta.get("sample_rate_hz") or 0.0)
        if rate > 0:
            return rate
    return float(fallback_hz)


def read_params(params_file: str | Path | None) -> tuple[str, dict]:
    """``(satellite, params)`` from a ``params.json`` (the backend's per-pass mode). Empty /
    missing → ("", {}). Used to seed the targeted demods + the gr-satellites NORAD selector."""
    if not params_file:
        return "", {}
    with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
        params = json.loads(Path(params_file).read_text(encoding="utf-8"))
        if isinstance(params, dict):
            return str(params.get("satellite", "") or ""), params
    return "", {}


def ax25_summary(frame: bytes) -> dict | None:
    """Best-effort AX.25 upper-layer summary (dest/src callsigns, control, PID). Returns None
    when the bytes don't look like AX.25 — gr-satellites frames are already higher-layer, and
    a non-AX.25 fallback frame simply has no summary. The raw bytes are always kept."""
    if len(frame) < 15:
        return None

    def _callsign(addr: bytes) -> str | None:
        chars = []
        for c in addr[:6]:
            ch = (c >> 1) & 0x7F
            if ch == 0x20:  # space pad
                continue
            if not (0x30 <= ch <= 0x39 or 0x41 <= ch <= 0x5A):  # A-Z0-9 only
                return None
            chars.append(chr(ch))
        return "".join(chars) or None

    def _addr(addr: bytes) -> str | None:
        call = _callsign(addr)
        if call is None:
            return None
        ssid = (addr[6] >> 1) & 0x0F
        return f"{call}-{ssid}" if ssid else call

    dest, src = _addr(frame[0:7]), _addr(frame[7:14])
    if dest is None or src is None:
        return None
    # End of the address field: the byte whose LSB is set (7-byte aligned).
    end = 0
    for i in range(6, min(len(frame), 70), 7):
        if frame[i] & 0x01:
            end = i + 1
            break
    if end == 0 or end >= len(frame):
        return None
    return {
        "dest": dest,
        "src": src,
        "control": frame[end],
        "pid": frame[end + 1] if end + 1 < len(frame) else None,
    }


def frame_record(demod: str, frame: bytes, *, ts: float | None = None) -> dict:
    """One ``frames.jsonl`` record for a post-pass frame: raw bytes (hex) always, plus the
    AX.25 summary when parseable. Schema-compatible with the live ``satellite_rx`` records
    (``decoder``/``len``/``raw_hex``/``deframed_hex``) so a single file feeds the uploader."""
    rec = {
        "ts": ts if ts is not None else time.time(),
        "phase": "postpass",
        "decoder": demod,
        "len": len(frame),
        "raw_hex": frame.hex(),
        "deframed_hex": frame.hex(),
    }
    ax25 = ax25_summary(frame)
    if ax25 is not None:
        rec["ax25"] = ax25
    return rec


def write_frames(jsonl_path: Path, records: list[dict]) -> int:
    """Append ``records`` to ``jsonl_path`` (one JSON object per line). Returns the count
    written. Appends so post-pass frames join any live ones in the same pass file."""
    if not records:
        return 0
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="iq_decode", description="Post-pass decode of a .cf32")
    ap.add_argument("--input", required=True, help="recorded .cf32 (whole pass)")
    ap.add_argument("--params-file", default="", help="params.json (backend per-pass mode)")
    ap.add_argument("--sample-rate", type=float, default=48000.0, help="fallback if no sidecar")
    ap.add_argument("--satellite", default="", help="NORAD id/name (else from params.json)")
    ap.add_argument("--output", default="", help="frames.jsonl path (default <dir>/frames.jsonl)")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0

    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("iq_decode")

    cf32 = Path(args.input)
    if not cf32.exists() or cf32.stat().st_size < 8:  # < one complex64 → nothing to do
        log.info("iq_decode: no usable cf32 at %s; nothing to decode", cf32)
        return 0
    params_file = args.params_file or (cf32.parent / "params.json")
    sat_from_params, params = read_params(params_file)
    satellite = args.satellite or sat_from_params
    rate = read_sidecar_rate(cf32, args.sample_rate)
    out = Path(args.output) if args.output else cf32.parent / "frames.jsonl"

    # DSP is bench-only (GNU Radio + gr-satellites): import lazily so this tool stays
    # importable (and testable) off the bench.
    from gnuradio_satellites import decode_file  # noqa: PLC0415

    decoded = decode_file(cf32, sample_rate_hz=rate, satellite=satellite, params=params)
    n = write_frames(out, [frame_record(demod, fr) for demod, fr in decoded])
    log.info("iq_decode: %d frame(s) from %s → %s", n, cf32.name, out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

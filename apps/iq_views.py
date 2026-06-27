#!/usr/bin/env python3
"""Derive view artifacts (waterfall PNG, VSA CSV) from a recorded ``.cf32`` capture.

Run AFTER the pass, decoupled from the flowgraph's stop path — so it has free CPU, no
30 s stop budget, and no gr-soapy teardown to fight. gs-client invokes this once the
flowgraph has exited (see the flowgraph supervisor); the ``.cf32`` is then final.

Pure numpy — no GNU Radio — so it runs anywhere and is unit-testable on the dev box. The
heavy lifting (spectrogram, VSA CSV) is reused from ``_recorder``.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
from _recorder import write_vsa_csv, write_waterfall_png

log = logging.getLogger("iq_views")


def derive_views(
    cf32: str | Path,
    *,
    center_hz: float,
    sample_rate_hz: float,
    formats: tuple[str, ...],
    csv_seconds: float = 30.0,
) -> list[Path]:
    """Write the requested views next to ``cf32``. Returns the paths written.

    The waterfall PNG spans the whole pass (the spectrogram bounds its own row count, and
    we memmap so only the FFT windows are read). The VSA CSV is a leading ``csv_seconds``
    window (a full-pass CSV is GB-scale text; the complete IQ lives in the .cf32)."""
    path = Path(cf32)
    want_png = "png" in formats
    want_csv = "csv" in formats
    if not (want_png or want_csv):
        return []
    if not path.exists():
        log.warning("iq_views: %s not found", path)
        return []
    # Prefer the recording's own metadata sidecar (the TRUE rate/centre the engine used —
    # it may have widened the channel for a high-baud bird) over the passed-in args.
    meta = path.with_name(path.name + ".json")
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            sample_rate_hz = float(d.get("sample_rate_hz", sample_rate_hz))
            center_hz = float(d.get("center_hz", center_hz))
        except (OSError, ValueError, TypeError):
            log.warning("iq_views: ignoring unreadable sidecar %s", meta.name)
    n_samp = path.stat().st_size // 8  # 8 B/complex64; floor a torn write to whole samples
    if n_samp < 1:
        log.warning("iq_views: %s has no samples", path)
        return []

    started = _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.UTC)
    iq = np.memmap(path, dtype=np.complex64, mode="r", shape=(n_samp,))
    written: list[Path] = []
    if want_png:
        png = path.with_suffix(".png")
        write_waterfall_png(png, iq)
        written.append(png)
    if want_csv:
        n = max(1, int(sample_rate_hz * csv_seconds))
        csv = path.with_suffix(".csv")
        write_vsa_csv(
            csv,
            np.asarray(iq[:n]),
            center_hz=center_hz,
            sample_rate_hz=sample_rate_hz,
            started_utc=started,
        )
        written.append(csv)
    log.info(
        "iq_views: derived %s from %s (%d samples)",
        ", ".join(p.suffix.lstrip(".") for p in written),
        path.name,
        n_samp,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="iq_views", description="Derive PNG/CSV views from a recorded .cf32 capture."
    )
    p.add_argument("--input", required=True, help="path to the .cf32 capture")
    p.add_argument("--center-hz", type=float, default=0.0, help="capture centre frequency, Hz")
    p.add_argument("--sample-rate", type=float, required=True, help="capture sample rate, Hz")
    p.add_argument("--formats", default="png,csv", help="comma list, subset of png,csv")
    p.add_argument(
        "--csv-seconds", type=float, default=30.0, help="leading window (s) for the VSA CSV"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    fmts = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    try:
        derive_views(
            args.input,
            center_hz=args.center_hz,
            sample_rate_hz=args.sample_rate,
            formats=fmts,
            csv_seconds=args.csv_seconds,
        )
    except Exception:
        log.exception("iq_views: failed to derive views from %s", args.input)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

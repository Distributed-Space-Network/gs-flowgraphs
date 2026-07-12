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
from _recorder import iq_to_sdf_bytes, write_vsa_csv, write_waterfall_png

log = logging.getLogger("iq_views")
_SDF_CHUNK = 1 << 20  # samples per chunk when transcoding cf32 → SDF (bounded memory)


def derive_views(
    cf32: str | Path,
    *,
    center_hz: float,
    sample_rate_hz: float,
    formats: tuple[str, ...],
    csv_seconds: float = 30.0,
) -> list[Path]:
    """Write the requested views next to ``cf32``. Returns the paths written.

    PNG = whole-pass waterfall (the spectrogram bounds its row count; we memmap so only the
    FFT windows are read). SDF = whole-pass Keysight int16 transcode of the cf32 (chunked).
    CSV = a leading ``csv_seconds`` VSA window (a full-pass CSV is GB-scale text; the
    complete IQ lives in the .cf32 / .sdf)."""
    path = Path(cf32)
    want_png = "png" in formats
    want_csv = "csv" in formats
    want_sdf = "sdf" in formats
    if not (want_png or want_csv or want_sdf):
        return []
    if not path.exists():
        log.warning("iq_views: %s not found", path)
        return []
    # Prefer the recording's own metadata sidecar (the TRUE rate/centre the engine used —
    # it may have widened the channel for a high-baud bird) over the passed-in args.
    meta = path.with_name(path.name + ".json")
    metadata_trusted = False
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            sample_rate_hz = float(d.get("sample_rate_hz", sample_rate_hz))
            center_hz = float(d.get("center_hz", center_hz))
            metadata_trusted = True
        except (OSError, ValueError, TypeError):
            log.warning("iq_views: ignoring unreadable sidecar %s", meta.name)
    if not metadata_trusted:
        # R2-20: without the sidecar, the rate and centre are the ORCHESTRATOR'S REQUEST —
        # not what the engine actually used. They differ whenever the channel was widened for
        # a high-baud bird, and every derived artifact (waterfall axes, VSA CSV, SDF) is then
        # MISLABELLED. The views are still worth producing — but they must SAY the axes are
        # unverified, because an operator reading a confidently-labelled frequency axis that
        # is simply wrong is worse off than one who knows the scale is a guess.
        log.error(
            "iq_views: no usable sidecar for %s — the true sample rate/centre are UNKNOWN. "
            "Deriving views with GUESSED axes (%.0f Hz @ %.0f Hz); they are labelled "
            "UNVERIFIED. The .cf32 itself is intact.",
            path.name, sample_rate_hz, center_hz,
        )
    n_samp = path.stat().st_size // 8  # 8 B/complex64; floor a torn write to whole samples
    if n_samp < 1:
        log.warning("iq_views: %s has no samples", path)
        return []

    started = _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.UTC)
    iq = np.memmap(path, dtype=np.complex64, mode="r", shape=(n_samp,))
    written: list[Path] = []
    if want_png:
        png = path.with_suffix(".png")
        title = path.stem if metadata_trusted else f"{path.stem}  [AXES UNVERIFIED: no sidecar]"
        write_waterfall_png(
            png, iq, sample_rate_hz=sample_rate_hz, center_hz=center_hz, title=title)
        # R2-21: write_waterfall_png SKIPS a capture shorter than one FFT window (it refuses
        # to fabricate an all-zeros image). Reporting the path anyway told the caller a
        # product exists when nothing was written — a phantom artifact.
        if png.exists():
            written.append(png)
        else:
            log.warning(
                "iq_views: no waterfall for %s (capture too short for one FFT window) — "
                "NOT reporting a PNG that does not exist", path.name,
            )
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
    if want_sdf:  # whole-pass Keysight int16 transcode, chunked so memory stays bounded
        sdf = path.with_suffix(".sdf")
        with sdf.open("wb") as fh:
            for off in range(0, n_samp, _SDF_CHUNK):
                fh.write(iq_to_sdf_bytes(np.asarray(iq[off : off + _SDF_CHUNK])))
        written.append(sdf)
    log.info(
        "iq_views: derived %s from %s (%d samples)",
        ", ".join(p.suffix.lstrip(".") for p in written),
        path.name,
        n_samp,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="iq_views", description="Derive SDF/PNG/CSV views from a recorded .cf32 capture."
    )
    p.add_argument("--input", required=True, help="path to the .cf32 capture")
    p.add_argument("--center-hz", type=float, default=0.0, help="capture centre frequency, Hz")
    p.add_argument("--sample-rate", type=float, required=True, help="capture sample rate, Hz")
    p.add_argument("--formats", default="sdf,png,csv", help="comma list, subset of sdf,png,csv")
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

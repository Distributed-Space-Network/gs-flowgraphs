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
import contextlib
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from _recorder import iq_to_sdf_bytes, write_vsa_csv, write_waterfall_png

log = logging.getLogger("iq_views")
_SDF_CHUNK = 1 << 20  # samples per chunk when transcoding cf32 → SDF (bounded memory)
_OGG_CHUNK = 1 << 20  # samples per chunk when deriving discriminator audio (bounded memory)
OGG_SAMPLE_RATE_HZ = 48_000  # SatNOGS discriminator-audio rate; audio_analyze.py expects it
# The OGG stream is written to ffmpeg as we walk the capture, so encoding overlaps the read;
# communicate() only awaits the final flush. The parent (gs-client supervisor) bounds the whole
# iq_views run, so this is just a standalone-run backstop against a wedged encoder.
_OGG_ENCODE_FLUSH_TIMEOUT_S = 120.0


def _unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _kill_and_reap(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the encoder and REAP it (bounded), so a cancelled/failed encode never orphans an
    ffmpeg child or leaves its pipes open. ``communicate`` (not bare ``wait``) drains stdout/stderr
    so the kill cannot deadlock on a full pipe; every step is best-effort under a short timeout."""
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.communicate(timeout=5.0)


def _discriminator_chunk(
    block: np.ndarray, prev: np.complex64 | None
) -> tuple[np.ndarray, np.complex64 | None]:
    """FM-discriminator for one chunk of IQ, phase-continuous with the previous chunk.

    ``discriminator[n] = angle(iq[n] * conj(iq[n-1])) / pi``. ``prev`` is the LAST IQ sample of
    the previous chunk (or ``None`` for the very first chunk); it becomes the predecessor of this
    chunk's first sample, so the phase is never reset at a chunk boundary. The first sample of the
    whole capture has no predecessor and is defined as ``0.0``. Returns
    ``(mono float32 discriminator for this chunk, new predecessor)``."""
    if block.size == 0:
        return np.empty(0, dtype=np.float32), prev
    if prev is None:
        disc = np.empty(block.size, dtype=np.float32)
        disc[0] = 0.0
        if block.size > 1:
            disc[1:] = np.angle(block[1:] * np.conj(block[:-1])) / np.pi
    else:
        extended = np.empty(block.size + 1, dtype=np.complex64)
        extended[0] = prev
        extended[1:] = block
        disc = (np.angle(extended[1:] * np.conj(extended[:-1])) / np.pi).astype(np.float32)
    return disc, np.complex64(block[-1])


def _discriminator(iq: np.ndarray, *, chunk_samples: int = _OGG_CHUNK) -> np.ndarray:
    """Whole-capture discriminator via the SAME chunked path the OGG encoder feeds ffmpeg.

    Exposed for tests: for ANY chunk size the result must equal the single-shot whole-array
    computation (phase continuity across boundaries is the correctness property)."""
    iq = np.asarray(iq, dtype=np.complex64)
    prev: np.complex64 | None = None
    parts: list[np.ndarray] = []
    for off in range(0, iq.shape[0], max(1, chunk_samples)):
        disc, prev = _discriminator_chunk(iq[off : off + chunk_samples], prev)
        parts.append(disc)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _ffmpeg_ogg_argv(
    ffmpeg: str, input_rate_hz: float, ogg_channels: int, out_path: Path
) -> list[str]:
    """Build the ffmpeg argv: read mono float32 discriminator from stdin at the capture's TRUE
    rate, resample to 48 kHz, encode Vorbis to ``out_path``. For ``ogg_channels == 2`` the single
    mono signal is EXPLICITLY duplicated into both output channels via a pan filter (a deterministic
    mapping, NOT the implicit mono->stereo upmix); stereo carries no extra information.

    The OUTPUT muxer is pinned with an explicit ``-f ogg`` because the file is written to a
    ``.ogg.tmp`` name from which ffmpeg CANNOT infer the container — without it ffmpeg errors out
    (or, worse, guesses a different muxer). ``-f f32le`` earlier is the INPUT demuxer, unrelated."""
    rate = max(1, int(round(input_rate_hz)))
    argv = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
    ]
    if ogg_channels == 2:
        argv += ["-af", "pan=stereo|c0=c0|c1=c0"]
    argv += [
        "-ar", str(OGG_SAMPLE_RATE_HZ), "-ac", str(ogg_channels),
        "-c:a", "libvorbis", "-f", "ogg", str(out_path),
    ]
    return argv


def write_discriminator_ogg(
    cf32_path: Path,
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    ogg_channels: int = 1,
    ffmpeg: str | None = None,
) -> Path | None:
    """Derive FM-discriminator audio from ``iq`` and encode it as a 48 kHz Vorbis ``<pass>.ogg``.

    The mono float32 discriminator is streamed to ONE quiet ffmpeg process in bounded chunks (phase
    continuity preserved across boundaries). ffmpeg is told the true input rate, resamples to 48 kHz
    and encodes Vorbis. The output is written to ``<pass>.ogg.tmp`` and renamed atomically to
    ``<pass>.ogg`` ONLY after ffmpeg exits 0; the temp file is removed on cancellation or failure,
    so a partial file never wears the final name. Returns the final path, or ``None`` when nothing
    was written (ffmpeg missing, bad channel count, too-short capture, or encode failure)."""
    if ogg_channels not in (1, 2):
        log.error("iq_views: ogg_channels must be 1 or 2, got %r — skipping OGG", ogg_channels)
        return None
    exe = ffmpeg or shutil.which("ffmpeg")
    if exe is None:
        log.warning("iq_views: ffmpeg not found on PATH — skipping OGG discriminator audio")
        return None
    n = int(np.asarray(iq).shape[0])
    if n < 2:
        log.warning("iq_views: capture too short (%d samples) for discriminator audio", n)
        return None
    final = cf32_path.with_suffix(".ogg")
    tmp = cf32_path.with_suffix(".ogg.tmp")
    _unlink_quiet(tmp)  # a stale temp from a crashed run must never be renamed as the final file
    argv = _ffmpeg_ogg_argv(exe, sample_rate_hz, ogg_channels, tmp)
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except OSError as e:
        log.warning("iq_views: could not start ffmpeg for OGG: %s", e)
        _unlink_quiet(tmp)
        return None
    assert proc.stdin is not None
    try:
        prev: np.complex64 | None = None
        for off in range(0, n, _OGG_CHUNK):
            block = np.ascontiguousarray(iq[off : off + _OGG_CHUNK], dtype=np.complex64)
            disc, prev = _discriminator_chunk(block, prev)
            proc.stdin.write(np.ascontiguousarray(disc, dtype="<f4").tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        # ffmpeg exited early (e.g. libvorbis missing) — reap it and report via the return code.
        with contextlib.suppress(Exception):
            proc.stdin.close()
    except BaseException:
        # Cancellation (KeyboardInterrupt / CancelledError / SystemExit) or any unexpected error
        # mid-stream: kill AND reap the encoder, drop the temp, never leave a partial file.
        _kill_and_reap(proc)
        _unlink_quiet(tmp)
        raise
    try:
        _, err = proc.communicate(timeout=_OGG_ENCODE_FLUSH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_and_reap(proc)
        _unlink_quiet(tmp)
        log.warning("iq_views: ffmpeg OGG encode timed out — removed %s", tmp.name)
        return None
    except BaseException:
        # Cancellation during the final flush must also kill+reap and clean up before propagating.
        _kill_and_reap(proc)
        _unlink_quiet(tmp)
        raise
    # A truthful success requires ALL of: clean exit, a temp file, and NON-EMPTY output. ffmpeg can
    # exit 0 yet write nothing/an empty file (no samples reached the muxer); reporting that as a
    # produced artifact is a false success, so an empty temp is a failure just like a non-zero exit.
    tmp_ok = tmp.exists() and tmp.stat().st_size > 0
    if proc.returncode != 0 or not tmp_ok:
        _unlink_quiet(tmp)
        detail = (err or b"").decode("utf-8", errors="replace").strip()[:500]
        log.warning(
            "iq_views: ffmpeg OGG encode failed (rc=%s, output=%s): %s",
            proc.returncode, "present" if tmp_ok else "missing/empty", detail,
        )
        return None
    try:
        os.replace(tmp, final)
    except OSError as e:
        _unlink_quiet(tmp)
        log.warning("iq_views: could not finalize OGG %s: %s", final.name, e)
        return None
    # Only NOW, with the final file confirmed present and non-empty, is success truthful.
    if not (final.exists() and final.stat().st_size > 0):
        _unlink_quiet(final)
        log.warning("iq_views: OGG %s missing/empty after finalize — reporting failure", final.name)
        return None
    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0
    log.info(
        "iq_views: derived OGG %s (%.1fs, %d ch, %d bytes)",
        final.name, duration_s, ogg_channels, final.stat().st_size,
    )
    return final


def derive_views(
    cf32: str | Path,
    *,
    center_hz: float,
    sample_rate_hz: float,
    formats: tuple[str, ...],
    csv_seconds: float = 30.0,
    ogg_channels: int = 1,
    ffmpeg: str | None = None,
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
    want_ogg = "ogg" in formats
    if not (want_png or want_csv or want_sdf or want_ogg):
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
    if want_ogg:  # optional FM-discriminator audio (48 kHz Vorbis), same selection policy as above
        ogg = write_discriminator_ogg(
            path, iq, sample_rate_hz=sample_rate_hz, ogg_channels=ogg_channels, ffmpeg=ffmpeg
        )
        if ogg is not None:
            written.append(ogg)
    if not written:
        return []
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
    p.add_argument(
        "--formats", default="sdf,png,csv", help="comma list, subset of sdf,png,csv,ogg"
    )
    p.add_argument(
        "--csv-seconds", type=float, default=30.0, help="leading window (s) for the VSA CSV"
    )
    p.add_argument(
        "--ogg-channels", type=int, default=1,
        help="discriminator-audio channels: 1=mono (default), 2=duplicate mono into L/R",
    )
    p.add_argument("--ffmpeg", default=None, help="path to ffmpeg when it is not on PATH")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    fmts = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    try:
        written = derive_views(
            args.input,
            center_hz=args.center_hz,
            sample_rate_hz=args.sample_rate,
            formats=fmts,
            csv_seconds=args.csv_seconds,
            ogg_channels=args.ogg_channels,
            ffmpeg=args.ffmpeg,
        )
    except Exception:
        log.exception("iq_views: failed to derive views from %s", args.input)
        return 1
    # RE-AUDIT (P2): a requested view that was NOT produced is a FAILURE, not a silent success. The
    # per-view helpers already return None / omit the path on failure or a legitimate skip (a
    # too-short capture has no waterfall; a failed ffmpeg has no OGG), so a requested format absent
    # from `written` is exactly that. Without this, main() exited 0 and the supervisor logged
    # "derived ogg" for an OGG that does not exist. Print the produced list so the caller logs what
    # ACTUALLY exists.
    produced = {p.suffix.lstrip(".").lower() for p in written}
    requested = {f for f in fmts if f in ("sdf", "csv", "png", "ogg")}
    print("produced=" + ",".join(sorted(produced)))  # noqa: T201 — machine-readable for the caller
    missing = sorted(requested - produced)
    if missing:
        log.error(
            "iq_views: requested view(s) %s were NOT produced from %s (produced: %s)",
            ",".join(missing), args.input, ",".join(sorted(produced)) or "none",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

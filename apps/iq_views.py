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
import sys
from pathlib import Path

import numpy as np
from _recorder import iq_to_sdf_bytes, write_vsa_csv, write_waterfall_png

log = logging.getLogger("iq_views")
_SDF_CHUNK = 1 << 20  # samples per chunk when transcoding cf32 → SDF (bounded memory)
_OGG_CHUNK = 1 << 20  # samples per chunk when deriving discriminator audio (bounded memory)
OGG_SAMPLE_RATE_HZ = 48_000  # SatNOGS discriminator-audio rate; audio_analyze.py expects it


def _unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


class _StreamingResampler:
    """Exact, stateful rational-rate resampler (``in_rate`` → ``out_rate``) for chunked streams.

    OPERATOR DECISION 2026-07-15: OGG encoding moved from the external ffmpeg binary to the
    pip-installable ``soundfile`` wheel (bundled libsndfile+Vorbis), so the resample ffmpeg
    used to do now happens here. Polyphase-equivalent: zero-stuff by L, FIR low-pass
    (windowed-sinc, gain L — the same design ``scipy.signal.resample_poly`` uses), take every
    M-th sample — with the FIR state (``lfilter`` ``zi``) and the decimation phase carried
    across chunks, so the output is bit-identical for ANY chunking of the same input.
    scipy is already a runtime dependency (apps/_stream.py uses the same primitives)."""

    MAX_UPSAMPLE = 16  # a larger L means an exotic capture rate; refuse honestly, never alias

    def __init__(self, in_rate: int, out_rate: int) -> None:
        from fractions import Fraction

        from scipy.signal import firwin

        if in_rate <= 0 or out_rate <= 0:
            msg = f"rates must be positive: {in_rate} -> {out_rate}"
            raise ValueError(msg)
        ratio = Fraction(out_rate, in_rate)
        self.up, self.down = ratio.numerator, ratio.denominator
        if self.up > self.MAX_UPSAMPLE:
            msg = (
                f"unsupported capture rate {in_rate} Hz for {out_rate} Hz audio "
                f"(upsample factor {self.up} > {self.MAX_UPSAMPLE})"
            )
            raise ValueError(msg)
        if self.up == 1 and self.down == 1:
            # Identity rate (a 48 kHz capture): pure passthrough — no filter, no delay.
            self._h = np.ones(1, dtype=np.float32)
            self._zi = np.zeros(0, dtype=np.float32)
            self._phase = 0
            return
        half = max(self.up, self.down)
        ntaps = 20 * half + 1  # resample_poly's default kaiser design (half_len = 10*half)
        self._h = (firwin(ntaps, 1.0 / half, window=("kaiser", 5.0)) * self.up).astype(np.float32)
        self._zi = np.zeros(len(self._h) - 1, dtype=np.float32)
        self._phase = 0  # upsampled-stream index (mod `down`) of the next chunk's first sample

    def process(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        if self.up == 1 and self.down == 1:
            return x
        up = np.zeros(x.size * self.up, dtype=np.float32)
        up[:: self.up] = x
        y, self._zi = lfilter(self._h, 1.0, up, zi=self._zi)
        first = (-self._phase) % self.down
        out = y[first :: self.down]
        self._phase = (self._phase + up.size) % self.down
        return np.asarray(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Drain the FIR group delay so the audio tail is not truncated at stream end."""
        if self.up == 1 and self.down == 1:
            return np.empty(0, dtype=np.float32)  # passthrough has no delay to drain
        tail_in = len(self._h) // (2 * self.up) + 1
        return self.process(np.zeros(tail_in, dtype=np.float32))


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
    """Whole-capture discriminator via the SAME chunked path the OGG encoder consumes.

    Exposed for tests: for ANY chunk size the result must equal the single-shot whole-array
    computation (phase continuity across boundaries is the correctness property)."""
    iq = np.asarray(iq, dtype=np.complex64)
    prev: np.complex64 | None = None
    parts: list[np.ndarray] = []
    for off in range(0, iq.shape[0], max(1, chunk_samples)):
        disc, prev = _discriminator_chunk(iq[off : off + chunk_samples], prev)
        parts.append(disc)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _write_audio_frames(sound_file: object, audio: np.ndarray, ogg_channels: int) -> None:
    """Write one resampled block: mono as-is, or the SAME mono signal explicitly duplicated
    into both channels (a deterministic mapping — stereo carries no extra information).
    Ringing from the resampler can overshoot ±1.0 slightly; clip so the encoder never wraps."""
    if audio.size == 0:
        return
    audio = np.clip(audio, -1.0, 1.0)
    if ogg_channels == 2:
        audio = np.repeat(audio[:, None], 2, axis=1)
    sound_file.write(audio)  # type: ignore[attr-defined]


def write_discriminator_ogg(
    cf32_path: Path,
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    ogg_channels: int = 1,
) -> Path | None:
    """Derive FM-discriminator audio from ``iq`` and encode it as a 48 kHz Vorbis ``<pass>.ogg``.

    OPERATOR DECISION 2026-07-15 (CA-INTEG-002): the encoder is the pip-installable
    ``soundfile`` wheel (bundled libsndfile + Vorbis) — no external ffmpeg binary, so the
    station's OGG dependency is provisioned by pip/wheelhouse like every other package.

    The mono float32 discriminator is derived in bounded chunks (phase continuity preserved
    across boundaries), resampled to 48 kHz by an exact stateful polyphase resampler, and
    written incrementally. The output goes to ``<pass>.ogg.tmp`` and is renamed atomically to
    ``<pass>.ogg`` ONLY after a clean, non-empty encode; the temp file is removed on
    cancellation or failure, so a partial file never wears the final name. Returns the final
    path, or ``None`` when nothing was written (soundfile missing, bad channel count,
    unsupported rate, too-short capture, or encode failure)."""
    if ogg_channels not in (1, 2):
        log.error("iq_views: ogg_channels must be 1 or 2, got %r — skipping OGG", ogg_channels)
        return None
    try:
        import soundfile as sf
    except ImportError:
        log.warning(
            "iq_views: python-soundfile is not installed — skipping OGG discriminator audio "
            "(pip install soundfile; its wheel bundles the Vorbis encoder)"
        )
        return None
    n = int(np.asarray(iq).shape[0])
    if n < 2:
        log.warning("iq_views: capture too short (%d samples) for discriminator audio", n)
        return None
    try:
        resampler = _StreamingResampler(int(round(sample_rate_hz)), OGG_SAMPLE_RATE_HZ)
    except ValueError as e:
        log.warning("iq_views: cannot derive OGG: %s", e)
        return None
    final = cf32_path.with_suffix(".ogg")
    tmp = cf32_path.with_suffix(".ogg.tmp")
    _unlink_quiet(tmp)  # a stale temp from a crashed run must never be renamed as the final file
    try:
        with sf.SoundFile(
            str(tmp), mode="w", samplerate=OGG_SAMPLE_RATE_HZ, channels=ogg_channels,
            format="OGG", subtype="VORBIS",
        ) as f:
            prev: np.complex64 | None = None
            for off in range(0, n, _OGG_CHUNK):
                block = np.ascontiguousarray(iq[off : off + _OGG_CHUNK], dtype=np.complex64)
                disc, prev = _discriminator_chunk(block, prev)
                _write_audio_frames(f, resampler.process(disc), ogg_channels)
            _write_audio_frames(f, resampler.flush(), ogg_channels)
    except (OSError, RuntimeError, ValueError) as e:
        _unlink_quiet(tmp)
        log.warning("iq_views: OGG encode failed: %s", e)
        return None
    except BaseException:
        # Cancellation (KeyboardInterrupt / SystemExit) mid-encode: drop the temp so a
        # partial file never wears the final name, then propagate.
        _unlink_quiet(tmp)
        raise
    # A truthful success requires a NON-EMPTY output: an encoder that produced nothing is a
    # failure, not a product (reporting it as produced is the false-success class).
    if not (tmp.exists() and tmp.stat().st_size > 0):
        _unlink_quiet(tmp)
        log.warning("iq_views: OGG encode produced no output — reporting failure")
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
            path, iq, sample_rate_hz=sample_rate_hz, ogg_channels=ogg_channels
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
        )
    except Exception:
        log.exception("iq_views: failed to derive views from %s", args.input)
        return 1
    # RE-AUDIT (P2): a requested view that was NOT produced is a FAILURE, not a silent success. The
    # per-view helpers already return None / omit the path on failure or a legitimate skip (a
    # too-short capture has no waterfall; a failed encode has no OGG), so a requested format absent
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

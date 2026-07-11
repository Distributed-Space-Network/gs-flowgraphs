#!/usr/bin/env python3
"""Decode SatNOGS discriminator-audio recordings.

This is deliberately separate from :mod:`iq_analyze`: an OGG produced by a
SatNOGS FSK flowgraph is demodulated mono audio, not RF IQ.  The tool recovers
hard symbols from that audio and passes them to the existing FCS-checked AX.25
and CRC-checked EnduroSat deframers.

Usage: python tools/audio_analyze.py capture.ogg [--baud 9600]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
import framings  # noqa: E402

from gfsk_ax25 import ax25  # noqa: E402
from gfsk_ax25.gfsk import _gardner  # noqa: E402

DEFAULT_BAUDS = (1200.0, 2400.0, 4800.0, 9600.0, 19200.0)


def _find_program(name: str, explicit: str | None) -> str:
    program = explicit or shutil.which(name)
    if not program:
        raise FileNotFoundError(
            f"{name} not found; install FFmpeg or pass --{name} /path/to/{name}"
        )
    return program


def probe_audio(path: str | Path, ffprobe: str | None = None) -> dict:
    """Return the first audio stream's basic metadata via ffprobe."""
    exe = _find_program("ffprobe", ffprobe)
    cmd = [
        exe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
        "-of", "json", os.fspath(Path(path)),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"{path}: no audio stream")
    stream = streams[0]
    return {
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": float(stream["sample_rate"]),
        "channels": int(stream.get("channels", 1)),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
    }


def load_audio(path: str | Path, ffmpeg: str | None = None, ffprobe: str | None = None):
    """Decode the first audio stream to mono float32 samples."""
    meta = probe_audio(path, ffprobe)
    exe = _find_program("ffmpeg", ffmpeg)
    cmd = [
        exe, "-v", "error", "-i", os.fspath(Path(path)), "-map", "0:a:0",
        "-ac", "1", "-f", "f32le", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype="<f4").copy(), meta


def _slice_soft(audio: np.ndarray, sample_rate: float, baud: float, phase: float = 0.0):
    """Integrate discriminator audio over each symbol and hard-slice it.

    Fractional samples/symbol are supported by linear interpolation. Centering
    between the two discriminator levels rejects residual carrier bias.
    """
    x = np.asarray(audio, dtype=np.float32)
    sps = float(sample_rate) / float(baud)
    if sps < 2.0:
        raise ValueError(f"sample rate {sample_rate:g} is too low for {baud:g} baud")
    sps_i = int(round(sps))
    if abs(sps - sps_i) < 1e-9 and abs(phase - round(phase)) < 1e-9:
        # SatNOGS FSK audio is normally 48 kHz and 9k6, exactly five
        # samples/symbol. This reshape path avoids constructing a large
        # interpolation grid for every pass window and timing phase.
        start = int(round(phase))
        nsym = (len(x) - start) // sps_i
        soft = x[start : start + nsym * sps_i].reshape(nsym, sps_i).mean(axis=1)
        if not len(soft):
            return np.empty(0, dtype=np.float32)
        low, high = np.percentile(soft, (10.0, 90.0))
        return (soft - (low + high) / 2.0).astype(np.float32)
    # Several samples across each symbol are averaged.  Avoid the edges where
    # Gaussian shaping carries most inter-symbol transition energy.
    count = max(3, int(np.ceil(sps)))
    offsets = np.linspace(0.2 * sps, 0.8 * sps, count)
    nsym = max(0, int(np.floor((len(x) - 1 - offsets[-1] - phase) / sps)) + 1)
    if not nsym:
        return np.empty(0, dtype=np.float32)
    centers = phase + np.arange(nsym) * sps
    positions = (centers[:, None] + offsets[None, :]).ravel()
    values = np.interp(positions, np.arange(len(x)), x)
    soft = values.reshape(nsym, count).mean(axis=1)
    # The midpoint of the two discriminator levels is stable even when a short
    # window contains more marks than spaces. A median would collapse onto the
    # majority level and turn every symbol into the same bit.
    low, high = np.percentile(soft, (10.0, 90.0))
    return (soft - (low + high) / 2.0).astype(np.float32)


def slice_symbols(audio: np.ndarray, sample_rate: float, baud: float, phase: float = 0.0):
    """Return hard discriminator symbols for one candidate clock phase."""
    return (_slice_soft(audio, sample_rate, baud, phase) > 0.0).astype(np.uint8)


def decode_audio(
    audio: np.ndarray,
    sample_rate: float,
    baud: float,
    framing_names: tuple[str, ...] = ("ax25", "endurosat"),
    *,
    window_s: float = 5.0,
) -> dict[str, list[tuple[float, bytes]]]:
    """Return unique CRC-valid frames grouped by framing name."""
    x = np.asarray(audio, dtype=np.float32)
    win = max(1, int(sample_rate * window_s))
    # One second of overlap is enough to keep normal AX.25 frames away from a
    # window boundary without decoding the whole recording twice.
    overlap = min(win // 2, int(sample_rate))
    hop = max(1, win - overlap)
    sps = sample_rate / baud
    sps_i = int(round(sps))
    phases = (
        np.arange(sps_i, dtype=float)
        if abs(sps - sps_i) < 1e-9
        else np.linspace(0.0, sps, max(5, int(np.ceil(sps)) * 2), endpoint=False)
    )
    found: dict[str, list[tuple[float, bytes]]] = {name: [] for name in framing_names}
    seen: dict[str, set[bytes]] = {name: set() for name in framing_names}
    for start in range(0, len(x), hop):
        segment = x[start : start + win]
        if len(segment) < sample_rate * 0.05:
            continue
        # Rank fixed phases by eye opening before the comparatively expensive
        # Python G3RUH/HDLC pass. Also add adaptive timing recovery: this is the
        # important path when the spacecraft symbol clock is not exactly the
        # nominal baud rate, as in SatNOGS's live M&M clock-recovery branch.
        candidates = []
        for phase in phases:
            soft = _slice_soft(segment, sample_rate, baud, float(phase))
            score = float(np.percentile(soft, 90) - np.percentile(soft, 10)) if len(soft) else 0.0
            candidates.append((score, soft))
        # Gardner recovery needs useful samples on both sides of a transition.
        # At 48 kHz / 19k2 there are only 2.5 samples/symbol; fixed best-eye
        # slicing remains meaningful, while this Python recovery is both weak
        # and disproportionately expensive. Higher-oversampling rates use it.
        adaptive = _gardner(segment, sps) if sps >= 4.0 else np.empty(0, dtype=np.float32)
        if len(adaptive):
            low, high = np.percentile(adaptive, (10.0, 90.0))
            adaptive = adaptive - (low + high) / 2.0
        selected = [max(candidates, key=lambda item: item[0])[1]]
        if len(adaptive):
            selected.append(adaptive)
        for soft in selected:
            bits = (soft > 0.0).astype(np.uint8)
            for framing_name in framing_names:
                frames, _ = framings.deframe(bits, framing_name)
                for frame in frames:
                    if frame not in seen[framing_name]:
                        seen[framing_name].add(frame)
                        found[framing_name].append((start / sample_rate, frame))
    return found


def decode_ax25_audio(
    audio: np.ndarray, sample_rate: float, baud: float, *, window_s: float = 5.0
) -> list[tuple[float, bytes]]:
    """Compatibility helper returning only CRC-valid AX.25 frames."""
    return decode_audio(audio, sample_rate, baud, ("ax25",), window_s=window_s)["ax25"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SatNOGS discriminator-audio analysis")
    parser.add_argument("capture", help="SatNOGS audio file (normally .ogg)")
    parser.add_argument(
        "--baud", type=float, action="append",
        help="baud rate to try (repeatable); default: 1200,2400,4800,9600,19200",
    )
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument(
        "--ax25", action="store_true",
        help="decode only AX.25; no framing flag decodes AX.25 and EnduroSat",
    )
    parser.add_argument(
        "--endurosat", action="store_true",
        help="decode only EnduroSat chip-packet framing",
    )
    parser.add_argument("--ffmpeg", help="path to ffmpeg when it is not on PATH")
    parser.add_argument("--ffprobe", help="path to ffprobe when it is not on PATH")
    args = parser.parse_args(argv)

    audio, meta = load_audio(args.capture, args.ffmpeg, args.ffprobe)
    duration = len(audio) / meta["sample_rate"]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    print(
        f"{Path(args.capture).name}: {meta['codec']} mono | fs={meta['sample_rate']:.0f} Hz | "
        f"dur={duration:.3f} s | peak={peak:.3f} rms={rms:.3f}",
        flush=True,
    )
    bauds = tuple(args.baud) if args.baud else DEFAULT_BAUDS
    framing_names = tuple(
        name for name, enabled in (("ax25", args.ax25), ("endurosat", args.endurosat)) if enabled
    ) or ("ax25", "endurosat")
    for baud in bauds:
        results = decode_audio(
            audio, meta["sample_rate"], baud, framing_names, window_s=args.window_s
        )
        for framing_name in framing_names:
            frames = results[framing_name]
            print(f"{framing_name}: {len(frames)} unique CRC-valid frame(s) at {baud:g} baud")
            for index, (at, frame) in enumerate(frames):
                detail = ""
                if framing_name == "ax25":
                    ui = ax25.decode_ui(frame)
                    if ui is not None:
                        detail = (
                            f" {ui.src}-{ui.src_ssid} -> {ui.dest}-{ui.dest_ssid}"
                            f" info_len={len(ui.info)}"
                        )
                print(
                    f"  frame {index}: t~{at:.3f}s len={len(frame)}{detail} hex={frame.hex()}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

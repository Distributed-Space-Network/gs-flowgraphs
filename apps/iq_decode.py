#!/usr/bin/env python3
"""Post-pass decode of a recorded ``.cf32`` capture with the NON-live (heavier) framings.

Runs AFTER the pass, decoupled from the flowgraph — free CPU, no 30 s stop budget (the same model
as ``iq_views``). The live RX engines already decode the LIGHT framings (ax25 + endurosat) in real
time; this sweeps the recorded IQ for the framings they do NOT run live
(``framings.POST_PASS_FRAMINGS`` — the other CRC-gated local link layers, currently ``ccsds_tm``),
so a pass that carried one of those still yields frames. ``kiss`` is NOT swept by default (it has
no integrity check, so a blind whole-pass sweep would emit noise "frames") — request it explicitly
if a pass is known KISS. gr-satellites-only framings (USP, AX100, …) are not handled here at all;
they need the GNU Radio engine.

**Doppler.** The recorded ``.cf32`` is RAW — captured BEFORE the live Doppler NCO — so it carries
the full pass Doppler swing (±~9 kHz at 400 MHz LEO), which is larger than a burst CFO estimate can
pull back. We do NOT try to. Doppler is DETERMINISTIC: gs-orbitd re-propagates the pass's TLE over
its time window to the exact same track it drove live. So gs-client hands us that track
(``doppler_track`` — ``[(t_seconds_from_capture_start, offset_hz), …]``, sampled from gs-orbitd),
and we de-rotate the raw IQ with it — reproducing precisely what the live decoder saw — before
demod. Nothing about the track is persisted anywhere; it is regenerated on demand. With NO track
(a lab/file capture with negligible Doppler) we fall back to per-window CFO, which is best-effort.
The track assumes a RAW capture — as the dsp / bidir RX engines record it (pre-NCO). The gnuradio
engine retunes the SDR source in HARDWARE, so its ``.cf32`` is already Doppler-corrected; feeding a
track there would double-correct, so post-pass decode targets the raw-recording engines (gs-client
only pairs the track with those). The de-rotation is per-window, so memory stays bounded to one
window (a whole-capture de-rotation would materialise multi-GB temporaries on a long pass).

Pure numpy — no GNU Radio, no gs-orbitd dependency (gs-client owns the gs-orbitd query and passes
the sampled track) — so it runs anywhere and is unit-testable on the dev box.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import framings
import numpy as np

from gfsk_ax25 import gfsk

log = logging.getLogger("iq_decode")
_DEFAULT_SYMBOL_RATE_HZ = 9600.0
_DEFAULT_WINDOW_S = 1.0  # short enough that any residual offset is ~constant across the window


def _derotate_doppler(
    iq: np.ndarray, sample_rate_hz: float, track: list[tuple[float, float]]
) -> np.ndarray:
    """De-rotate ``iq`` by a Doppler ``track`` (``[(t_s, offset_hz), …]`` from gs-orbitd),
    reproducing the live NCO. The offset at each sample is linearly interpolated over the track and
    the phase accumulated continuously, so the whole pass is brought near DC exactly as it was live
    (the live NCO applies ``exp(-j·2π·offset·n/fs)`` per chunk; this is its continuous form)."""
    if not track or len(iq) == 0:
        return np.asarray(iq, dtype=np.complex64)
    ts = np.asarray([float(t) for t, _ in track], dtype=np.float64)
    offs = np.asarray([float(o) for _, o in track], dtype=np.float64)
    t = np.arange(len(iq), dtype=np.float64) / float(sample_rate_hz)
    off_per_sample = np.interp(t, ts, offs)  # Hz at each sample (flat outside the track ends)
    phase = -2.0 * np.pi * np.cumsum(off_per_sample) / float(sample_rate_hz)
    return (np.asarray(iq, dtype=np.complex64) * np.exp(1j * phase)).astype(np.complex64)


def decode_capture(
    cf32: str | Path,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float = _DEFAULT_SYMBOL_RATE_HZ,
    framings_to_try: tuple[str, ...] = framings.POST_PASS_FRAMINGS,
    doppler_track: list[tuple[float, float]] | None = None,
    window_s: float = _DEFAULT_WINDOW_S,
    mod_index: float = 0.5,
    bt: float = 0.5,
) -> list[dict]:
    """Doppler-de-rotate (from ``doppler_track``) + windowed demod + deframe of ``cf32`` with each
    framing in ``framings_to_try``. Returns the decoded frame records (also appended to
    ``<pass>/frames.jsonl`` when any are found)."""
    path = Path(cf32)
    if not framings_to_try:
        return []
    if not path.exists():
        log.warning("iq_decode: %s not found", path)
        return []
    # Prefer the recording's own sidecar (the TRUE rate the engine used — it may have widened the
    # channel for a high-baud bird) over the passed-in rate. Mirrors iq_views.
    meta = path.with_name(path.name + ".json")
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            sample_rate_hz = float(d.get("sample_rate_hz", sample_rate_hz))
        except (OSError, ValueError, TypeError):
            log.warning("iq_decode: ignoring unreadable sidecar %s", meta.name)
    n_samp = path.stat().st_size // 8  # 8 B/complex64; floor a torn write to whole samples
    if n_samp < 1:
        log.warning("iq_decode: %s has no samples", path)
        return []
    iq = np.memmap(path, dtype=np.complex64, mode="r", shape=(n_samp,))
    # We do NOT de-rotate the whole capture up front: that materialises a complex128 phase array +
    # exp over the ENTIRE pass (multi-GB on a long capture → OOM on a constrained station). Instead
    # each window is sliced from the memmap and de-rotated locally, bounding memory to one window.
    have_track = bool(doppler_track)
    if not have_track:
        log.warning("iq_decode: no Doppler track — falling back to per-window CFO (best-effort)")
    win = max(1, int(sample_rate_hz * window_s))
    # Overlap windows by ~half so a frame straddling a window boundary is fully contained in the
    # next window (a non-overlapping sweep silently drops boundary frames). A seen-set dedups the
    # re-decoded overlap frames by (framing, payload) — a genuine exact-duplicate payload is also
    # collapsed, acceptable for a post-pass completeness sweep (CCSDS frame counters differ anyway).
    step = max(1, win - win // 2)
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for off in range(0, n_samp, step):
        seg = np.asarray(iq[off : off + win])  # bounded window materialised from the memmap here
        if have_track:
            # Shift the track to window-local time (_derotate_doppler times from 0), so only THIS
            # window's samples are rotated — no whole-capture temporaries.
            t_off = off / sample_rate_hz
            wtrack = [(t - t_off, o) for t, o in doppler_track or []]
            seg = _derotate_doppler(seg, sample_rate_hz, wtrack)
        try:
            bits = gfsk.demodulate_capture(
                seg,
                sample_rate_hz,
                symbol_rate_hz=symbol_rate_hz,
                mod_index=mod_index,
                bt=bt,
                # With the track applied the window is already near DC → correct only the small
                # residual. Without a track, this per-window CFO is the (weaker) sole correction.
                correct_cfo=True,
                # Max-eye sampling (NOT Gardner): demodulate_capture is tuned for it — Gardner's
                # timing recovery diverges on a capture, so recover_timing=True yields no frames.
                recover_timing=False,
            )
        except Exception:  # noqa: BLE001 — one bad window must not abort the whole sweep
            log.exception("iq_decode: demod failed on window @%d", off)
            continue
        if not len(bits):
            continue
        for name in framings_to_try:
            try:
                frames, matched = framings.deframe(bits, name)
            except Exception:  # noqa: BLE001 — a deframer bug must not abort the sweep
                log.exception("iq_decode: %s deframe failed @%d", name, off)
                continue
            for body in frames:
                key = (matched or name, body.hex())
                if key in seen:  # a boundary frame re-decoded in the overlapping next window
                    continue
                seen.add(key)
                records.append(
                    {
                        "ts": round(time.time(), 3),
                        "framing": matched or name,
                        "len": len(body),
                        "crc_ok": True,
                        "payload_hex": body.hex(),
                        "post_pass": True,  # distinguishes these from the live-decoded frames
                    }
                )
    if records:
        _append_frames(path, records)
    log.info(
        "iq_decode: %d post-pass frame(s) from %s (framings=%s, doppler=%s)",
        len(records),
        path.name,
        ",".join(framings_to_try),
        "track" if have_track else "cfo-only",
    )
    return records


def _append_frames(cf32: Path, records: list[dict]) -> None:
    """Append post-pass frames to the pass's ``frames.jsonl`` (the decoded-frames product), each
    tagged ``post_pass=True``. Runs after the flowgraph exited, so there is no race with the live
    writer. Never raises."""
    out = cf32.parent / "frames.jsonl"
    try:
        with out.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    except OSError as e:
        log.warning("iq_decode: frames.jsonl append failed: %s", e)


def _load_track(path_str: str) -> list[tuple[float, float]]:
    """Load a Doppler track JSON (``[[t_s, offset_hz], …]`` from gs-client / gs-orbitd)."""
    if not path_str:
        return []
    try:
        raw = json.loads(Path(path_str).read_text())
        return [(float(t), float(o)) for t, o in raw]
    except (OSError, ValueError, TypeError):
        log.warning("iq_decode: unreadable doppler track %s; ignoring", path_str)
        return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="iq_decode",
        description="Post-pass decode of a recorded .cf32 with the non-live (CRC-gated) framings.",
    )
    p.add_argument("--input", required=True, help="path to the .cf32 capture")
    p.add_argument("--sample-rate", type=float, required=True, help="capture sample rate, Hz")
    p.add_argument(
        "--symbol-rate", type=float, default=_DEFAULT_SYMBOL_RATE_HZ, help="link symbol rate, Hz"
    )
    p.add_argument(
        "--framings",
        default=",".join(framings.POST_PASS_FRAMINGS),
        help="comma list; default = the non-live CRC-gated local framings",
    )
    p.add_argument(
        "--doppler-track",
        default="",
        help="JSON [[t_s, offset_hz], …] Doppler track from gs-orbitd (else per-window CFO)",
    )
    p.add_argument(
        "--window-s", type=float, default=_DEFAULT_WINDOW_S, help="demod window (s); keep short"
    )
    p.add_argument("--mod-index", type=float, default=0.5)
    p.add_argument("--bt", type=float, default=0.5)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    fmts = tuple(f.strip().lower() for f in args.framings.split(",") if f.strip())
    try:
        decode_capture(
            args.input,
            sample_rate_hz=args.sample_rate,
            symbol_rate_hz=args.symbol_rate,
            framings_to_try=fmts,
            doppler_track=_load_track(args.doppler_track) or None,
            window_s=args.window_s,
            mod_index=args.mod_index,
            bt=args.bt,
        )
    except Exception:
        log.exception("iq_decode: failed on %s", args.input)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

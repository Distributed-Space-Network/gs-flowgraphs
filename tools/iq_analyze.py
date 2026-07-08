#!/usr/bin/env python3
"""IQ capture analysis for EnduroSat UHF reverse-engineering.

Loads a capture, detects bursts, demodulates 2-GFSK via the tested ``gfsk_ax25``
library, and locates the chip-packet sync (0xAA preamble + 0x7E sync word). Used to
reverse the AirMAC framing from lab captures (see memory: endurosat-airmac-protocol).

Two input formats, by extension:
  * ``.cf32`` — raw complex64 the GR engines record (the WHOLE pass), with rate/centre
    read from the ``<file>.cf32.json`` sidecar (or ``--sample-rate``). Preferred: it's
    the full capture, not a window.
  * ``.csv``  — Keysight 89600 VSA "Main Time" export: key,value header, a ``Y`` line,
    then ``I,Q`` float rows. Sample rate from ``XDelta``. Note the VSA CSV is typically
    only an analysis WINDOW (e.g. 10 s) of a longer recording — use the cf32 for the
    full pass.

Usage:  python tools/iq_analyze.py <capture.cf32|.csv> [--symbol-rate 9600] [--sample-rate HZ]

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
import framings  # noqa: E402 — numpy deframe registry (ax25 FCS-checked, etc.)

from gfsk_ax25 import gfsk  # noqa: E402

DEFAULT_SYMBOL_RATE = 9600.0
DEFAULT_MOD_INDEX = 0.5  # dev +-2400 Hz at 9600 sym/s (per the lab capture)
DEFAULT_BT = 0.5
PREAMBLE_BITS = (1, 0)  # 0xAA = 1010..., MSB-first
SYNC_FLAG = (0, 1, 1, 1, 1, 1, 1, 0)  # 0x7E
# Standard amateur symbol rates the AX.25 sweep tries when the pass didn't decode at the labelled
# rate (baud == symbol rate; a wrong-rate demod yields no FCS-valid frame, so the sweep is safe).
DEFAULT_SWEEP_BAUDS = (1200.0, 2400.0, 4800.0, 9600.0)
# A spectral line this many dB over the noise floor counts as a real carrier (vs a spur/noise bin).
CARRIER_SNR_DB = 6.0


@dataclass
class Capture:
    iq: np.ndarray
    fs: float
    center_hz: float
    meta: dict[str, str]


def load_vsa_csv(path: str | Path) -> Capture:
    """Load a VSA Main-Time CSV export into a :class:`Capture`."""
    path = Path(path)
    meta: dict[str, str] = {}
    y_line = None
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s == "Y":
                y_line = i + 1
                break
            if "," in s and not s[0].isdigit() and s[0] != "-":
                key, _, val = s.partition(",")
                meta[key] = val
            if i > 256:  # header is short; guard against scanning a huge file
                break
    if y_line is None:
        msg = f"no 'Y' data marker found in {path}"
        raise ValueError(msg)
    iq = _read_iq_rows(path, y_line)
    xdelta = float(meta.get("XDelta", "0") or 0.0)
    fs = 1.0 / xdelta if xdelta else 0.0
    center = float(meta.get("InputCenter", "0") or 0.0)
    return Capture(iq=iq, fs=fs, center_hz=center, meta=meta)


def load_cf32(path: str | Path, sample_rate_hz: float = 0.0) -> Capture:
    """Load a raw complex64 (cf32) capture — the WHOLE pass the GR engines record. Rate
    and centre come from the ``<file>.cf32.json`` sidecar (PassRecorder writes it), or
    from ``sample_rate_hz`` when there's no sidecar."""
    path = Path(path).expanduser()
    if not path.is_file():
        # Clear error instead of a cryptic numpy one — shows the RESOLVED absolute path + cwd, so a
        # relative-vs-absolute path mix-up (e.g. the leading '/' lost) is obvious immediately.
        msg = f"capture not found: {path} (resolved {path.resolve()}; cwd {Path.cwd()})"
        raise FileNotFoundError(msg)
    iq = np.fromfile(str(path), dtype=np.complex64)  # str(): some numpy builds mishandle Path here
    fs, center, meta = sample_rate_hz, 0.0, {}
    sidecar = path.with_name(path.name + ".json")
    if sidecar.exists():
        with contextlib.suppress(OSError, ValueError, TypeError):
            d = json.loads(sidecar.read_text())
            fs = float(d.get("sample_rate_hz", fs))
            center = float(d.get("center_hz", center))
            meta = {k: str(v) for k, v in d.items()}
    return Capture(iq=iq, fs=fs, center_hz=center, meta=meta)


def load_capture(path: str | Path, sample_rate_hz: float = 0.0) -> Capture:
    """Load a ``.cf32`` (raw, whole-pass) or ``.csv`` (VSA window) capture by extension."""
    p = Path(path)
    cap = load_cf32(p, sample_rate_hz) if p.suffix.lower() == ".cf32" else load_vsa_csv(p)
    if not cap.fs:
        msg = f"{p.name}: unknown sample rate (no cf32 sidecar / XDelta) — pass --sample-rate"
        raise ValueError(msg)
    return cap


def _read_iq_rows(path: Path, skiprows: int) -> np.ndarray:
    try:
        import pandas as pd  # fast path

        df = pd.read_csv(
            path, skiprows=skiprows, header=None, usecols=[0, 1], names=["i", "q"], dtype=np.float64
        )
        return (df["i"].to_numpy() + 1j * df["q"].to_numpy()).astype(np.complex64)
    except ImportError:
        data = np.loadtxt(path, delimiter=",", skiprows=skiprows, usecols=(0, 1))
        return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)


def find_bursts(
    iq: np.ndarray, fs: float, *, min_ms: float = 2.0, threshold_mult: float = 4.0
) -> list[tuple[int, int]]:
    """Return (start, end) sample indices of on-air bursts via a magnitude gate."""
    mag = np.abs(iq)
    thr = max(np.median(mag) * threshold_mult, mag.max() * 0.08)
    on = (mag > thr).astype(np.int8)
    d = np.diff(on, prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    min_samp = fs * min_ms / 1000.0
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False) if (e - s) > min_samp]


def gfsk_params(fs: float, symbol_rate: float = DEFAULT_SYMBOL_RATE) -> gfsk.GfskParams:
    return gfsk.GfskParams(
        sample_rate_hz=fs, symbol_rate_hz=symbol_rate, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT
    )


def spectrum_summary(iq: np.ndarray, fs: float, *, nfft: int = 8192) -> dict[str, float] | None:
    """Averaged periodogram of the WHOLE capture → the strongest spectral line and how far it
    stands above the noise floor (median). Unlike :func:`find_bursts` (a magnitude gate tuned for
    strong EnduroSat bursts), this also catches a WEAK CONTINUOUS carrier. The recording is
    post-Doppler-rotator, so a real downlink sits near 0 Hz; a peak far off DC is a spur/RFI, and a
    flat spectrum (SNR below :data:`CARRIER_SNR_DB`) is a dead capture — nothing to decode."""
    n = int(iq.size)
    if n < 16:
        return None
    nfft = min(int(nfft), 1 << int(np.log2(n)))
    win = np.hanning(nfft)
    nseg = n // nfft
    acc = np.zeros(nfft, dtype=np.float64)
    for i in range(nseg):
        seg = iq[i * nfft : (i + 1) * nfft] * win
        acc += np.abs(np.fft.fft(seg)) ** 2
    psd_db = 10.0 * np.log10(np.fft.fftshift(acc / max(nseg, 1)) + 1e-30)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    floor = float(np.median(psd_db))
    pk = int(np.argmax(psd_db))
    return {"peak_hz": float(freqs[pk]), "snr_db": float(psd_db[pk] - floor), "nseg": float(nseg)}


def framing_sweep(
    iq: np.ndarray, fs: float, framing: str = "ax25", bauds=DEFAULT_SWEEP_BAUDS,
    *, carriers=None, target_sps: int = 8,
) -> list[tuple[float, float, int, list[bytes]]]:
    """Run OUR real deframer (``framings.deframe`` for the given ``framing``: ax25 = G3RUH+plain,
    NRZI, FCS + callsign-validated; endurosat = chip-packet, CRC-16 gated, both polarities) on the
    capture at each candidate baud, with COARSE CARRIER RECOVERY. Doppler compensation leaves the
    bird's fixed oscillator offset (tens of kHz), which parks the carrier outside the narrow demod
    filter; we de-rotate a candidate carrier to DC first. TWO-STAGE (cost): try DC + the spectral
    peak first, and only fall back to a coarse ±21 kHz / 3 kHz GRID when neither decodes — so an
    on-freq or peak-locked bird is a couple of demods, and the expensive grid runs only when needed.
    The checksum gate means a wrong (carrier, baud) yields nothing. ``target_sps`` keeps the
    polyphase resample light (16 upsamples 9600→153.6 k = a huge per-demod convolve on an embedded
    CPU). Returns ``(baud, carrier_hz, n_frames, frames)`` — the winning carrier per baud."""
    if carriers is not None:
        stages: list[list[float]] = [[float(c) for c in carriers]]
    else:
        sp = spectrum_summary(iq, fs)
        peak = round(sp["peak_hz"]) if sp else 0.0
        grid = sorted({float(h) for h in range(-21000, 21001, 3000)} - {0.0, float(peak)})
        stages = [[0.0, float(peak)], grid]  # cheap {DC, peak} first; coarse grid only if needed
    out: list[tuple[float, float, int, list[bytes]]] = []
    for baud in bauds:
        best: tuple[float, float, int, list[bytes]] = (float(baud), 0.0, 0, [])
        for stage in stages:
            for carrier in stage:
                # demodulate_capture: derotates the carrier + residual CFO and polyphase-resamples
                # to an integer target_sps before slicing (raw demodulate's timing recovery DIVERGES
                # at the channel's native sps, e.g. sps=40 for 1200 Bd over 48 kHz).
                bits = gfsk.demodulate_capture(
                    iq, fs, symbol_rate_hz=baud, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
                    carrier_hz=carrier, target_sps=target_sps)
                frames, _ = framings.deframe(bits, framing)
                if len(frames) > best[2]:
                    best = (float(baud), float(carrier), len(frames), frames)
            if best[2] > 0:
                break  # decoded without needing the coarse grid
        out.append(best)
    return out


def ax25_sweep(
    iq: np.ndarray, fs: float, bauds=DEFAULT_SWEEP_BAUDS, *, carriers=None, target_sps: int = 8
) -> list[tuple[float, float, int, list[bytes]]]:
    """AX.25 carrier/baud sweep — thin wrapper over :func:`framing_sweep` (kept for tests)."""
    return framing_sweep(iq, fs, "ax25", bauds, carriers=carriers, target_sps=target_sps)


def demodulate_burst(
    iq: np.ndarray, fs: float, *, symbol_rate: float = DEFAULT_SYMBOL_RATE, carrier_hz: float = 0.0
) -> np.ndarray:
    """Demodulate one burst (caller passes a slice incl. guard) to hard bits, DE-ROTATING
    ``carrier_hz`` to DC first. Doppler + the bird's fixed oscillator offset park the carrier
    outside the narrow demod filter, so a raw demod there gives NO-SYNC — the caller estimates the
    offset (from the spectrum, or --carrier-hz) and passes it here; ``correct_cfo`` then cleans the
    residual. Via ``demodulate_capture`` this also polyphase-resamples, so a non-integer
    samples/symbol rate (e.g. 500 kHz / 9600) still demods — handy for the sample-rate sweeps."""
    return gfsk.demodulate_capture(
        iq, fs, symbol_rate_hz=symbol_rate, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
        carrier_hz=carrier_hz, correct_cfo=True, recover_timing=False,
    )


def find_sync(bits: np.ndarray, *, min_preamble: int = 2) -> int | None:
    """Index of the first payload bit after a 0xAA-preamble + 0x7E sync; None if
    no sync. Bit-pattern search, so it is byte-packing agnostic."""
    s = "".join(map(str, np.asarray(bits).tolist()))
    pat = f"(?:10101010){{{min_preamble},}}01111110"
    m = re.search(pat, s) or re.search("01111110", s)
    return m.end() if m else None


def frame_bytes(bits: np.ndarray, *, order: str = "big") -> bytes:
    b = np.asarray(bits, dtype=np.uint8)
    n = (len(b) // 8) * 8
    return np.packbits(b[:n], bitorder=order).tobytes() if n else b""


def _center_window(iq: np.ndarray, fs: float, window_s: float) -> tuple[np.ndarray, float]:
    """A ``window_s`` slice centered on the MIDDLE of the capture (~TCA — strongest signal, Doppler
    ≈ 0, where the packets are), or the whole capture when ``window_s`` <= 0. The old first-N-sec
    window sat at AOS (low/weak) and missed the TCA bursts. Returns (window, start_time_s)."""
    if window_s <= 0 or len(iq) <= int(fs * window_s):
        return iq, 0.0
    half = int(fs * window_s / 2)
    mid = len(iq) // 2
    start = max(0, mid - half)
    return iq[start : start + 2 * half], start / fs


def analyze_file(
    path: str | Path, symbol_rate: float = DEFAULT_SYMBOL_RATE, sample_rate_hz: float = 0.0,
    *, run_ax25: bool = False, run_endurosat: bool = False, sweep_window_s: float = 40.0,
    carrier_hz: float | None = None, want_waterfall: bool = False,
) -> None:
    cap = load_capture(path, sample_rate_hz)
    dur = len(cap.iq) / cap.fs if cap.fs else 0.0
    print(
        f"{Path(path).name}: {len(cap.iq):,} samples | fs={cap.fs:.0f} Hz | "
        f"center={cap.center_hz/1e6:.4f} MHz | dur={dur:.3f} s"
    )
    if want_waterfall:
        from _recorder import (
            write_waterfall_png,  # noqa: PLC0415 — matplotlib color / gray fallback
        )

        wf = Path(path).with_suffix(".analyze.png")
        write_waterfall_png(
            wf, cap.iq, sample_rate_hz=cap.fs, center_hz=cap.center_hz, title=Path(path).stem)
        print(f"waterfall: wrote {wf.name}")
    # Carrier check FIRST: is there ANY signal (weak/continuous too, not just bursts)? A NO CARRIER
    # verdict makes the demod/framing/baud debate moot — the capture is empty (freq/antenna/off).
    sp = spectrum_summary(cap.iq, cap.fs)
    if sp is not None:
        verdict = (f"CARRIER at {sp['peak_hz']:+.0f} Hz from DC"
                   if sp["snr_db"] >= CARRIER_SNR_DB else "NO CARRIER — flat noise (dead capture)")
        print(f"spectrum: strongest line {sp['peak_hz']:+.0f} Hz, {sp['snr_db']:.1f} dB over floor "
              f"→ {verdict}")
    if run_ax25:
        win, t0 = _center_window(cap.iq, cap.fs, sweep_window_s)
        span = len(win) / cap.fs if cap.fs else 0.0
        carriers = None if carrier_hz is None else [float(carrier_hz)]
        # Sweep ONLY the labelled --symbol-rate (default 9600): the baud is known from the pass
        # params, and a full 4-baud x carrier-grid sweep is 4x the (heavy) demods for nothing.
        bauds = (float(symbol_rate),) if symbol_rate else DEFAULT_SWEEP_BAUDS
        print(f"ax25 sweep ({int(symbol_rate)} Bd, G3RUH+plain, FCS-checked, carrier-recovered; "
              f"{span:.0f}s @ t={t0:.0f}s{' [whole pass]' if sweep_window_s <= 0 else ''}):")
        for baud, carrier, nframes, frames in ax25_sweep(win, cap.fs, bauds, carriers=carriers):
            head = frames[0][:16].hex() if frames else "-"
            print(f"  {int(baud):5d} Bd @ carrier {carrier:+.0f} Hz: {nframes} FCS frame(s)  "
                  f"first={head}")
    if run_endurosat:
        # Same carrier-recovering sweep as --ax25, but the EnduroSat chip-packet deframer (CRC-16
        # gated, both bit polarities). This is the DEFRAMED, checksum-validated EnduroSat extraction
        # — vs the raw byte dump in the burst listing below.
        win, t0 = _center_window(cap.iq, cap.fs, sweep_window_s)
        span = len(win) / cap.fs if cap.fs else 0.0
        carriers = None if carrier_hz is None else [float(carrier_hz)]
        bauds = (float(symbol_rate),) if symbol_rate else DEFAULT_SWEEP_BAUDS
        print(f"endurosat sweep ({int(symbol_rate)} Bd, chip-packet CRC-16, carrier-recovered; "
              f"{span:.0f}s @ t={t0:.0f}s{' [whole pass]' if sweep_window_s <= 0 else ''}):")
        for baud, carrier, nframes, frames in framing_sweep(
            win, cap.fs, "endurosat", bauds, carriers=carriers
        ):
            head = frames[0][:16].hex() if frames else "-"
            print(f"  {int(baud):5d} Bd @ carrier {carrier:+.0f} Hz: {nframes} CRC frame(s)  "
                  f"first={head}")
    # Carrier for the burst demod: --carrier-hz if given, else the capture's dominant spectral line.
    # This is what makes the EnduroSat framing readable off a raw .cf32: the bird's fixed oscillator
    # offset + Doppler park the signal off the narrow demod filter, so de-rotating it to DC turns
    # NO-SYNC into a decode. 0.0 falls back to correct_cfo alone (fine for a near-DC capture).
    burst_carrier = (
        float(carrier_hz)
        if carrier_hz is not None
        else (round(sp["peak_hz"]) if sp and sp["snr_db"] >= CARRIER_SNR_DB else 0.0)
    )
    bursts = find_bursts(cap.iq, cap.fs)
    guard = int(cap.fs * 0.003)
    print(f"{len(bursts)} bursts (demod carrier {burst_carrier:+.0f} Hz):")
    for k, (s, e) in enumerate(bursts):
        seg = cap.iq[max(0, s - guard) : e + guard]
        bits = demodulate_burst(seg, cap.fs, symbol_rate=symbol_rate, carrier_hz=burst_carrier)
        idx = find_sync(bits)
        fb = frame_bytes(bits[idx:]) if idx is not None else b""
        synced = f"sync@bit{idx}" if idx is not None else "NO-SYNC"
        # FULL on-wire frame bytes after the 0xAA/0x7E sync (len + payload + CRC) — the EnduroSat
        # framing to inspect/extract; bounded per burst. '-' when the burst didn't sync.
        print(
            f"  burst {k}: t={s/cap.fs:7.3f}s dur={(e-s)/cap.fs*1000:6.1f}ms "
            f"{synced:>10}  frame={fb.hex() if fb else '-'}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="iq_analyze", description="EnduroSat UHF IQ analysis")
    p.add_argument("capture", help="capture file: .cf32 (raw, whole pass) or VSA .csv")
    p.add_argument("--symbol-rate", type=float, default=DEFAULT_SYMBOL_RATE)
    p.add_argument(
        "--sample-rate", type=float, default=0.0, help="cf32 sample rate if no sidecar (Hz)"
    )
    p.add_argument("--ax25", action="store_true",
                   help="run OUR AX.25 deframer (FCS-checked) sweeping standard bauds + carriers")
    p.add_argument("--endurosat", action="store_true",
                   help="run OUR EnduroSat chip-packet deframer (CRC-16) with the carrier sweep")
    p.add_argument("--sweep-window-s", type=float, default=40.0,
                   help="seconds around TCA (mid-pass) to run the --ax25 sweep on (0 = whole pass)")
    p.add_argument("--carrier-hz", type=float, default=None,
                   help="de-rotate this exact carrier offset (Hz) instead of the auto grid")
    p.add_argument("--waterfall", action="store_true",
                   help="write a colored spectrogram <capture>.analyze.png (needs matplotlib)")
    args = p.parse_args(argv)
    analyze_file(args.capture, symbol_rate=args.symbol_rate, sample_rate_hz=args.sample_rate,
                 run_ax25=args.ax25, run_endurosat=args.endurosat,
                 sweep_window_s=args.sweep_window_s,
                 carrier_hz=args.carrier_hz, want_waterfall=args.waterfall)
    return 0


if __name__ == "__main__":
    sys.exit(main())

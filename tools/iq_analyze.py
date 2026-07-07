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
    path = Path(path)
    iq = np.fromfile(path, dtype=np.complex64)
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


def ax25_sweep(
    iq: np.ndarray, fs: float, bauds=DEFAULT_SWEEP_BAUDS, *, carriers=None
) -> list[tuple[float, float, int, list[bytes]]]:
    """Run OUR real AX.25 deframer (``framings.deframe`` — G3RUH + plain, NRZI, FCS-checked) on the
    capture at each candidate baud, with COARSE CARRIER RECOVERY. Doppler compensation leaves the
    bird's fixed oscillator offset (tens of kHz), which parks the carrier outside the narrow demod
    filter; we de-rotate a candidate carrier to DC first. Candidates default to DC (on-freq bird)
    AND the spectral peak (offset bird); FCS-gating means a wrong (carrier, baud) yields nothing.
    Returns ``(baud, carrier_hz, n_frames, frames)`` — the winning carrier per baud."""
    if carriers is None:
        sp = spectrum_summary(iq, fs)
        peak = round(sp["peak_hz"]) if sp else 0.0
        carriers = sorted({0.0, float(peak)})
    out: list[tuple[float, float, int, list[bytes]]] = []
    for baud in bauds:
        best: tuple[float, float, int, list[bytes]] = (float(baud), 0.0, 0, [])
        for carrier in carriers:
            # demodulate_capture (not raw demodulate): derotates the carrier + residual CFO and
            # polyphase-resamples to an integer target_sps before slicing — raw demodulate's timing
            # recovery DIVERGES at the channel's native sps (e.g. sps=40 for 1200 Bd over 48 kHz).
            bits = gfsk.demodulate_capture(
                iq, fs, symbol_rate_hz=baud, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
                carrier_hz=carrier)
            frames, _ = framings.deframe(bits, "ax25")
            if len(frames) > best[2]:
                best = (float(baud), float(carrier), len(frames), frames)
        out.append(best)
    return out


def demodulate_burst(
    iq: np.ndarray, fs: float, *, symbol_rate: float = DEFAULT_SYMBOL_RATE, guard_ms: float = 3.0
) -> np.ndarray:
    """Demodulate one burst (caller passes a slice incl. guard) to hard bits."""
    return gfsk.demodulate(iq, gfsk_params(fs, symbol_rate), recover_timing=True)


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


def analyze_file(
    path: str | Path, symbol_rate: float = DEFAULT_SYMBOL_RATE, sample_rate_hz: float = 0.0,
    *, run_ax25: bool = False, sweep_window_s: float = 60.0,
) -> None:
    cap = load_capture(path, sample_rate_hz)
    dur = len(cap.iq) / cap.fs if cap.fs else 0.0
    print(
        f"{Path(path).name}: {len(cap.iq):,} samples | fs={cap.fs:.0f} Hz | "
        f"center={cap.center_hz/1e6:.4f} MHz | dur={dur:.3f} s"
    )
    # Carrier check FIRST: is there ANY signal (weak/continuous too, not just bursts)? A NO CARRIER
    # verdict makes the demod/framing/baud debate moot — the capture is empty (freq/antenna/off).
    sp = spectrum_summary(cap.iq, cap.fs)
    if sp is not None:
        verdict = (f"CARRIER at {sp['peak_hz']:+.0f} Hz from DC"
                   if sp["snr_db"] >= CARRIER_SNR_DB else "NO CARRIER — flat noise (dead capture)")
        print(f"spectrum: strongest line {sp['peak_hz']:+.0f} Hz, {sp['snr_db']:.1f} dB over floor "
              f"→ {verdict}")
    if run_ax25:
        win = cap.iq if sweep_window_s <= 0 else cap.iq[: int(cap.fs * sweep_window_s)]
        span = len(win) / cap.fs if cap.fs else 0.0
        print(f"ax25 sweep (our deframer, G3RUH+plain, FCS-checked, carrier-recovered; "
              f"first {span:.0f}s):")
        for baud, carrier, nframes, frames in ax25_sweep(win, cap.fs):
            head = frames[0][:16].hex() if frames else "-"
            print(f"  {int(baud):5d} Bd @ carrier {carrier:+.0f} Hz: {nframes} FCS frame(s)  "
                  f"first={head}")
    bursts = find_bursts(cap.iq, cap.fs)
    guard = int(cap.fs * 0.003)
    print(f"{len(bursts)} bursts:")
    for k, (s, e) in enumerate(bursts):
        seg = cap.iq[max(0, s - guard) : e + guard]
        bits = demodulate_burst(seg, cap.fs, symbol_rate=symbol_rate)
        idx = find_sync(bits)
        fb = frame_bytes(bits[idx:]) if idx is not None else b""
        synced = f"sync@bit{idx}" if idx is not None else "NO-SYNC"
        head = fb[:12].hex() if fb else "-"
        print(
            f"  burst {k}: t={s/cap.fs:7.3f}s dur={(e-s)/cap.fs*1000:6.1f}ms "
            f"{synced:>10}  payload[0:12]={head}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="iq_analyze", description="EnduroSat UHF IQ analysis")
    p.add_argument("capture", help="capture file: .cf32 (raw, whole pass) or VSA .csv")
    p.add_argument("--symbol-rate", type=float, default=DEFAULT_SYMBOL_RATE)
    p.add_argument(
        "--sample-rate", type=float, default=0.0, help="cf32 sample rate if no sidecar (Hz)"
    )
    p.add_argument("--ax25", action="store_true",
                   help="run OUR AX.25 deframer (FCS-checked) sweeping standard bauds")
    p.add_argument("--sweep-window-s", type=float, default=60.0,
                   help="seconds of capture to run the --ax25 sweep on (0 = whole pass)")
    args = p.parse_args(argv)
    analyze_file(args.capture, symbol_rate=args.symbol_rate, sample_rate_hz=args.sample_rate,
                 run_ax25=args.ax25, sweep_window_s=args.sweep_window_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())

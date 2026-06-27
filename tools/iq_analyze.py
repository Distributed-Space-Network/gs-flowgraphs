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
from gfsk_ax25 import gfsk  # noqa: E402

DEFAULT_SYMBOL_RATE = 9600.0
DEFAULT_MOD_INDEX = 0.5  # dev +-2400 Hz at 9600 sym/s (per the lab capture)
DEFAULT_BT = 0.5
PREAMBLE_BITS = (1, 0)  # 0xAA = 1010..., MSB-first
SYNC_FLAG = (0, 1, 1, 1, 1, 1, 1, 0)  # 0x7E


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
    path: str | Path, symbol_rate: float = DEFAULT_SYMBOL_RATE, sample_rate_hz: float = 0.0
) -> None:
    cap = load_capture(path, sample_rate_hz)
    dur = len(cap.iq) / cap.fs if cap.fs else 0.0
    print(
        f"{Path(path).name}: {len(cap.iq):,} samples | fs={cap.fs:.0f} Hz | "
        f"center={cap.center_hz/1e6:.4f} MHz | dur={dur:.3f} s"
    )
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
    args = p.parse_args(argv)
    analyze_file(args.capture, symbol_rate=args.symbol_rate, sample_rate_hz=args.sample_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())

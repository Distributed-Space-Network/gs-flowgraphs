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
# Candidate baud rates for auto-detection. The declared/labelled baud CAN BE WRONG (a real pass
# labelled "9600" actually carried a 2400-baud bird), so the tool sweeps these and reports which one
# shows a real preamble. Bounded by the channel: a rate above the recording's Nyquist can't fit, so
# a 48 kHz capture caps meaningfully at ~19200 — sweeping MHz-range bauds is physically pointless.
SWEEP_BAUDS = (1200.0, 2400.0, 4800.0, 9600.0, 19200.0)
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
    iq: np.ndarray, fs: float, *, min_ms: float = 2.0, threshold_mult: float = 4.0,
    exclude_hz: float | None = None, exclude_bw_hz: float = 12000.0, nfft: int = 1024,
) -> list[tuple[int, int]]:
    """Return (start, end) sample indices of on-air bursts.

    Default (``exclude_hz is None``): a simple magnitude gate — fine when the wanted bursts are the
    only strong energy in the capture.

    ``exclude_hz`` (a strong CONTINUOUS off-channel carrier, e.g. a co-visible satellite): a plain
    ``|iq|`` gate is DEFEATED — the always-on carrier pins ``|iq|`` high the whole pass, so the
    median rises above the bursty data and NOTHING clears the threshold (the observed "0 bursts" on
    cmd_107). Instead detect on the OFF-INTERFERER spectral energy: an STFT with the interferer band
    zeroed, so only the bursty data drives detection — the burst view usable beside a carrier."""
    if exclude_hz is None:
        mag = np.abs(iq)
        thr = max(np.median(mag) * threshold_mult, mag.max() * 0.08)
        on = (mag > thr).astype(np.int8)
        d = np.diff(on, prepend=0, append=0)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        min_samp = fs * min_ms / 1000.0
        return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False) if (e - s) > min_samp]
    n = int(len(iq))
    if n < nfft:
        return []
    hop = nfft  # non-overlapping frames — burst edges to ~one frame is plenty for a raw view
    nframes = n // hop
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    keep = np.abs(freqs - float(exclude_hz)) >= float(exclude_bw_hz) / 2.0  # bins off interferer
    win = np.hanning(nfft)
    energy = np.empty(nframes, dtype=np.float64)
    for i in range(nframes):
        seg = np.asarray(iq[i * hop : (i + 1) * hop]) * win
        p = np.abs(np.fft.fft(seg)) ** 2
        energy[i] = float(np.sum(p[keep]))
    thr = float(np.median(energy)) * threshold_mult
    on = (energy > thr).astype(np.int8)
    d = np.diff(on, prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    min_frames = max(1, int((fs * min_ms / 1000.0) / hop))
    return [
        (int(s * hop), int(min(e * hop, n)))
        for s, e in zip(starts, ends, strict=False)
        if (e - s) >= min_frames
    ]


def _peak_excluding(
    iq: np.ndarray, fs: float, exclude_hz: float | None = None,
    exclude_bw_hz: float = 12000.0, *, snr_mult: float = 4.0,
) -> float | None:
    """The strongest spectral line in ``iq`` EXCLUDING the interferer band (``exclude_hz`` ±
    ``exclude_bw_hz``/2), or ``None`` when nothing there stands ``snr_mult``× over the median (a
    quiet/noise window). Used per-decode-window to find the DATA carrier while ignoring a loud
    continuous carrier — so the demod locks the bursty downlink, not the interferer."""
    n = int(len(iq))
    if n < 64:
        return None
    spec = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(iq) * np.hanning(n))))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    if exclude_hz is not None:
        spec = spec.copy()
        spec[np.abs(freqs - float(exclude_hz)) < float(exclude_bw_hz) / 2.0] = 0.0
    pk = int(np.argmax(spec))
    if spec[pk] <= (float(np.median(spec)) + 1e-30) * snr_mult:
        return None
    return float(freqs[pk])


def _longest_alt_run(bits: np.ndarray) -> int:
    """Length (in bits) of the longest strictly-alternating ``1010…`` / ``0101…`` run — the
    demodulated footprint of a 0xAA/0x55 modem PREAMBLE. A clean run well above the noise floor (a
    random stream gives runs of only ~8-12) is the tell that the demod is locked at the RIGHT baud
    and carrier, even when the framing/CRC that follows can't be validated (encrypted/whitened
    payload). Measured directly as the longest run of consecutive bit-changes (a regex like
    ``(?:01|10)+`` is WRONG — it accepts ``0110`` whose ``11`` seam is not alternating)."""
    b = np.asarray(bits, dtype=np.uint8)
    if len(b) < 2:
        return int(len(b))
    changes = b[1:] != b[:-1]  # True where the alternation continues
    if not changes.any():
        return 1
    brk = np.flatnonzero(~changes)  # positions where alternation breaks
    bounds = np.concatenate(([-1], brk, [len(changes)]))
    return int(np.max(np.diff(bounds) - 1)) + 1


def detect_baud(
    iq: np.ndarray, fs: float, *, carrier_hz: float = 0.0,
    candidates: tuple[float, ...] = SWEEP_BAUDS, channel_bw_hz: float = 0.0,
) -> list[tuple[float, int]]:
    """Rank candidate baud rates by the longest 0xAA-preamble run they produce (see
    :func:`_longest_alt_run`) — a label-independent baud detector. The declared baud can be wrong
    (a "9600" pass actually carried a 2400-baud signal); demodulating at each candidate and scoring
    the preamble reveals the true rate. Returns ``[(baud, alt_run_bits), …]`` in candidate order."""
    out: list[tuple[float, int]] = []
    for baud in candidates:
        ch = channel_bw_hz if channel_bw_hz > 0 else 2.0 * baud
        bits = gfsk.demodulate_capture(
            iq, fs, symbol_rate_hz=baud, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
            carrier_hz=carrier_hz, channel_bw_hz=ch, correct_cfo=True, recover_timing=False,
        )
        out.append((float(baud), _longest_alt_run(bits)))
    return out


def _strongest_burst_window(
    iq: np.ndarray, fs: float, exclude_hz: float | None,
    *, probe_s: float = 0.5, pad_s: float = 2.5,
) -> tuple[np.ndarray, float] | None:
    """Locate the strongest NON-interferer burst and return a WIDE window around it plus its carrier
    — the best place to run baud detection. A short ``probe_s`` scan finds the peak TIME; the
    returned window is then padded ``pad_s`` on each side so the WHOLE burst — crucially its leading
    0xAA preamble, which sits at the burst START, not at the strong mid-burst peak — is contained. A
    window that merely centres on the strong point clips the preamble and mis-detects the baud.
    ``None`` if nothing stands out."""
    n = int(len(iq))
    probe = max(1, int(fs * probe_s))
    freqs = np.fft.fftshift(np.fft.fftfreq(probe, d=1.0 / fs))
    hwin = np.hanning(probe)
    best: tuple[float, int, float] | None = None  # (amp, offset, carrier)
    for off in range(0, max(1, n - probe), probe):
        seg = np.asarray(iq[off : off + probe])
        pk = _peak_excluding(seg, fs, exclude_hz)
        if pk is None:
            continue
        spec = np.abs(np.fft.fftshift(np.fft.fft(seg * hwin)))
        amp = float(spec[int(np.argmin(np.abs(freqs - pk)))])
        if best is None or amp > best[0]:
            best = (amp, off, float(pk))
    if best is None:
        return None
    pad = int(fs * pad_s)
    lo = max(0, best[1] - pad)
    hi = min(n, best[1] + probe + pad)
    return (np.asarray(iq[lo:hi]), best[2])


def decode_pass(
    iq: np.ndarray, fs: float, symbol_rate: float = DEFAULT_SYMBOL_RATE,
    framings_to_try: tuple[str, ...] = ("ax25", "endurosat"),
    *, exclude_hz: float | None = None, exclude_bw_hz: float = 12000.0,
    channel_bw_hz: float = 0.0, window_s: float = 1.0, overlap: float = 0.5,
    carriers: list[float] | None = None, bauds: tuple[float, ...] | None = None,
) -> dict[str, dict]:
    """Whole-pass decode of a BURSTY GFSK downlink recorded next to a strong CONTINUOUS carrier.

    The single-window carrier-recovering sweep (:func:`framing_sweep`) demodulates one big window
    at one carrier — which fails here two ways: it can only lock ONE carrier (the loud continuous
    interferer wins the CFO/discriminator) and it covers only part of the pass. This slides SHORT
    windows over the ENTIRE capture and, per window, de-rotates the strongest NON-interferer peak
    (the data burst — which tracks Doppler window-to-window with no external track) to DC,
    CHANNEL-FILTERS to reject the interferer (``channel_bw_hz``; default ``2*symbol_rate``), demods,
    and runs each deframer (CRC-gated). DC is always also tried (a near-centre bird). Frames are
    deduped per framing by payload. Returns ``{framing: {"frames": [...], "carriers": {...}}}``.

    ``carriers`` forces an explicit per-window candidate list (``--carrier-hz``) instead of the
    auto {DC, peak-excluding-interferer}. ``bauds`` sweeps several symbol rates per window (the
    label can be wrong); ``None`` uses just ``symbol_rate``. The channel filter defaults to
    ``2*baud`` per swept baud, so a narrow low-baud signal is not drowned by a wide filter."""
    iq = np.asarray(iq, dtype=np.complex64)
    n = int(len(iq))
    baud_list = tuple(bauds) if bauds else (symbol_rate,)
    win = max(1, int(fs * window_s))
    step = max(1, int(win * (1.0 - overlap)))
    out: dict[str, dict] = {
        name: {"frames": [], "carriers": set(), "bauds": set()} for name in framings_to_try
    }
    seen: dict[str, set] = {name: set() for name in framings_to_try}
    for off in range(0, n, step):
        seg = np.asarray(iq[off : off + win])
        if len(seg) < win // 2:
            break
        if carriers is not None:
            cands: set[float] = {float(c) for c in carriers}
        else:
            cands = {0.0}
            pk = _peak_excluding(seg, fs, exclude_hz, exclude_bw_hz)
            if pk is not None:
                cands.add(float(round(pk)))
        for baud in baud_list:
            ch = channel_bw_hz if channel_bw_hz > 0.0 else 2.0 * baud
            for carrier in cands:
                bits = gfsk.demodulate_capture(
                    seg, fs, symbol_rate_hz=baud, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
                    carrier_hz=float(carrier), channel_bw_hz=ch,
                    correct_cfo=True, recover_timing=False,
                )
                if not len(bits):
                    continue
                for name in framings_to_try:
                    frames, _ = framings.deframe(bits, name)  # FCS/CRC-gated + ax25 addr-checked
                    for f in frames:
                        h = f.hex()
                        if h in seen[name]:
                            continue
                        seen[name].add(h)
                        out[name]["frames"].append(f)
                        out[name]["carriers"].add(int(carrier))
                        out[name]["bauds"].add(int(baud))
    return out


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
    *, carriers=None, target_sps: int = 8, channel_bw_hz: float = 0.0,
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
                    carrier_hz=carrier, target_sps=target_sps, channel_bw_hz=channel_bw_hz)
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
    iq: np.ndarray, fs: float, *, symbol_rate: float = DEFAULT_SYMBOL_RATE, carrier_hz: float = 0.0,
    channel_bw_hz: float = 0.0,
) -> np.ndarray:
    """Demodulate one burst (caller passes a slice incl. guard) to hard bits, DE-ROTATING
    ``carrier_hz`` to DC first. Doppler + the bird's fixed oscillator offset park the carrier
    outside the narrow demod filter, so a raw demod there gives NO-SYNC — the caller estimates the
    offset (from the spectrum, or --carrier-hz) and passes it here; ``correct_cfo`` then cleans the
    residual. ``channel_bw_hz`` > 0 channel-selects at DC to reject an off-channel interferer (a
    loud continuous carrier). Via ``demodulate_capture`` this also polyphase-resamples, so a
    non-integer samples/symbol rate (e.g. 500 kHz / 9600) still demods."""
    return gfsk.demodulate_capture(
        iq, fs, symbol_rate_hz=symbol_rate, mod_index=DEFAULT_MOD_INDEX, bt=DEFAULT_BT,
        carrier_hz=carrier_hz, channel_bw_hz=channel_bw_hz, correct_cfo=True, recover_timing=False,
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


def analyze_file(
    path: str | Path, symbol_rate: float = DEFAULT_SYMBOL_RATE, sample_rate_hz: float = 0.0,
    *, run_ax25: bool = False, run_endurosat: bool = False, sweep_window_s: float = 40.0,
    carrier_hz: float | None = None, want_waterfall: bool = False, channel_bw_hz: float = 0.0,
    interferer_hz: float | None = None, no_interferer_exclude: bool = False,
    decode_window_s: float = 1.0, max_burst_list: int = 40, sweep_baud: bool = True,
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
    # Carrier check: is there ANY signal (weak/continuous too, not just bursts)? A NO CARRIER
    # verdict makes the demod/framing/baud debate moot — the capture is empty (freq/antenna/off).
    sp = spectrum_summary(cap.iq, cap.fs)
    if sp is not None:
        verdict = (f"CARRIER at {sp['peak_hz']:+.0f} Hz from DC"
                   if sp["snr_db"] >= CARRIER_SNR_DB else "NO CARRIER - flat noise (dead capture)")
        print(f"spectrum: strongest line {sp['peak_hz']:+.0f} Hz, {sp['snr_db']:.1f} dB over floor "
              f"-> {verdict}")
    # A commercial-band EnduroSat downlink is BURSTS ONLY (band research); a strong CONTINUOUS
    # Doppler-tracking line is a co-visible satellite tens of kHz away, NOT our data. Treat it as an
    # off-channel INTERFERER: exclude it from carrier estimation and channel-filter it out — else it
    # captures the CFO/discriminator and every burst decodes as noise. --interferer-hz overrides the
    # auto pick (the loud continuous line); --no-exclude-interferer turns it off (clean captures).
    interferer: float | None = None
    if not no_interferer_exclude:
        if interferer_hz is not None:
            interferer = float(interferer_hz)
        elif sp is not None and sp["snr_db"] >= CARRIER_SNR_DB:
            interferer = float(round(sp["peak_hz"]))
    if interferer is not None:
        print(f"  -> treating {interferer:+.0f} Hz as an off-channel interferer (continuous carrier"
              " / co-visible sat); channel-filtering it out, decoding the bursty data")
    selected = [f for f, on in (("ax25", run_ax25), ("endurosat", run_endurosat)) if on]
    if not selected:  # no framing flag → decode BOTH light framings (what a labelled pass carries)
        selected = ["ax25", "endurosat"]
    # BAUD DETECTION: the labelled/declared baud can be WRONG (a pass labelled 9600 actually carried
    # a 2400-baud bird). Find the strongest off-interferer burst and report the 0xAA-preamble run
    # per candidate baud — a run >> the ~10-bit noise level flags the true rate even when the
    # framing that follows is encrypted/whitened and won't validate. Label-independent ground truth.
    sweep_bauds: tuple[float, ...] | None = None
    if sweep_baud:
        strong = _strongest_burst_window(cap.iq, cap.fs, interferer)
        if strong is not None:
            wseg, wcar = strong
            ranked = sorted(detect_baud(wseg, cap.fs, carrier_hz=wcar), key=lambda r: -r[1])
            print(f"baud detect (0xAA-preamble run/baud @ strongest burst carrier {wcar:+.0f} Hz; "
                  "run>>10 = real):")
            for b, r in ranked:
                flag = "  <- likely TRUE baud" if r >= 32 and r == ranked[0][1] else ""
                print(f"  {int(b):6d} Bd: preamble-run {r:3d} bits{flag}")
            top_baud, top_run = ranked[0]
            if top_run >= 32 and abs(top_baud - symbol_rate) > 1:
                print(f"  NOTE: detected baud {int(top_baud)} != labelled {int(symbol_rate)} - "
                      "decoding will sweep both.")
            sweep_bauds = tuple(sorted({symbol_rate, *[b for b, r in ranked if r >= 24]}))
    # PRIMARY decode: whole-pass, short-window, channel-filtered, interferer-excluding. Recovers the
    # bursty downlink frames across the ENTIRE pass (the single-window sweep only saw one carrier /
    # one slice). --carrier-hz forces the data carrier; else it is estimated per window. Sweeps the
    # detected candidate bauds (CRC-gated, so a wrong baud yields nothing).
    forced = None if carrier_hz is None else [float(carrier_hz)]
    ch = channel_bw_hz if channel_bw_hz > 0 else 2.0 * symbol_rate
    bshow = ("/".join(str(int(b)) for b in sweep_bauds) if sweep_bauds else int(symbol_rate))
    print(f"whole-pass decode ({bshow} Bd, channel~{ch/1e3:.1f} kHz, "
          f"{decode_window_s:g}s windows, CRC-gated; framings={','.join(selected)}):")
    res = decode_pass(
        cap.iq, cap.fs, symbol_rate, tuple(selected), exclude_hz=interferer,
        channel_bw_hz=channel_bw_hz, window_s=decode_window_s, carriers=forced, bauds=sweep_bauds,
    )
    for name in selected:
        frames = res[name]["frames"]
        carriers = sorted(res[name]["carriers"])
        head = frames[0][:16].hex() if frames else "-"
        cshow = f" carriers~{[f'{c:+d}' for c in carriers[:6]]}" if carriers else ""
        print(f"  {name}: {len(frames)} frame(s){cshow}  first={head}")
    # Raw burst view (diagnostic): interferer-aware detection + per-burst carrier + channel filter.
    bursts = find_bursts(cap.iq, cap.fs, exclude_hz=interferer)
    guard = int(cap.fs * 0.003)
    shown = min(len(bursts), max_burst_list)
    print(f"{len(bursts)} bursts (showing {shown}; per-burst carrier, channel-filtered):")
    for k, (s, e) in enumerate(bursts[:max_burst_list]):
        seg = cap.iq[max(0, s - guard) : e + guard]
        # Per-burst DATA carrier: the burst's own strongest line (excluding the interferer), so each
        # burst is de-rotated to its Doppler-current frequency — not a single whole-pass carrier.
        bc = carrier_hz if carrier_hz is not None else _peak_excluding(seg, cap.fs, interferer)
        bits = demodulate_burst(seg, cap.fs, symbol_rate=symbol_rate,
                                carrier_hz=float(bc or 0.0), channel_bw_hz=ch)
        idx = find_sync(bits)
        fb = frame_bytes(bits[idx:]) if idx is not None else b""
        synced = f"sync@bit{idx}" if idx is not None else "NO-SYNC"
        print(
            f"  burst {k}: t={s/cap.fs:7.3f}s dur={(e-s)/cap.fs*1000:6.1f}ms "
            f"carrier={float(bc or 0):+7.0f}Hz {synced:>10}  frame={fb[:24].hex() if fb else '-'}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="iq_analyze", description="EnduroSat UHF IQ analysis")
    p.add_argument("capture", help="capture file: .cf32 (raw, whole pass) or VSA .csv")
    p.add_argument("--symbol-rate", type=float, default=DEFAULT_SYMBOL_RATE)
    p.add_argument(
        "--sample-rate", type=float, default=0.0, help="cf32 sample rate if no sidecar (Hz)"
    )
    p.add_argument("--ax25", action="store_true",
                   help="decode ONLY AX.25 (FCS-checked); no framing flag = both ax25+endurosat")
    p.add_argument("--endurosat", action="store_true",
                   help="decode ONLY EnduroSat chip-packet framing (CRC-16)")
    p.add_argument("--sweep-window-s", type=float, default=40.0,
                   help="(legacy) unused by the whole-pass decoder; kept for compatibility")
    p.add_argument("--carrier-hz", type=float, default=None,
                   help="force this exact data-carrier offset (Hz) instead of per-window estimate")
    p.add_argument("--channel-bw", type=float, default=0.0,
                   help="channel-select bandwidth Hz to reject an off-channel carrier (0=2*baud)")
    p.add_argument("--interferer-hz", type=float, default=None,
                   help="Hz of a continuous carrier to reject (default: the loud continuous line)")
    p.add_argument("--no-exclude-interferer", action="store_true",
                   help="do NOT treat the loud continuous line as interference (clean captures)")
    p.add_argument("--decode-window-s", type=float, default=1.0,
                   help="whole-pass decode window (s); short -> Doppler ~const per window")
    p.add_argument("--no-sweep-baud", action="store_true",
                   help="trust --symbol-rate; skip auto baud detection/sweep (1200..19200)")
    p.add_argument("--waterfall", action="store_true",
                   help="write a colored spectrogram <capture>.analyze.png (needs matplotlib)")
    args = p.parse_args(argv)
    analyze_file(args.capture, symbol_rate=args.symbol_rate, sample_rate_hz=args.sample_rate,
                 run_ax25=args.ax25, run_endurosat=args.endurosat,
                 sweep_window_s=args.sweep_window_s, channel_bw_hz=args.channel_bw,
                 interferer_hz=args.interferer_hz,
                 no_interferer_exclude=args.no_exclude_interferer,
                 decode_window_s=args.decode_window_s, sweep_baud=not args.no_sweep_baud,
                 carrier_hz=args.carrier_hz, want_waterfall=args.waterfall)
    return 0


if __name__ == "__main__":
    sys.exit(main())

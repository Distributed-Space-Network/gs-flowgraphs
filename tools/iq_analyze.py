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

The default is a REAL whole-pass sweep: every physically usable baud in
``SWEEP_BAUDS`` is decoded, and each time window tries DC, the strongest narrow
carrier, and broadband packet candidates.  This matters when a weak burst is
visible beside a much stronger Doppler track: the old implementation probed all
bauds but decoded at most three of them, then locked only the strongest line.
With ``--raw-bits-dir``, strict locks also receive a bounded carrier/filter/SPS
refinement; separate bursts at the same baud are ranked by checksum evidence or
payload-agnostic bit correlation.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
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
DEFAULT_SWEEP_BAUDS = (1200.0, 2400.0, 4800.0, 9600.0, 19200.0)
# Candidate baud rates for auto-detection. The declared/labelled baud CAN BE WRONG (a real pass
# labelled "9600" actually carried a 2400-baud bird), so the tool sweeps these and reports which one
# shows a real preamble. Bounded by the channel: a rate above the recording's Nyquist can't fit, so
# a 48 kHz capture caps meaningfully at ~19200 — sweeping MHz-range bauds is physically pointless.
SWEEP_BAUDS = DEFAULT_SWEEP_BAUDS
# A spectral line this many dB over the noise floor counts as a real carrier (vs a spur/noise bin).
CARRIER_SNR_DB = 6.0
# Per-window carrier recovery keeps the strongest narrow line AND packet-band energy peaks.  The
# latter is how a weaker 2400-baud burst beside a bright CW/Doppler track is found.
DEFAULT_CARRIER_COUNT = 3
BROAD_CARRIER_SNR_DB = 1.0
# Raw hard decisions are persisted without a valid CRC only when there is independent modem-lock
# evidence.  The alternating-run threshold is derived from the number of sliced bits, keeping the
# per-demod random false-alarm probability bounded instead of baking in one capture's preamble.
RAW_ALT_FALSE_ALARM = 5e-5
# A lock earns a bounded second pass over nearby carrier/filter/resampler settings.  Weak secondary
# preamble anchors are allowed only inside an already-established event, then ranked by CRC/FCS or
# consistency with another burst at the same baud.  This improves damaged repeated beacons without
# teaching the analyzer any capture-specific payload bytes.
RAW_REFINE_CARRIER_FRACTIONS = (-0.25, -1 / 6, -1 / 12, 0.0, 1 / 12, 1 / 6, 0.25)
RAW_REFINE_BW_MULTIPLIERS = (1.25, 4 / 3, 1.5, 2.0)
RAW_REFINE_TARGET_SPS = (8, 16)
RAW_REFINE_MAX_LOCKS = 8
RAW_REFINE_MAX_ANCHORS = 2
RAW_REFINE_MAX_VARIANTS = 256
RAW_REPEAT_MIN_SIMILARITY = 0.65
RAW_REPEAT_TIE_TOLERANCE = 0.005


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
    if len(iq) == 0:
        return []
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


def _window_spectrum(iq: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequency_hz, power)`` for one decode window.

    Kept separate so :func:`decode_pass` computes the FFT once per time window and reuses it for
    every baud.  A full five-rate sweep used to repeat carrier work unnecessarily.
    """
    n = int(len(iq))
    if n < 64:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    x = np.asarray(iq) * np.hanning(n)
    power = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    return freqs, power


def _carrier_candidates(
    iq: np.ndarray,
    fs: float,
    *,
    channel_bw_hz: float,
    max_candidates: int = DEFAULT_CARRIER_COUNT,
    exclude_hz: float | None = None,
    exclude_bw_hz: float = 12000.0,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[float]:
    """Find narrow-line and BROADBAND packet carrier candidates in one window.

    Selecting only the largest FFT bin is wrong in the common case of a weak packet beside a much
    stronger CW/Doppler line.  We therefore keep that narrow peak for compatibility, then search a
    clipped-power rolling band.  Clipping makes a one-bin CW line contribute almost nothing to a
    multi-kHz band while a real FSK/GFSK packet contributes across hundreds/thousands of bins.
    The returned list is bounded, so the extra coverage cannot become an unbounded frequency grid.
    """
    limit = max(0, int(max_candidates))
    if limit == 0:
        return []
    freqs, power = spectrum if spectrum is not None else _window_spectrum(iq, fs)
    if not len(power):
        return []
    valid = np.isfinite(power)
    # Leave a small FFT-edge guard: a channel filter cannot recover a packet centred at Nyquist.
    edge = max(50.0, float(channel_bw_hz) / 2.0)
    valid &= np.abs(freqs) <= max(0.0, fs / 2.0 - edge)
    if exclude_hz is not None:
        valid &= np.abs(freqs - float(exclude_hz)) >= float(exclude_bw_hz) / 2.0
    vals = power[valid]
    if not len(vals):
        return []
    floor = float(np.median(vals)) + 1e-30
    out: list[float] = []

    # Narrow candidate: preserves the old carrier recovery for an ordinary isolated signal.
    narrow = np.where(valid, power, 0.0)
    pk = int(np.argmax(narrow))
    # _peak_excluding historically used 4x amplitude over median amplitude.  Power is squared, so
    # use 16x here.  The broadband path below has its own much lower integrated-energy threshold.
    if narrow[pk] >= floor * 16.0:
        out.append(float(freqs[pk]))

    if len(out) >= limit or channel_bw_hz <= 0.0:
        return out[:limit]

    # Packet-band candidate(s).  Use ~75% of the channel width (1.5*baud for the default 2*baud
    # channel) so a 2-FSK/GFSK occupied band raises the average without diluting it excessively.
    df = abs(float(freqs[1] - freqs[0])) if len(freqs) > 1 else fs
    band_bins = max(8, min(len(power) - 1, int(round(0.75 * channel_bw_hz / max(df, 1e-9)))))
    clipped = np.minimum(power, floor * 16.0)
    clipped = np.where(valid, clipped, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(clipped, dtype=np.float64)))
    band = csum[band_bins:] - csum[:-band_bins]
    centres = np.arange(len(band)) + band_bins // 2
    band_valid = valid[centres]
    if not np.any(band_valid):
        return out[:limit]
    baseline = float(np.median(band[band_valid])) + 1e-30
    threshold = baseline * (10.0 ** (BROAD_CARRIER_SNR_DB / 10.0))
    work = np.where(band_valid, band, 0.0)
    separation_bins = max(1, band_bins // 2)
    while len(out) < limit:
        k = int(np.argmax(work))
        if work[k] < threshold:
            break
        centre_bin = int(centres[k])
        carrier = float(freqs[centre_bin])
        # A broadband estimate that lands on the already-kept narrow line adds no coverage.
        if all(abs(carrier - old) >= max(100.0, channel_bw_hz / 4.0) for old in out):
            out.append(carrier)
        lo, hi = max(0, k - separation_bins), min(len(work), k + separation_bins + 1)
        work[lo:hi] = 0.0
    return out[:limit]


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


def _longest_alt_span(bits: np.ndarray) -> tuple[int, int]:
    """``(start, end_exclusive)`` of the longest strictly alternating run."""
    spans = _alternating_spans(bits, max_spans=1)
    return spans[0] if spans else (0, 0)


def _alternating_spans(
    bits: np.ndarray, *, min_bits: int = 1, max_spans: int = 0,
) -> list[tuple[int, int]]:
    """Alternating spans ordered by decreasing length, then earliest occurrence.

    Refinement deliberately considers more than the single longest span: a damaged preamble can
    split into two shorter runs, and the first equally-long random run is not necessarily the frame
    anchor.  The initial lock gate remains strict; weaker spans are used only inside that lock.
    """
    b = np.asarray(bits, dtype=np.uint8)
    if not len(b):
        return []
    if len(b) == 1:
        return [(0, 1)] if min_bits <= 1 else []
    breaks = np.flatnonzero(b[1:] == b[:-1])  # break lies between i and i+1
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(b)]))
    lengths = ends - starts
    order = sorted(
        (int(i) for i in np.flatnonzero(lengths >= max(1, int(min_bits)))),
        key=lambda i: (-int(lengths[i]), int(starts[i])),
    )
    if max_spans > 0:
        order = order[: int(max_spans)]
    return [(int(starts[i]), int(ends[i])) for i in order]


def _max_hdlc_flag_run(bits: np.ndarray) -> int:
    """Longest consecutive 0x7e flag train after plain and G3RUH AX.25 transforms.

    A single bare 0x7e is common in random hard decisions.  A sustained flag train is independent
    evidence of symbol/framing lock and is therefore useful for deciding whether raw bits deserve
    to be persisted even when the final FCS is bad.
    """
    from gfsk_ax25 import g3ruh  # noqa: PLC0415

    best = 0
    arr = np.asarray(bits, dtype=np.uint8)
    for scramble in (False, True):
        decoded = g3ruh.descramble(arr) if scramble else arr
        decoded = g3ruh.nrzi_decode(decoded)
        s = "".join(map(str, decoded.tolist()))
        for m in re.finditer(r"(?:01111110){2,}", s):
            best = max(best, (m.end() - m.start()) // 8)
    return best


def _alt_lock_threshold(bit_count: int, override: int = 0) -> int:
    """Alternating-run length unlikely to occur by chance in ``bit_count`` random bits."""
    if override > 0:
        return int(override)
    n = max(2, int(bit_count))
    return max(16, int(math.ceil(1.0 + math.log2(n / RAW_ALT_FALSE_ALARM))))


def _raw_lock_candidate(
    bits: np.ndarray,
    *,
    frames: list[tuple[str, bytes]],
    min_alt_run: int = 0,
) -> dict[str, object] | None:
    """Describe a defensible raw-bit lock, or return ``None`` for arbitrary sliced noise."""
    arr = np.asarray(bits, dtype=np.uint8)
    start, end = _longest_alt_span(arr)
    alt_run = end - start
    alt_threshold = _alt_lock_threshold(len(arr), min_alt_run)
    hdlc_flags = _max_hdlc_flag_run(arr)
    reasons: list[str] = []
    if frames:
        reasons.append("crc_valid_frame")
    if alt_run >= alt_threshold:
        reasons.append("alternating_preamble")
    if hdlc_flags >= 4:
        reasons.append("hdlc_flag_train")
    if not reasons:
        return None

    # A payload's first bit can continue the alternating run, so the likely candidate starts one
    # bit before the detected break.  The .rawbits file still contains the COMPLETE demod window;
    # this is only an offset into it, not a protocol-specific extraction length.
    candidate_start = max(0, end - 1)
    return {
        "reasons": reasons,
        "alternating_run_bits": int(alt_run),
        "alternating_run_threshold_bits": int(alt_threshold),
        "alternating_run_start_bit": int(start),
        "alternating_run_end_bit": int(end),
        "candidate_start_bit": int(candidate_start),
        "max_consecutive_hdlc_flags": int(hdlc_flags),
        "validated_frames": [
            {"framing": name, "payload_hex": body.hex()} for name, body in frames
        ],
    }


def _select_raw_candidates(
    candidates: list[dict[str, object]],
    *,
    window_s: float,
    max_files: int = 20,
) -> list[dict[str, object]]:
    """Return the strongest non-overlapping physical lock events."""
    ranked = sorted(
        candidates,
        key=lambda c: (
            bool(c.get("validated_frames")),
            float(c.get("repeat_similarity", 0.0)),
            int(c.get("alternating_run_bits", 0)),
            int(c.get("max_consecutive_hdlc_flags", 0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, object]] = []
    for cand in ranked:
        if len(kept) >= max(1, int(max_files)):
            break
        # Overlapping decode windows around one physical packet produce near-identical candidates.
        if any(
            int(old["baud"]) == int(cand["baud"])
            and abs(float(old["time_s"]) - float(cand["time_s"])) < 0.75 * window_s
            for old in kept
        ):
            continue
        kept.append(cand)
    return kept


def _persist_raw_candidates(
    candidates: list[dict[str, object]],
    output_dir: str | Path,
    *,
    window_s: float,
    max_files: int = 20,
) -> list[Path]:
    """Keep the strongest non-overlapping locks and write ASCII raw bits + JSON metadata."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = _select_raw_candidates(candidates, window_s=window_s, max_files=max_files)

    written: list[Path] = []
    for cand in sorted(kept, key=lambda c: float(c["time_s"])):
        sign = "p" if float(cand["carrier_hz"]) >= 0 else "m"
        carrier = abs(int(round(float(cand["carrier_hz"]))))
        stem = (
            f"rawbits_t{float(cand['time_s']):010.3f}_b{int(cand['baud'])}_"
            f"c{sign}{carrier:05d}"
        )
        bits_path = out_dir / f"{stem}.rawbits"
        meta_path = out_dir / f"{stem}.json"
        arr = np.asarray(cand["_bits"], dtype=np.uint8)
        metadata = {k: v for k, v in cand.items() if not k.startswith("_")}
        bits_path.write_text("".join(map(str, arr.tolist())) + "\n", encoding="ascii")
        meta_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.extend((bits_path, meta_path))
    return written


def _refined_anchor_evidence(
    bits: np.ndarray,
    *,
    frames: list[tuple[str, bytes]],
    min_alt_run: int,
) -> list[dict[str, object]]:
    """Candidate anchors inside an event that already passed the strict raw-lock gate."""
    arr = np.asarray(bits, dtype=np.uint8)
    alt_threshold = _alt_lock_threshold(len(arr), min_alt_run)
    weak_threshold = max(12, alt_threshold // 2)
    spans = _alternating_spans(
        arr, min_bits=weak_threshold, max_spans=RAW_REFINE_MAX_ANCHORS
    )
    hdlc_flags = _max_hdlc_flag_run(arr)
    if not spans and not frames and hdlc_flags < 4:
        return []
    if not spans:  # A checksum/flag lock is still useful even without an alternating preamble.
        spans = [(0, 0)]
    strongest = _longest_alt_run(arr)
    out: list[dict[str, object]] = []
    for start, end in spans:
        alt_run = end - start
        reasons: list[str] = []
        if frames:
            reasons.append("crc_valid_frame")
        if alt_run >= alt_threshold:
            reasons.append("alternating_preamble")
        elif alt_run:
            reasons.append("refined_preamble_anchor")
        if hdlc_flags >= 4:
            reasons.append("hdlc_flag_train")
        out.append({
            "reasons": reasons,
            "alternating_run_bits": int(alt_run),
            "strongest_alternating_run_bits": int(strongest),
            "alternating_run_threshold_bits": int(alt_threshold),
            "alternating_run_start_bit": int(start),
            "alternating_run_end_bit": int(end),
            "candidate_start_bit": int(max(0, end - 1)),
            "max_consecutive_hdlc_flags": int(hdlc_flags),
            "validated_frames": [
                {"framing": name, "payload_hex": body.hex()} for name, body in frames
            ],
        })
    return out


def _repeat_correlation_bits(baud: float) -> int:
    """Comparison span for repeated bursts: a quarter-second, bounded for cost/statistics."""
    return min(1024, max(256, int(round(float(baud) * 0.25))))


def _refine_raw_candidates(
    iq: np.ndarray,
    fs: float,
    candidates: list[dict[str, object]],
    framings_to_try: tuple[str, ...],
    *,
    window_s: float,
    overlap: float,
    capture_offset_s: float,
    channel_bw_hz: float,
    allow_carrier_refine: bool,
    mod_index: float,
    bt: float,
    target_sps: int,
    correct_cfo: bool,
    recover_timing: bool,
    raw_min_alt_run: int,
) -> tuple[list[dict[str, object]], list[tuple[str, bytes, float, float]]]:
    """Refine established locks and rank repeated bursts without known payload bytes.

    The expensive ensemble runs only after the strict first pass has found a physical event.  Each
    event tries nearby carrier centres, packet-width filters, 8/16 SPS, and the adjacent overlapping
    window.  A checksum wins outright; otherwise candidate anchors are compared across separate
    events at the same baud.  Strong repeated-burst agreement is an objective ranking signal even
    when every copy has a damaged CRC.
    """
    if not candidates:
        return [], []
    iq = np.asarray(iq, dtype=np.complex64)
    n = int(len(iq))
    win = max(1, int(fs * window_s))
    step = max(1, int(win * (1.0 - overlap)))
    events = _select_raw_candidates(
        candidates, window_s=window_s, max_files=RAW_REFINE_MAX_LOCKS
    )
    variants_by_event: list[list[dict[str, object]]] = []
    discoveries: list[tuple[str, bytes, float, float]] = []

    for event_index, base in enumerate(events):
        baud = float(base["baud"])
        base_carrier = float(base["carrier_hz"])
        base_off = int(base.get("_window_offset_samples", 0))
        segments = [(base_off, min(n, base_off + win))]
        previous = (max(0, base_off - step), min(n, base_off + win))
        if previous not in segments and previous[1] - previous[0] >= win // 2:
            segments.append(previous)
        offsets = (
            tuple(float(f) * baud for f in RAW_REFINE_CARRIER_FRACTIONS)
            if allow_carrier_refine else (0.0,)
        )
        widths = {float(channel_bw_hz)} if channel_bw_hz > 0.0 else set()
        widths.update(float(m) * baud for m in RAW_REFINE_BW_MULTIPLIERS)
        widths = {w for w in widths if 0.0 < w < fs}
        sps_values = sorted({max(2, int(target_sps)), *RAW_REFINE_TARGET_SPS})
        variants: list[dict[str, object]] = []

        for lo, hi in segments:
            seg = np.asarray(iq[lo:hi])
            segment_start_s = float(capture_offset_s + lo / fs)
            for carrier_offset in offsets:
                carrier = base_carrier + carrier_offset
                for width in sorted(widths):
                    for sps in sps_values:
                        bits = gfsk.demodulate_capture(
                            seg, fs, symbol_rate_hz=baud, mod_index=mod_index, bt=bt,
                            target_sps=sps, carrier_hz=carrier, channel_bw_hz=width,
                            correct_cfo=correct_cfo, recover_timing=recover_timing,
                        )
                        if not len(bits):
                            continue
                        validated: list[tuple[str, bytes]] = []
                        for name in framings_to_try:
                            frames, _ = framings.deframe(bits, name)
                            validated.extend((name, body) for body in frames)
                            discoveries.extend((name, body, carrier, baud) for body in frames)
                        anchors = _refined_anchor_evidence(
                            bits, frames=validated, min_alt_run=raw_min_alt_run
                        )
                        for evidence in anchors:
                            start_bit = int(evidence["candidate_start_bit"])
                            evidence.update({
                                "capture_time_basis": "seconds from capture start",
                                "time_s": segment_start_s + start_bit / baud,
                                "window_start_time_s": segment_start_s,
                                "preamble_end_time_s": segment_start_s
                                + int(evidence["alternating_run_end_bit"]) / baud,
                                "window_s": float((hi - lo) / fs),
                                "baud": int(round(baud)),
                                "carrier_hz": float(carrier),
                                "channel_bw_hz": float(width),
                                "mod_index": float(mod_index),
                                "bt": float(bt),
                                "target_sps": int(sps),
                                "correct_cfo": bool(correct_cfo),
                                "recover_timing": bool(recover_timing),
                                "polarity": 0,
                                "bit_count": int(len(bits)),
                                "bit_format": "ASCII 0/1 hard decisions; first char is bit 0",
                                "refined": True,
                                "_bits": np.asarray(bits, dtype=np.uint8),
                                "_event_index": int(event_index),
                                "_base_carrier_hz": float(base_carrier),
                                "_window_offset_samples": int(lo),
                            })
                            variants.append(evidence)

        # Repeated parameter settings often produce byte-identical hard decisions.  Collapse them
        # before the correlation matrix so duplicates cannot vote themselves into first place.
        unique: list[dict[str, object]] = []
        seen_slices: set[bytes] = set()
        compare_bits = _repeat_correlation_bits(baud)
        for variant in variants:
            start = int(variant["candidate_start_bit"])
            arr = np.asarray(variant["_bits"], dtype=np.uint8)
            sample = arr[start : start + compare_bits]
            if len(sample) < compare_bits:
                continue
            key = np.packbits(sample, bitorder="big").tobytes()
            if key in seen_slices:
                continue
            seen_slices.add(key)
            unique.append(variant)
            if len(unique) >= RAW_REFINE_MAX_VARIANTS:
                break
        variants_by_event.append(unique or [base])

    # Compare only distinct physical events at the same baud.  Mapping bits to +/-1 makes the dot
    # product an exact Hamming similarity, and abs(dot) naturally accepts discriminator inversion.
    for left_index, left in enumerate(variants_by_event):
        for right_index in range(left_index + 1, len(variants_by_event)):
            right = variants_by_event[right_index]
            if not left or not right or int(left[0]["baud"]) != int(right[0]["baud"]):
                continue
            nbits = _repeat_correlation_bits(float(left[0]["baud"]))
            left_slices = []
            right_slices = []
            for variant in left:
                start = int(variant["candidate_start_bit"])
                left_slices.append(
                    np.asarray(variant["_bits"], dtype=np.uint8)[start : start + nbits]
                )
            for variant in right:
                start = int(variant["candidate_start_bit"])
                right_slices.append(
                    np.asarray(variant["_bits"], dtype=np.uint8)[start : start + nbits]
                )
            if any(len(x) != nbits for x in (*left_slices, *right_slices)):
                continue
            left_pm = 1 - 2 * np.stack(left_slices).astype(np.int16)
            right_pm = 1 - 2 * np.stack(right_slices).astype(np.int16)
            agreement = left_pm @ right_pm.T
            for i, variant in enumerate(left):
                j = int(np.argmax(np.abs(agreement[i])))
                score = int(agreement[i, j])
                similarity = (nbits + abs(score)) / (2.0 * nbits)
                if similarity > float(variant.get("_repeat_similarity", 0.0)):
                    variant["_repeat_similarity"] = similarity
                    variant["_repeat_peer"] = right[j]
                    variant["_repeat_bits"] = nbits
                    variant["_repeat_inverted"] = score < 0
            for j, variant in enumerate(right):
                i = int(np.argmax(np.abs(agreement[:, j])))
                score = int(agreement[i, j])
                similarity = (nbits + abs(score)) / (2.0 * nbits)
                if similarity > float(variant.get("_repeat_similarity", 0.0)):
                    variant["_repeat_similarity"] = similarity
                    variant["_repeat_peer"] = left[i]
                    variant["_repeat_bits"] = nbits
                    variant["_repeat_inverted"] = score < 0

    selected: list[dict[str, object]] = []
    for event_index, variants in enumerate(variants_by_event):
        best_repeat = max(float(v.get("_repeat_similarity", 0.0)) for v in variants)

        def rank(
            variant: dict[str, object],
            best_repeat: float = best_repeat,
        ) -> tuple[bool, int, float, int, int, float, float, int]:
            repeat = float(variant.get("_repeat_similarity", 0.0))
            flags = int(variant.get("max_consecutive_hdlc_flags", 0))
            if repeat < RAW_REPEAT_MIN_SIMILARITY:
                repeat_class = 0
            elif repeat >= best_repeat - RAW_REPEAT_TIE_TOLERANCE:
                repeat_class = 2  # statistically indistinguishable from this event's best
            else:
                repeat_class = 1
            return (
                bool(variant.get("validated_frames")),
                repeat_class,
                repeat if repeat_class == 1 else 0.0,
                flags if flags >= 4 else 0,
                int(variant.get("alternating_run_bits", 0)),
                -abs(
                    float(variant["carrier_hz"])
                    - float(variant.get("_base_carrier_hz", variant["carrier_hz"]))
                ),
                -float(variant["channel_bw_hz"]),
                -abs(int(variant["target_sps"]) - int(target_sps)),
            )

        best = max(variants, key=rank)
        similarity = float(best.get("_repeat_similarity", 0.0))
        if similarity >= RAW_REPEAT_MIN_SIMILARITY:
            reasons = list(best.get("reasons", []))
            if "repeated_burst_correlation" not in reasons:
                reasons.append("repeated_burst_correlation")
            best["reasons"] = reasons
            best["ensemble_repeat_similarity"] = similarity
        best["refinement"] = {
            "method": "bounded carrier/filter/SPS ensemble; CRC/FCS then repeated-burst ranking",
            "variants_evaluated": len(variants),
            "event_index": event_index,
        }
        selected.append(best)

    # Report correlation between the SELECTED outputs, not a discarded ensemble peer.  The larger
    # ensemble score above is still retained as selection provenance.
    for candidate in selected:
        baud = float(candidate["baud"])
        nbits = _repeat_correlation_bits(baud)
        start = int(candidate["candidate_start_bit"])
        bits = np.asarray(candidate["_bits"], dtype=np.uint8)[start : start + nbits]
        best_peer: dict[str, object] | None = None
        best_score = 0
        for peer in selected:
            if peer is candidate or int(peer["baud"]) != int(candidate["baud"]):
                continue
            peer_start = int(peer["candidate_start_bit"])
            peer_bits = np.asarray(peer["_bits"], dtype=np.uint8)[
                peer_start : peer_start + nbits
            ]
            if len(bits) != nbits or len(peer_bits) != nbits:
                continue
            score = int(np.sum((1 - 2 * bits.astype(np.int16))
                               * (1 - 2 * peer_bits.astype(np.int16))))
            if abs(score) > abs(best_score):
                best_score = score
                best_peer = peer
        if best_peer is None:
            continue
        similarity = (nbits + abs(best_score)) / (2.0 * nbits)
        if similarity < RAW_REPEAT_MIN_SIMILARITY:
            continue
        candidate["repeat_similarity"] = similarity
        candidate["repeat_correlation"] = {
            "compared_bits": nbits,
            "matching_bits": int(round(similarity * nbits)),
            "similarity": similarity,
            "inverted": best_score < 0,
            "peer_time_s": float(best_peer["time_s"]),
            "peer_carrier_hz": float(best_peer["carrier_hz"]),
        }
    return selected, discoveries


def _usable_sweep_bauds(fs: float, labelled: float) -> tuple[float, ...]:
    """All requested rates that fit the recorded channel, plus a usable nonstandard label."""
    rates = {float(b) for b in SWEEP_BAUDS if 0 < float(b) <= float(fs) / 2.0}
    if 0 < float(labelled) <= float(fs) / 2.0:
        rates.add(float(labelled))
    return tuple(sorted(rates))


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
    if n < 64:  # too short for _peak_excluding to find anything → skip baud detection
        return None
    # Clamp the probe to the capture length so hwin/freqs always match seg (a fixed 0.5 s probe on a
    # sub-0.5 s capture would broadcast-crash on `seg * hwin`). range keeps every seg full-probe.
    probe = min(max(1, int(fs * probe_s)), n)
    freqs = np.fft.fftshift(np.fft.fftfreq(probe, d=1.0 / fs))
    hwin = np.hanning(probe)
    best: tuple[float, int, float] | None = None  # (amp, offset, carrier)
    for off in range(0, n - probe + 1, probe):
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
    carrier_count: int = DEFAULT_CARRIER_COUNT,
    mod_index: float = DEFAULT_MOD_INDEX, bt: float = DEFAULT_BT, target_sps: int = 16,
    correct_cfo: bool = True, recover_timing: bool = False,
    raw_bits_dir: str | Path | None = None, capture_offset_s: float = 0.0,
    raw_min_alt_run: int = 0,
    refine_raw: bool = True,
    raw_outputs: list[Path] | None = None,
) -> dict[str, dict]:
    """Whole-pass decode of a BURSTY GFSK downlink recorded next to a strong CONTINUOUS carrier.

    The single-window carrier-recovering sweep (:func:`framing_sweep`) demodulates one big window
    at one carrier — which fails here two ways: it can only lock ONE carrier (the loud continuous
    interferer wins the CFO/discriminator) and it covers only part of the pass. This slides SHORT
    windows over the ENTIRE capture and, per window, de-rotates bounded narrow-line and broadband
    packet candidates (which track Doppler window-to-window with no external track) to DC,
    CHANNEL-FILTERS to reject the interferer (``channel_bw_hz``; default ``2*symbol_rate``), demods,
    and runs each deframer (CRC-gated). DC is always also tried (a near-centre bird). Frames are
    deduped per framing by payload. Returns ``{framing: {"frames": [...], "carriers": {...}}}``.

    ``carriers`` forces an explicit per-window candidate list (``--carrier-hz``) instead of the
    automatic DC/narrow/broadband candidates. ``bauds`` sweeps several symbol rates per window (the
    label can be wrong); ``None`` uses just ``symbol_rate``. The channel filter defaults to
    ``2*baud`` per swept baud, so a narrow low-baud signal is not drowned by a wide filter.
    When raw output is requested, ``refine_raw`` runs the bounded post-lock ensemble and uses
    repeated-burst correlation without any known payload bytes."""
    iq = np.asarray(iq, dtype=np.complex64)
    n = int(len(iq))
    baud_list = tuple(bauds) if bauds else (symbol_rate,)
    win = max(1, int(fs * window_s))
    step = max(1, int(win * (1.0 - overlap)))
    out: dict[str, dict] = {
        name: {"frames": [], "carriers": set(), "bauds": set()} for name in framings_to_try
    }
    seen: dict[str, set] = {name: set() for name in framings_to_try}
    raw_candidates: list[dict[str, object]] = []
    for off in range(0, n, step):
        seg = np.asarray(iq[off : off + win])
        if len(seg) < win // 2:
            break
        spectrum = _window_spectrum(seg, fs) if carriers is None else None
        for baud in baud_list:
            ch = channel_bw_hz if channel_bw_hz > 0.0 else 2.0 * baud
            if carriers is not None:
                cands: set[float] = {float(c) for c in carriers}
            else:
                cands = {0.0}
                cands.update(
                    float(round(c)) for c in _carrier_candidates(
                        seg, fs, channel_bw_hz=ch, max_candidates=carrier_count,
                        exclude_hz=exclude_hz, exclude_bw_hz=exclude_bw_hz,
                        spectrum=spectrum,
                    )
                )
            for carrier in sorted(cands):
                bits = gfsk.demodulate_capture(
                    seg, fs, symbol_rate_hz=baud, mod_index=mod_index, bt=bt,
                    target_sps=max(2, int(target_sps)),
                    carrier_hz=float(carrier), channel_bw_hz=ch,
                    correct_cfo=correct_cfo, recover_timing=recover_timing,
                )
                if not len(bits):
                    continue
                validated_here: list[tuple[str, bytes]] = []
                for name in framings_to_try:
                    frames, _ = framings.deframe(bits, name)  # FCS/CRC-gated + ax25 addr-checked
                    validated_here.extend((name, f) for f in frames)
                    for f in frames:
                        h = f.hex()
                        if h in seen[name]:
                            continue
                        seen[name].add(h)
                        out[name]["frames"].append(f)
                        out[name]["carriers"].add(int(carrier))
                        out[name]["bauds"].add(int(baud))
                if raw_bits_dir is not None:
                    evidence = _raw_lock_candidate(
                        bits, frames=validated_here, min_alt_run=raw_min_alt_run,
                    )
                    if evidence is not None:
                        _alt_start, alt_end = _longest_alt_span(bits)
                        window_start_s = float(capture_offset_s + off / fs)
                        candidate_start = int(evidence["candidate_start_bit"])
                        evidence.update({
                            "capture_time_basis": "seconds from capture start",
                            "time_s": window_start_s + candidate_start / float(baud),
                            "window_start_time_s": window_start_s,
                            "preamble_end_time_s": window_start_s + alt_end / float(baud),
                            "window_s": float(window_s),
                            "baud": int(round(baud)),
                            "carrier_hz": float(carrier),
                            "channel_bw_hz": float(ch),
                            "mod_index": float(mod_index),
                            "bt": float(bt),
                            "target_sps": int(target_sps),
                            "correct_cfo": bool(correct_cfo),
                            "recover_timing": bool(recover_timing),
                            "polarity": 0,
                            "bit_count": int(len(bits)),
                            "bit_format": "ASCII 0/1 hard decisions; first char is bit 0",
                            "_bits": np.asarray(bits, dtype=np.uint8).copy(),
                            "_window_offset_samples": int(off),
                        })
                        raw_candidates.append(evidence)
    if raw_bits_dir is not None and raw_candidates:
        raw_candidates = _select_raw_candidates(
            raw_candidates, window_s=window_s, max_files=20
        )
        if refine_raw:
            refine_input = raw_candidates[:RAW_REFINE_MAX_LOCKS]
            unrefined = raw_candidates[RAW_REFINE_MAX_LOCKS:]
            refined, discoveries = _refine_raw_candidates(
                iq, fs, refine_input, framings_to_try,
                window_s=window_s, overlap=overlap, capture_offset_s=capture_offset_s,
                channel_bw_hz=channel_bw_hz, allow_carrier_refine=carriers is None,
                mod_index=mod_index, bt=bt, target_sps=target_sps,
                correct_cfo=correct_cfo, recover_timing=recover_timing,
                raw_min_alt_run=raw_min_alt_run,
            )
            raw_candidates = refined + unrefined
            for name, body, carrier, baud in discoveries:
                h = body.hex()
                if h in seen[name]:
                    continue
                seen[name].add(h)
                out[name]["frames"].append(body)
                out[name]["carriers"].add(int(round(carrier)))
                out[name]["bauds"].add(int(round(baud)))
    # Return sorted lists (not sets) for carriers/bauds so the result is JSON-serializable + stable.
    for name in out:
        out[name]["carriers"] = sorted(out[name]["carriers"])
        out[name]["bauds"] = sorted(out[name]["bauds"])
    if raw_bits_dir is not None and raw_candidates:
        written = _persist_raw_candidates(raw_candidates, raw_bits_dir, window_s=window_s)
        if raw_outputs is not None:
            raw_outputs.extend(written)
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
    *, run_ax25: bool = False, run_endurosat: bool = False, sweep_window_s: float = 0.0,
    carrier_hz: float | None = None, want_waterfall: bool = False, channel_bw_hz: float = 0.0,
    interferer_hz: float | None = None, no_interferer_exclude: bool = False,
    auto_exclude_interferer: bool = False,
    decode_window_s: float = 1.0, max_burst_list: int = 40, sweep_baud: bool = True,
    carrier_count: int = DEFAULT_CARRIER_COUNT,
    mod_index: float = DEFAULT_MOD_INDEX, bt: float = DEFAULT_BT, target_sps: int = 16,
    correct_cfo: bool = True, recover_timing: bool = False,
    raw_bits_dir: str | Path | None = None,
    raw_min_alt_run: int = 0,
    refine_raw: bool = True,
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
    # explicit pick. Auto-exclusion is OPT-IN: the previous default silently masked a weaker packet
    # track near the strongest line (cmd_107). Multi-carrier recovery now handles the normal case.
    interferer: float | None = None
    if not no_interferer_exclude:
        if interferer_hz is not None:
            interferer = float(interferer_hz)
        elif auto_exclude_interferer and sp is not None and sp["snr_db"] >= CARRIER_SNR_DB:
            interferer = float(round(sp["peak_hz"]))
    if interferer is not None:
        print(f"  -> treating {interferer:+.0f} Hz as an off-channel interferer (continuous carrier"
              " / co-visible sat); channel-filtering it out, decoding the bursty data")
    selected = [f for f, on in (("ax25", run_ax25), ("endurosat", run_endurosat)) if on]
    if not selected:  # no framing flag → decode BOTH light framings (what a labelled pass carries)
        selected = ["ax25", "endurosat"]
    # Optional span cap: --sweep-window-s > 0 restricts the (potentially minutes-long) whole-pass
    # decode to a centred window around mid-pass, trading coverage for speed; 0 (default) = whole
    # pass, since real bursts occur throughout the pass, not just at TCA.
    decode_iq, decode_off = cap.iq, 0
    if sweep_window_s and sweep_window_s > 0 and len(cap.iq) > int(cap.fs * sweep_window_s):
        half = max(1, int(cap.fs * sweep_window_s / 2))  # never a degenerate/empty window
        decode_off = max(0, len(cap.iq) // 2 - half)
        decode_iq = cap.iq[decode_off : decode_off + 2 * half]
        print(f"  decode span capped to {sweep_window_s:g}s around mid-pass "
              f"(t={decode_off/cap.fs:.0f}..{(decode_off + len(decode_iq))/cap.fs:.0f}s of "
              f"{dur:.0f}s; --sweep-window-s 0 = whole pass)")
    # BAUD DETECTION: the labelled/declared baud can be WRONG (a pass labelled 9600 actually carried
    # a 2400-baud bird). Find the strongest off-interferer burst and report the 0xAA-preamble run
    # per candidate baud — a run >> the ~10-bit noise level flags the true rate even when the
    # framing that follows is encrypted/whitened and won't validate. Label-independent ground truth.
    sweep_bauds: tuple[float, ...] | None = (
        _usable_sweep_bauds(cap.fs, symbol_rate) if sweep_baud else None
    )
    if sweep_baud:
        strong = _strongest_burst_window(decode_iq, cap.fs, interferer)
        if strong is not None:
            wseg, wcar = strong
            ranked = sorted(
                detect_baud(wseg, cap.fs, carrier_hz=wcar, candidates=sweep_bauds),
                key=lambda r: -r[1],
            )
            print(f"baud detect (0xAA-preamble run/baud @ strongest burst carrier {wcar:+.0f} Hz; "
                  "run>>10 = real):")
            for b, r in ranked:
                flag = "  <- likely TRUE baud" if r >= 32 and r == ranked[0][1] else ""
                print(f"  {int(b):6d} Bd: preamble-run {r:3d} bits{flag}")
            top_baud, top_run = ranked[0]
            if top_run >= 32 and abs(top_baud - symbol_rate) > 1:
                print(f"  NOTE: detected baud {int(top_baud)} != labelled {int(symbol_rate)} - "
                      "the exhaustive sweep will still decode every usable rate.")
    # PRIMARY decode: whole-pass, short-window, channel-filtered, interferer-excluding. Recovers the
    # bursty downlink frames across the ENTIRE pass (the single-window sweep only saw one carrier /
    # one slice). --carrier-hz forces the data carrier; else it is estimated per window. Sweeps the
    # complete usable baud set (CRC-gated, so a wrong baud yields nothing).
    forced = None if carrier_hz is None else [float(carrier_hz)]
    bshow = ("/".join(str(int(b)) for b in sweep_bauds) if sweep_bauds else int(symbol_rate))
    n_baud = len(sweep_bauds) if sweep_bauds else 1
    n_win = max(1, int((len(decode_iq) / cap.fs) / max(decode_window_s * 0.5, 1e-9)))
    channel_text = (f"{channel_bw_hz/1e3:.1f} kHz" if channel_bw_hz > 0 else "2*baud")
    n_carriers = len(forced) if forced is not None else 1 + max(0, int(carrier_count))
    print(f"whole-pass decode ({bshow} Bd, channel={channel_text}, {decode_window_s:g}s windows, "
          f"CRC-gated; framings={','.join(selected)}; up to "
          f"~{n_win * n_baud * n_carriers} demods):")
    raw_outputs: list[Path] = []
    res = decode_pass(
        decode_iq, cap.fs, symbol_rate, tuple(selected), exclude_hz=interferer,
        channel_bw_hz=channel_bw_hz, window_s=decode_window_s, carriers=forced, bauds=sweep_bauds,
        carrier_count=carrier_count, mod_index=mod_index, bt=bt, target_sps=target_sps,
        correct_cfo=correct_cfo, recover_timing=recover_timing,
        raw_bits_dir=raw_bits_dir, capture_offset_s=decode_off / cap.fs,
        raw_min_alt_run=raw_min_alt_run, refine_raw=refine_raw, raw_outputs=raw_outputs,
    )
    for name in selected:
        frames = res[name]["frames"]
        cshow = f" carriers~{[f'{c:+d}' for c in res[name]['carriers'][:6]]}" if frames else ""
        bshow2 = f" bauds={res[name]['bauds']}" if frames else ""
        print(f"  {name}: {len(frames)} frame(s){cshow}{bshow2}")
        for k, frame in enumerate(frames):
            print(f"    frame[{k}]={frame.hex()}")
    if raw_bits_dir is not None:
        print(f"  rawbits: {len(raw_outputs) // 2} lock candidate(s) written to "
              f"{Path(raw_bits_dir).resolve()}")
    # Raw burst view (diagnostic): interferer-aware detection + per-burst carrier + channel filter.
    bursts = find_bursts(decode_iq, cap.fs, exclude_hz=interferer)
    guard = int(cap.fs * 0.003)
    ch = channel_bw_hz if channel_bw_hz > 0 else 2.0 * symbol_rate
    shown = min(len(bursts), max_burst_list)
    print(f"{len(bursts)} bursts (showing {shown}; per-burst carrier, channel-filtered):")
    for k, (s, e) in enumerate(bursts[:max_burst_list]):
        seg = decode_iq[max(0, s - guard) : e + guard]
        # Per-burst DATA carrier: the burst's own strongest line (excluding the interferer), so each
        # burst is de-rotated to its Doppler-current frequency — not a single whole-pass carrier.
        bc = carrier_hz if carrier_hz is not None else _peak_excluding(seg, cap.fs, interferer)
        bits = demodulate_burst(seg, cap.fs, symbol_rate=symbol_rate,
                                carrier_hz=float(bc or 0.0), channel_bw_hz=ch)
        idx = find_sync(bits)
        fb = frame_bytes(bits[idx:]) if idx is not None else b""
        synced = f"sync@bit{idx}" if idx is not None else "NO-SYNC"
        print(  # burst t is absolute in the pass (decode_off accounts for any --sweep-window-s cap)
            f"  burst {k}: t={(decode_off + s)/cap.fs:7.3f}s dur={(e-s)/cap.fs*1000:6.1f}ms "
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
    p.add_argument("--sweep-window-s", type=float, default=0.0,
                   help="0=decode whole pass (default); >0 caps decode to N s around mid-pass")
    p.add_argument("--carrier-hz", type=float, default=None,
                   help="force this exact data-carrier offset (Hz) instead of per-window estimate")
    p.add_argument("--channel-bw", type=float, default=0.0,
                   help="channel-select bandwidth Hz to reject an off-channel carrier (0=2*baud)")
    p.add_argument("--interferer-hz", type=float, default=None,
                   help="explicit continuous-carrier offset to reject (Hz)")
    p.add_argument("--auto-exclude-interferer", action="store_true",
                   help="reject the whole-pass strongest line (off by default; can hide weak data)")
    p.add_argument("--no-exclude-interferer", action="store_true",
                   help="disable interferer rejection, including an explicit/automatic choice")
    p.add_argument("--carrier-count", type=int, default=DEFAULT_CARRIER_COUNT,
                   help="non-DC narrow/broad carrier candidates per window (default: 3)")
    p.add_argument("--decode-window-s", type=float, default=1.0,
                   help="whole-pass decode window (s); short -> Doppler ~const per window")
    p.add_argument("--no-sweep-baud", action="store_true",
                   help="trust --symbol-rate; otherwise decode every usable rate 1200..19200")
    p.add_argument("--mod-index", type=float, default=DEFAULT_MOD_INDEX,
                   help="GFSK modulation index (default: 0.5)")
    p.add_argument("--bt", type=float, default=DEFAULT_BT,
                   help="GFSK Gaussian BT product (default: 0.5)")
    p.add_argument("--target-sps", type=int, default=16,
                   help="demodulator resample target, samples/symbol (default: 16)")
    p.add_argument("--no-cfo", action="store_true",
                   help="disable residual carrier-frequency correction")
    p.add_argument("--recover-timing", action="store_true",
                   help="enable symbol timing recovery")
    p.add_argument("--raw-bits-dir", default=None,
                   help="write full hard-decision streams with lock evidence into this directory")
    p.add_argument("--raw-min-alt-run", type=int, default=0,
                   help="override alternating-preamble lock threshold (0=statistical default)")
    p.add_argument("--no-refine-raw", action="store_true",
                   help="skip bounded post-lock carrier/filter/SPS and repeat-correlation "
                        "refinement")
    p.add_argument("--waterfall", action="store_true",
                   help="write a colored spectrogram <capture>.analyze.png (needs matplotlib)")
    args = p.parse_args(argv)
    analyze_file(args.capture, symbol_rate=args.symbol_rate, sample_rate_hz=args.sample_rate,
                 run_ax25=args.ax25, run_endurosat=args.endurosat,
                 sweep_window_s=args.sweep_window_s, channel_bw_hz=args.channel_bw,
                 interferer_hz=args.interferer_hz,
                 no_interferer_exclude=args.no_exclude_interferer,
                 auto_exclude_interferer=args.auto_exclude_interferer,
                 decode_window_s=args.decode_window_s, sweep_baud=not args.no_sweep_baud,
                 carrier_count=args.carrier_count, mod_index=args.mod_index, bt=args.bt,
                 target_sps=args.target_sps, correct_cfo=not args.no_cfo,
                 recover_timing=args.recover_timing, raw_bits_dir=args.raw_bits_dir,
                 raw_min_alt_run=args.raw_min_alt_run, refine_raw=not args.no_refine_raw,
                 carrier_hz=args.carrier_hz, want_waterfall=args.waterfall)
    return 0


if __name__ == "__main__":
    sys.exit(main())

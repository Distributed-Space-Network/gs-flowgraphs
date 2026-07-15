"""Pre-demod IQ capture — raw IQ (Keysight SDF), VSA CSV, and a waterfall PNG.

When a pass is spawned with ``--record-iq`` the RX engine taps the SDR complex
stream BEFORE the demodulator and writes the requested artifacts into
``--output-dir``:

  * ``<pass>.sdf`` — raw interleaved **int16 big-endian** I/Q (Keysight N5106A
    waveform-data format: I,Q,I,Q…, no header, full-scale ±32767, −3 dB headroom).
  * ``<pass>.csv`` — Keysight 89600 VSA layout (key/value header → ``Y`` → I,Q float
    pairs), derived from the SDF.
  * ``<pass>.png`` — waterfall / spectrogram, derived from the SDF.

Design: the real-time path writes ONLY the SDF (a cheap vectorized sink). The CSV +
PNG are produced once at stop by :func:`finalize_recording` (numpy only — no GNU
Radio and no matplotlib, so they are unit-testable off the bench; the PNG uses a
tiny zlib encoder). gs-client owns the enable/formats/retention config and reaps
these by mtime.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np

_FULL_SCALE = 32767.0
# Keysight PXB factory preset: scale to 70% (−3 dB) so interpolation can't overshoot.
_HEADROOM = 0.7


def parse_formats(spec: str) -> tuple[str, ...]:
    """``"sdf,csv,png"`` → ``("sdf", "csv", "png")`` (lowercased, de-spaced)."""
    return tuple(tok.strip().lower() for tok in spec.split(",") if tok.strip())


# ---------------------------------------------------------------- SDF (raw IQ)


def iq_to_sdf_bytes(samples: np.ndarray, *, headroom: float = _HEADROOM) -> bytes:
    """Complex baseband (|·| ≲ 1) → Keysight N5106A SDF bytes: interleaved int16
    **big-endian** I,Q, full-scale ±32767 with ``headroom`` (−3 dB default)."""
    iq = np.asarray(samples, dtype=np.complex64)
    out = np.empty(iq.size * 2, dtype=">i2")  # big-endian int16, interleaved I,Q
    scale = _FULL_SCALE * headroom
    out[0::2] = np.clip(np.round(iq.real * scale), -_FULL_SCALE, _FULL_SCALE)
    out[1::2] = np.clip(np.round(iq.imag * scale), -_FULL_SCALE, _FULL_SCALE)
    return out.tobytes()


def read_sdf_iq(path: Path, *, headroom: float = _HEADROOM) -> np.ndarray:
    """Inverse of :func:`iq_to_sdf_bytes`: SDF file → complex64 baseband."""
    raw = np.fromfile(path, dtype=">i2").astype(np.float32)
    if raw.size < 2:
        return np.zeros(0, dtype=np.complex64)
    raw = raw[: raw.size - (raw.size % 2)] / (_FULL_SCALE * headroom)
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ---------------------------------------------------------------- VSA CSV


def write_vsa_csv(
    path: Path,
    iq: np.ndarray,
    *,
    center_hz: float,
    sample_rate_hz: float,
    started_utc: _dt.datetime,
) -> None:
    """Write the Keysight 89600 VSA CSV layout: key/value header → ``Y`` → I,Q
    float pairs (matches the bench's exported ``T*.csv``)."""
    iq = np.asarray(iq, dtype=np.complex64)
    span = sample_rate_hz
    header = [
        ("InputZoom", "TRUE"),
        ("InputCenter", f"{center_hz:.1f}"),
        ("InputRange", "1.0"),
        ("InputRefImped", "50.0"),
        ("XDelta", repr(1.0 / sample_rate_hz if sample_rate_hz else 0.0)),
        ("XDomain", "2"),
        ("XUnit", "Sec"),
        ("YUnit", "V"),
        ("IQ", "TRUE"),
        ("FreqValidMax", f"{center_hz + span / 2.0:.1f}"),
        ("FreqValidMin", f"{center_hz - span / 2.0:.1f}"),
        ("DataOverload", "FALSE"),
        ("TimeUtcString", started_utc.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")),
    ]
    with path.open("w", encoding="ascii", newline="\r\n") as f:
        f.write("AppVersion,gs-flowgraphs\n")
        for key, val in header:
            f.write(f"{key},{val}\n")
        f.write(f"NextItemArray[4|{iq.size}|TRUE]\n")
        f.write("Y\n")
        # Vectorized — a per-sample Python loop is minutes-slow on a full-pass capture.
        # The text layer (newline="\r\n") turns savetxt's '\n' row endings into CRLF.
        np.savetxt(f, np.column_stack((iq.real, iq.imag)), fmt="%.6e", delimiter=",")


_log = logging.getLogger("gs_flowgraphs._recorder")


# ---------------------------------------------------------------- waterfall PNG
#
# EXT-WF-001: docs/waterfall_guide.md is the CONTRACT for this pipeline — complex
# Hann-windowed FFTs (an IQ real-cast would mirror the spectrum), 75 % overlap
# with several FFTs averaged per row in LINEAR power, power normalized by
# sum(window)² so every value is ABSOLUTE dBFS (a full-scale tone peaks at
# 0 dBFS), the DC bins interpolated (no black centre line), and ONE global
# display window for the whole image — never per-row/AGC normalization, which
# stretches noise across the colormap and washes weak signals out entirely.

_WF_NFFT = 2048            # ~23 Hz/bin at 48 kHz
_WF_STEP_DIV = 4           # 75 % overlap -> step = nfft // 4
_WF_N_AVG = 8              # FFTs averaged per waterfall row (linear domain)
_WF_MAX_ROWS = 1024        # bounds image height AND total FFT work
_WF_DC_BINS = 3            # centre bins replaced by neighbour interpolation (bin-relative: leakage
                           # is ±1 bin at ANY sample rate, so 3 is universal)
_WF_LOG_FLOOR = 1e-30      # keeps log10(0) finite (guide §4.5)
# Global display window geometry (guide §6). The guide's literal -95..-62 dBFS
# assumes the reference station's ≈ -85 dBFS/bin noise floor — dBFS-per-bin moves
# with front-end gain, analog bandwidth and sample rate, so a hard-coded window is
# NOT universal. We keep the guide's GEOMETRY (floor sits 10 dB above the dark
# end; 33 dB contrast span) anchored to each capture's MEASURED global floor:
# a capture at the reference floor reproduces -95..-62 exactly, and any other
# gain/rate/bandwidth gets the same SatNOGS look. Still ONE window per image.
_WF_SPAN_DB = 33.0
_WF_FLOOR_TO_VMIN_DB = 10.0
# SatNOGS 13-stop colormap (guide §7): dark purple -> blue -> cyan -> green ->
# yellow. The hottest signal is YELLOW — there is deliberately no red.
_WF_CMAP_STOPS = [
    (0.067, 0.004, 0.118), (0.125, 0.031, 0.271), (0.161, 0.094, 0.392),
    (0.165, 0.180, 0.463), (0.137, 0.278, 0.463), (0.106, 0.376, 0.439),
    (0.114, 0.502, 0.412), (0.180, 0.627, 0.349), (0.322, 0.749, 0.259),
    (0.522, 0.827, 0.153), (0.722, 0.878, 0.094), (0.878, 0.902, 0.094),
    (0.965, 0.902, 0.125),
]


def _spectrogram_dbfs(
    iq: np.ndarray, *, nfft: int = _WF_NFFT, max_rows: int = _WF_MAX_ROWS,
    n_avg: int = _WF_N_AVG,
) -> np.ndarray:
    """Absolute-dBFS spectrogram, shape (time, freq), DC-centred, per the guide.

    Each row is the LINEAR-domain mean of ``n_avg`` complex Hann FFTs whose
    starts are spread evenly across the row's time span, normalized by
    sum(window)² (0 dBFS = full-scale tone). Long captures keep ≤ ``max_rows``
    rows by widening each row's span — every row still averages real data
    (the old implementation hopped over the gap, so a burst between hops
    simply never appeared). The ``_WF_DC_BINS`` centre bins are interpolated
    from their neighbours (an SDR DC spike would otherwise paint a line)."""
    if iq.size < nfft:
        # Audit round 2: this used to return a zeros row, which write_waterfall_png
        # then normalized into a perfectly valid-looking uniform PNG — a fabricated
        # spectrogram for a capture that is too short to have one. An empty array
        # makes the caller skip the artifact instead of inventing it.
        return np.zeros((0, nfft), dtype=np.float32)
    step = max(1, nfft // _WF_STEP_DIV)
    total_ffts = 1 + (iq.size - nfft) // step
    rows_n = int(min(max_rows, max(1, total_ffts // n_avg)))
    starts = (
        np.linspace(0, iq.size - nfft, rows_n * n_avg).astype(np.int64)
        .reshape(rows_n, n_avg)
    )
    win = np.hanning(nfft).astype(np.float32)
    norm = float(np.sum(win)) ** 2
    out = np.empty((rows_n, nfft), dtype=np.float32)
    for r in range(rows_n):
        # np.asarray keeps memmap reads chunked to one row's segments at a time.
        segs = np.stack([np.asarray(iq[s : s + nfft]) for s in starts[r]])
        segs = segs.astype(np.complex64, copy=False) * win  # COMPLEX windowing — never a real cast
        power = np.abs(np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)) ** 2 / norm
        out[r] = 10.0 * np.log10(power.mean(axis=0) + _WF_LOG_FLOOR)
    _interpolate_dc_bins(out)
    return out


def _interpolate_dc_bins(spec_db: np.ndarray, *, dc_bins: int = _WF_DC_BINS) -> None:
    """Replace the ``dc_bins`` centre bins with a linear ramp between their
    outer neighbours, in place (guide §5 — never NaN: NaN renders black)."""
    if spec_db.shape[0] == 0 or spec_db.shape[1] < dc_bins + 2:
        return
    center = spec_db.shape[1] // 2
    half = dc_bins // 2
    left = spec_db[:, center - half - 1]
    right = spec_db[:, center + half + 1]
    for j, d in enumerate(range(-half, half + 1)):
        t = (j + 1) / (dc_bins + 1)
        spec_db[:, center + d] = left * (1.0 - t) + right * t


def _display_window_dbfs(spec_db: np.ndarray) -> tuple[float, float]:
    """The ONE global display window for the whole image (guide §6) — never per-row.

    Anchored to the capture's measured GLOBAL noise floor: the median dBFS over
    every bin of every row (robust — real signals occupy few bins; the anti-alias
    band edges pull it down only slightly), placed ``_WF_FLOOR_TO_VMIN_DB`` above
    the dark end with the guide's ``_WF_SPAN_DB`` contrast span. This holds at any
    sample rate, bandwidth or front-end gain; a floor at the guide's reference
    level (≈ -85 dBFS/bin) reproduces its literal -95..-62 dBFS window exactly.
    Known limit: interference occupying >50 % of all bins raises the floor
    estimate and darkens weak signals — still global, never per-row AGC."""
    floor = float(np.median(spec_db))
    vmin = floor - _WF_FLOOR_TO_VMIN_DB
    return vmin, vmin + _WF_SPAN_DB


def _encode_png_gray(img: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG (no external deps). ``img``: 2D uint8 (H, W)."""
    height, width = img.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + body + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit, grayscale
    raw = bytearray()
    for row in img:
        raw.append(0)  # filter type 0 (None) per scanline
        raw.extend(row.tobytes())
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )


def _write_waterfall_matplotlib(
    path: Path, spec: np.ndarray, *, sample_rate_hz: float, duration_s: float,
    center_hz: float, title: str | None, vmin: float, vmax: float,
) -> None:
    """SatNOGS-style colored waterfall per the guide: the 13-stop SatNOGS colormap
    (max = yellow, no red), dark theme, a Power(dBFS) colorbar, Frequency(kHz) X +
    Time(s) Y, one FIXED global scale, and a processing-parameters footer.
    Raises when matplotlib is unavailable so :func:`write_waterfall_png` falls back
    to grayscale."""
    import matplotlib  # noqa: PLC0415 — optional; falls back to grayscale when absent

    matplotlib.use("Agg")  # headless (no display on the bench)
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    cmap = LinearSegmentedColormap.from_list("satnogs", _WF_CMAP_STOPS, N=256)
    half = (sample_rate_hz / 2.0) / 1e3 if sample_rate_hz > 0 else 0.5  # kHz, DC-centered
    dur = duration_s if duration_s > 0 else float(spec.shape[0])
    fig = plt.figure(figsize=(9, 13), dpi=100, facecolor="black")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        # flipud + default (upper) origin with extent 0..dur renders time
        # increasing upward, matching the guide's reference renders.
        np.flipud(spec), aspect="auto", cmap=cmap,
        extent=(-half, half, 0.0, dur), vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_facecolor("black")
    ax.set_xlabel("Frequency (kHz)", fontsize=10, fontweight="bold", color="#cccccc")
    ax.set_ylabel("Time (seconds)", fontsize=10, fontweight="bold", color="#cccccc")
    ax.set_title(
        title or (f"Waterfall - {center_hz/1e6:.4f} MHz" if center_hz else "waterfall"),
        fontsize=12, fontweight="bold", color="#dddddd",
    )
    ax.grid(True, alpha=0.12, color="#888888", linewidth=0.3)
    ax.tick_params(colors="#888888", labelsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Power (dBFS)", color="#cccccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#888888")
    info = (
        f"FS: {sample_rate_hz/1e3:.0f} kHz | FFT: {spec.shape[1]} | 75% Overlap | "
        f"Avg: {_WF_N_AVG}\nHann Window | Floor-anchored Global Scale: {vmin:.0f} to "
        f"{vmax:.0f} dBFS | Duration: {dur:.0f}s | Complex FFT | No per-row AGC"
    )
    ax.text(0.5, -0.045, info, transform=ax.transAxes, fontsize=6,
            ha="center", va="top", color="#666666", fontfamily="monospace")
    fig.tight_layout()
    fig.savefig(str(path), dpi=100, facecolor="black", edgecolor="none")
    plt.close(fig)


def write_waterfall_png(
    path: Path, iq: np.ndarray, *, nfft: int = _WF_NFFT,
    sample_rate_hz: float = 0.0, center_hz: float = 0.0, title: str | None = None,
) -> bool:
    """Write a whole-pass waterfall (time on Y, frequency on X) from the IQ, per
    docs/waterfall_guide.md: absolute dBFS with one GLOBAL display window (never
    per-row AGC), linear-domain FFT averaging, DC bins interpolated, SatNOGS
    colormap via matplotlib when available; otherwise a dependency-free 8-bit
    grayscale PNG mapped through the SAME global window (so the recorder never
    fails a pass just because matplotlib is missing). ``sample_rate_hz`` scales
    the frequency axis to kHz; without it the frequency axis is unlabeled.

    Returns True when a PNG was written, False when the write was SKIPPED (capture
    shorter than one FFT window). CA-FLOW-007: callers must use this explicit
    outcome — probing ``path.exists()`` mistakes a stale PNG left in a reused pass
    workspace for the product of THIS run."""
    spec = _spectrogram_dbfs(np.asarray(iq), nfft=nfft)
    if spec.shape[0] == 0:
        # Audit round 2: a capture shorter than one FFT window has no spectrogram. We
        # used to fabricate one (a zeros row -> a uniform, perfectly plausible PNG).
        # Skip the artifact and say why — an absent waterfall is honest; an invented
        # one is a lie an operator will read as "the band was quiet".
        _log.warning(
            "capture is %d samples — shorter than one %d-point FFT window; NO waterfall "
            "written (a synthetic all-zeros image would misrepresent the band as quiet)",
            int(np.asarray(iq).size), nfft,
        )
        return False
    duration_s = (float(len(iq)) / sample_rate_hz) if sample_rate_hz > 0 else 0.0
    vmin, vmax = _display_window_dbfs(spec)
    try:
        _write_waterfall_matplotlib(
            path, spec, sample_rate_hz=sample_rate_hz, duration_s=duration_s,
            center_hz=center_hz, title=title, vmin=vmin, vmax=vmax)
        return True
    except Exception as e:  # noqa: BLE001 — matplotlib absent/broken → grayscale, never fail the pass
        logging.getLogger("gs_flowgraphs._recorder").info(
            "waterfall: matplotlib unavailable (%s); writing grayscale PNG", e)
    # Same GLOBAL window as the colour path — grayscale must not reintroduce an
    # auto-stretch that renders every capture's noise floor differently.
    img = np.clip((spec - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    path.write_bytes(_encode_png_gray(np.flipud(img)))
    return True


# ---------------------------------------------------------------- finalize


def finalize_recording(
    sdf_path: Path,
    *,
    center_hz: float,
    sample_rate_hz: float,
    formats: tuple[str, ...],
    started_utc: _dt.datetime,
) -> None:
    """After the pass: derive the CSV + PNG (per ``formats``) from the captured SDF.
    The SDF itself was written in real time; this only adds the view formats."""
    need_csv, need_png = "csv" in formats, "png" in formats
    if not (need_csv or need_png) or not sdf_path.exists():
        return
    iq = read_sdf_iq(sdf_path)
    if need_csv:
        write_vsa_csv(
            sdf_path.with_suffix(".csv"),
            iq,
            center_hz=center_hz,
            sample_rate_hz=sample_rate_hz,
            started_utc=started_utc,
        )
    if need_png:
        write_waterfall_png(
            sdf_path.with_suffix(".png"), iq,
            sample_rate_hz=sample_rate_hz, center_hz=center_hz)


# ------------------------------------------------- real-time hookup (bench-only)


def make_iq_sink(path: Path):  # type: ignore[no-untyped-def]
    """A **native** GNU Radio file sink streaming raw ``complex64`` (cf32) to ``path``.

    Used for wideband IQ capture (the SDR runs at the 2+ Msps capture rate). A C++
    ``blocks.file_sink`` keeps up effortlessly and cannot stall the scheduler the way a
    Python ``sync_block`` does at that rate (which produced 0-byte captures + a hung
    ``tb.wait()`` → SIGTERM). Unbuffered, so the IQ hits disk continuously and survives
    a hard stop. cf32 is directly replayable through the dsp engine / ``iq_analyze``."""
    from gnuradio import blocks, gr  # noqa: PLC0415 — bench-only

    sink = blocks.file_sink(gr.sizeof_gr_complex, str(path), False)
    sink.set_unbuffered(True)
    return sink


def make_sdf_sink(path: Path, *, headroom: float = _HEADROOM):  # type: ignore[no-untyped-def]
    """A GNU Radio sink that streams the live complex input to ``path`` as Keysight
    SDF (int16 BE I,Q). Bench-only — imports ``gnuradio`` lazily so this module's
    pure functions stay importable (and unit-testable) without GNU Radio. The byte
    conversion is the tested :func:`iq_to_sdf_bytes`."""
    from gnuradio import gr  # noqa: PLC0415 — bench-only import, keeps pure fns GR-free

    class _SdfSink(gr.sync_block):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            gr.sync_block.__init__(self, name="sdf_sink", in_sig=[np.complex64], out_sig=None)
            self._fh = path.open("wb")

        def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
            self._fh.write(iq_to_sdf_bytes(input_items[0], headroom=headroom))
            return len(input_items[0])

        def stop(self) -> bool:
            self._fh.close()
            return True

    return _SdfSink()


def write_cf32_sidecar(iq_path: Path, *, sample_rate_hz: float, center_hz: float) -> None:
    """Write the self-describing ``<file>.cf32.json`` metadata sidecar. A raw cf32 is
    uninterpretable without its rate/centre; post-pass iq_views reads this (the rate may
    differ from the orchestrator's --sample-rate when the channel was widened)."""
    meta = {
        "sample_rate_hz": float(sample_rate_hz),
        "center_hz": float(center_hz),
        "format": "cf32le",
    }
    try:
        iq_path.with_name(iq_path.name + ".json").write_text(json.dumps(meta))
    except OSError as e:
        # Audit round 2: this was suppressed SILENTLY. The sidecar is the only record
        # of the capture's TRUE rate/centre (they differ from the orchestrator's
        # --sample-rate whenever the channel was widened), so without it every derived
        # artifact — waterfall axes, post-pass decode, CFO search — is mislabelled and
        # nothing says so.
        _log.warning(
            "IQ sidecar write FAILED for %s (%s) — the capture's true rate/centre is "
            "now UNRECORDED; derived artifacts and post-pass decode may be mislabelled",
            iq_path.name, e,
        )


class PassRecorder:
    """Per-pass channel IQ capture for a GNU Radio engine. ``maybe_start`` taps the
    (decimated) channel stream with a **native** cf32 file sink — small, fast, and
    crash-safe (unbuffered, so the IQ is on disk even if the gr-soapy source hangs at
    stop). That ``.cf32`` is the artifact — directly replayable through the dsp engine /
    ``iq_analyze`` — and always kept (the SatNOGS "record every pass" model).

    The view formats (waterfall PNG, VSA CSV) are NOT derived here: gs-client runs
    ``iq_views`` on the ``.cf32`` AFTER the flowgraph has exited, so the derivation has
    free CPU and no stop-deadline to race (the in-stop-path approach kept losing to the
    SoC contention + the gr-soapy teardown hang)."""

    def __init__(self, iq_path: Path) -> None:
        self.iq_path = iq_path

    @classmethod
    def maybe_start(cls, args, tb, src, *, sample_rate_hz: float):  # type: ignore[no-untyped-def]
        """Return a recorder wired into ``tb`` (src → native cf32 file sink), or None
        when capture is off. ``sample_rate_hz`` is the ACTUAL rate of ``src`` (the channel
        rate, which the engine may have widened for a high-baud bird)."""
        if not getattr(args, "record_iq", False):
            return None
        if not parse_formats(getattr(args, "record_formats", "")):
            return None
        out = Path(args.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        iq_path = out / f"{out.name or 'capture'}.cf32"
        tb.connect(src, make_iq_sink(iq_path))
        write_cf32_sidecar(
            iq_path,
            sample_rate_hz=sample_rate_hz,
            center_hz=float(getattr(args, "center_freq_hz", 0.0)),
        )
        return cls(iq_path)


class StreamRecorder:
    """Engine-agnostic **cf32** IQ capture for the numpy (dsp) path: append complex64
    chunks (RAW, pre-demod) as they stream off the SDR. Same raw format as the GR engines
    (:class:`PassRecorder`); the view artifacts (SDF / CSV / PNG) are derived AFTER the
    pass by iq_views (run by gs-client), so nothing is generated in-pass but the cf32."""

    def __init__(self, iq_path: Path, center_hz: float, sample_rate_hz: float) -> None:
        self.iq_path = iq_path
        self._fh = iq_path.open("wb")
        write_cf32_sidecar(iq_path, sample_rate_hz=sample_rate_hz, center_hz=center_hz)

    @classmethod
    def maybe_start(cls, args, *, sample_rate_hz: float):  # type: ignore[no-untyped-def]
        """A recorder writing ``<output_dir>/<dir-name>.cf32``, or None when capture
        is off."""
        if not getattr(args, "record_iq", False):
            return None
        if not parse_formats(getattr(args, "record_formats", "")):
            return None
        out = Path(args.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        iq_path = out / f"{out.name or 'capture'}.cf32"
        return cls(iq_path, float(args.center_freq_hz), float(sample_rate_hz))

    def write(self, chunk: np.ndarray) -> None:
        self._fh.write(np.asarray(chunk, dtype=np.complex64).tobytes())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def first_sample_probe(recorder, *, source=None, counter=None) -> object:  # type: ignore[no-untyped-def]
    """R-11: a first-sample proof — evidence the SDR actually DELIVERED a sample.

    R2-15 (audit): this used to derive the proof ONLY from the recorder's on-disk cf32, so
    it returned None whenever IQ recording was disabled — and the RX apps then skipped the
    check entirely and declared `ready` with no evidence at all. A deaf radio was
    undetectable on exactly the passes that keep no recording. The proof must not depend on
    an unrelated feature flag.

    Three sources, in order of preference:
      * the recorder's unbuffered cf32 (bytes hit disk as soon as the SDR delivers);
      * ``source`` — any GNU Radio block, via ``nitems_written(0)`` (the SDR source itself);
      * ``counter`` — a ``() -> int`` the dsp engines increment as chunks arrive.
    Returns a ``() -> int`` callable, or ``None`` only when NOTHING can prove it.
    """
    iq_path = getattr(recorder, "iq_path", None)
    if iq_path is None:
        if source is not None and hasattr(source, "nitems_written"):
            def _gr_probe() -> int:
                try:
                    return int(source.nitems_written(0))
                except Exception:  # noqa: BLE001 — a probe must never kill the pass
                    return 0
            return _gr_probe
        if callable(counter):
            return counter
        return None

    def _probe() -> int:
        try:
            return iq_path.stat().st_size
        except OSError:
            return 0

    return _probe


__all__ = [
    "PassRecorder",
    "StreamRecorder",
    "finalize_recording",
    "first_sample_probe",
    "iq_to_sdf_bytes",
    "make_sdf_sink",
    "parse_formats",
    "read_sdf_iq",
    "write_vsa_csv",
    "write_waterfall_png",
]

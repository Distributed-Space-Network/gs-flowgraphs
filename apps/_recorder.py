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


# ---------------------------------------------------------------- waterfall PNG


def _spectrogram_db(iq: np.ndarray, *, nfft: int = 1024, max_rows: int = 1024) -> np.ndarray:
    """STFT magnitude in dB, shape (time, freq), DC-centered. Hops are sized so the
    whole capture fits in ≤ ``max_rows`` time slices (so the image stays bounded)."""
    if iq.size < nfft:
        return np.zeros((1, nfft), dtype=np.float32)
    slices = max(1, (iq.size - nfft) // nfft + 1)
    hop = max(nfft, ((iq.size - nfft) // max_rows) + 1) if slices > max_rows else nfft
    win = np.hanning(nfft).astype(np.float32)
    starts = range(0, iq.size - nfft + 1, hop)
    rows = [
        20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq[s : s + nfft] * win))) + 1e-12)
        for s in starts
    ]
    return np.asarray(rows, dtype=np.float32)


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


def write_waterfall_png(path: Path, iq: np.ndarray, *, nfft: int = 1024) -> None:
    """Write a grayscale waterfall (time on Y, frequency on X) from the IQ."""
    spec = _spectrogram_db(np.asarray(iq, dtype=np.complex64), nfft=nfft)
    lo, hi = np.percentile(spec, 5.0), np.percentile(spec, 99.5)
    rng = hi - lo if hi > lo else 1.0
    img = np.clip((spec - lo) / rng * 255.0, 0, 255).astype(np.uint8)
    path.write_bytes(_encode_png_gray(img))


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
        write_waterfall_png(sdf_path.with_suffix(".png"), iq)


# ------------------------------------------------- real-time hookup (bench-only)


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


class PassRecorder:
    """Per-pass capture glue for an RX engine. ``maybe_start`` attaches the SDF sink
    to the SDR source (when ``--record-iq``); ``finalize`` derives the CSV/PNG from
    the SDF and drops the SDF if it wasn't requested."""

    def __init__(
        self, sdf_path: Path, center_hz: float, sample_rate_hz: float, formats: tuple[str, ...]
    ) -> None:
        self._sdf_path = sdf_path
        self._center_hz = center_hz
        self._sample_rate_hz = sample_rate_hz
        self._formats = formats
        self._started = _dt.datetime.now(_dt.UTC)

    @classmethod
    def maybe_start(cls, args, tb, src, *, sample_rate_hz: float):  # type: ignore[no-untyped-def]
        """Return a recorder wired into ``tb`` (src → SDF sink), or None when capture
        is off. The SDF is always written (CSV/PNG derive from it); an unrequested
        SDF is removed in ``finalize``."""
        if not getattr(args, "record_iq", False):
            return None
        formats = parse_formats(getattr(args, "record_formats", ""))
        if not formats:
            return None
        out = Path(args.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        sdf_path = out / f"{out.name or 'capture'}.sdf"
        tb.connect(src, make_sdf_sink(sdf_path))
        return cls(sdf_path, float(args.center_freq_hz), float(sample_rate_hz), formats)

    def finalize(self) -> None:
        finalize_recording(
            self._sdf_path,
            center_hz=self._center_hz,
            sample_rate_hz=self._sample_rate_hz,
            formats=self._formats,
            started_utc=self._started,
        )
        if "sdf" not in self._formats:
            self._sdf_path.unlink(missing_ok=True)


class StreamRecorder:
    """Engine-agnostic IQ capture for the **numpy (dsp) path**: append complex64
    chunks to the SDF as they stream off the SDR, then derive CSV/PNG at finalize.
    Pure numpy — no GNU Radio (the GR engines use :class:`PassRecorder` instead).
    ``write`` records the RAW pre-demod chunk (before any Doppler NCO / demod)."""

    def __init__(
        self, sdf_path: Path, center_hz: float, sample_rate_hz: float, formats: tuple[str, ...]
    ) -> None:
        self._sdf_path = sdf_path
        self._center_hz = center_hz
        self._sample_rate_hz = sample_rate_hz
        self._formats = formats
        self._started = _dt.datetime.now(_dt.UTC)
        self._fh = sdf_path.open("wb")

    @classmethod
    def maybe_start(cls, args, *, sample_rate_hz: float):  # type: ignore[no-untyped-def]
        """A recorder writing ``<output_dir>/<dir-name>.sdf``, or None when capture
        is off."""
        if not getattr(args, "record_iq", False):
            return None
        formats = parse_formats(getattr(args, "record_formats", ""))
        if not formats:
            return None
        out = Path(args.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        sdf_path = out / f"{out.name or 'capture'}.sdf"
        return cls(sdf_path, float(args.center_freq_hz), float(sample_rate_hz), formats)

    def write(self, chunk: np.ndarray) -> None:
        self._fh.write(iq_to_sdf_bytes(chunk))

    def finalize(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        finalize_recording(
            self._sdf_path,
            center_hz=self._center_hz,
            sample_rate_hz=self._sample_rate_hz,
            formats=self._formats,
            started_utc=self._started,
        )
        if "sdf" not in self._formats:
            self._sdf_path.unlink(missing_ok=True)


__all__ = [
    "PassRecorder",
    "StreamRecorder",
    "finalize_recording",
    "iq_to_sdf_bytes",
    "make_sdf_sink",
    "parse_formats",
    "read_sdf_iq",
    "write_vsa_csv",
    "write_waterfall_png",
]

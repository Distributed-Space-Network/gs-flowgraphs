"""Post-pass view derivation: PNG + CSV from a recorded .cf32 (pure numpy, no GNU Radio).

This is what gs-client runs AFTER the flowgraph exits, decoupled from the stop path."""

from __future__ import annotations

import logging
from pathlib import Path

import iq_views
import numpy as np
import pytest
from iq_views import derive_views, main


def _capture(path: Path, n: int = 144_000, fs: float = 48_000.0) -> None:
    t = np.arange(n)
    iq = (0.2 * np.exp(2j * np.pi * 5_000 * t / fs)).astype(np.complex64)
    iq.tofile(path)


def _random_iq(n: int, seed: int = 7) -> np.ndarray:
    """A phase-varying complex64 signal so the discriminator is non-trivial at every boundary."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_derive_views_writes_png_and_csv(tmp_path: Path) -> None:
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)
    written = derive_views(
        cf32, center_hz=401_510_000.0, sample_rate_hz=48_000.0, formats=("png", "csv")
    )
    assert cf32.with_suffix(".png").exists()
    assert cf32.with_suffix(".csv").exists()
    assert set(written) == {cf32.with_suffix(".png"), cf32.with_suffix(".csv")}


def test_stale_png_in_reused_workspace_is_not_reported_as_produced(tmp_path: Path) -> None:
    # CA-FLOW-007 (R2-21 recurrence): a retry reuses the pass workspace, which still
    # holds the PNG from an earlier attempt. This run's capture is SHORTER than one
    # FFT window, so the writer skips — the stale PNG must be neither reported as
    # produced nor left on disk to be swept up as this pass's product.
    cf32 = tmp_path / "retry.cf32"
    _capture(cf32, n=100)  # < one 2048-point FFT window: waterfall write is skipped
    stale = cf32.with_suffix(".png")
    stale.write_bytes(b"\x89PNG stale artifact from the previous attempt")
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png",))
    assert written == []
    assert not stale.exists()


def test_fresh_png_still_produced_over_a_stale_one(tmp_path: Path) -> None:
    # The stale file must not suppress a REAL product either: a long-enough capture
    # replaces it and reports the path.
    cf32 = tmp_path / "retry_ok.cf32"
    _capture(cf32)
    stale = cf32.with_suffix(".png")
    stale.write_bytes(b"stale")
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png",))
    assert written == [stale]
    assert stale.stat().st_size > 100  # a real PNG, not the seeded bytes


def test_derive_views_respects_requested_formats(tmp_path: Path) -> None:
    cf32 = tmp_path / "p.cf32"
    _capture(cf32)
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png",))
    assert cf32.with_suffix(".png").exists()
    assert not cf32.with_suffix(".csv").exists()
    assert not cf32.with_suffix(".sdf").exists()


def test_derive_views_sdf_is_whole_pass_int16(tmp_path: Path) -> None:
    # SDF is the whole-pass Keysight int16 transcode of the cf32 (4 B/sample).
    cf32 = tmp_path / "s.cf32"
    n = 144_000
    _capture(cf32, n=n)
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("sdf",))
    sdf = cf32.with_suffix(".sdf")
    assert sdf.exists()
    assert sdf.stat().st_size == n * 4  # int16 I + int16 Q per sample (whole pass)


def test_torn_write_is_truncated_not_fatal(tmp_path: Path) -> None:
    # A SIGTERM mid-write can leave a non-multiple-of-8 file; memmap must still work.
    cf32 = tmp_path / "torn.cf32"
    _capture(cf32, n=60_000)
    with cf32.open("ab") as fh:
        fh.write(b"\x01\x02\x03")  # 3 stray bytes
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png", "csv"))
    assert cf32.with_suffix(".png").exists()
    assert cf32.with_suffix(".csv").exists()


def test_sidecar_rate_overrides_arg(tmp_path: Path) -> None:
    # A high-baud bird widens the channel, so the .cf32 rate differs from the
    # orchestrator's --sample-rate. The .cf32.json sidecar (true rate) must win.
    cf32 = tmp_path / "s.cf32"
    _capture(cf32, n=200_000)
    (tmp_path / "s.cf32.json").write_text('{"sample_rate_hz": 96000.0, "center_hz": 401000000.0}')
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("csv",), csv_seconds=1.0)
    csv = cf32.with_suffix(".csv").read_text()
    assert f"XDelta,{1.0 / 96000.0!r}" in csv  # sidecar 96 kHz used, not the 48 kHz arg


def test_missing_or_empty_input_is_noop(tmp_path: Path) -> None:
    assert derive_views(tmp_path / "nope.cf32", center_hz=0.0, sample_rate_hz=48_000.0,
                        formats=("png", "csv")) == []
    empty = tmp_path / "empty.cf32"
    empty.write_bytes(b"")
    assert derive_views(empty, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png", "csv")) == []


def test_cli_main(tmp_path: Path) -> None:
    cf32 = tmp_path / "cli.cf32"
    _capture(cf32)
    rc = main(["--input", str(cf32), "--sample-rate", "48000", "--center-hz", "401e6",
               "--formats", "png,csv", "--csv-seconds", "1"])
    assert rc == 0
    assert cf32.with_suffix(".png").exists() and cf32.with_suffix(".csv").exists()


def test_main_fails_when_a_requested_view_is_not_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """RE-AUDIT (P2): a requested view that was NOT produced (e.g. ogg when ffmpeg failed) must make
    main() return NON-ZERO and print the ACTUAL produced set — never a silent success (which made
    the supervisor log 'derived ogg' for an OGG that does not exist)."""
    cf32 = tmp_path / "partial.cf32"
    _capture(cf32)

    def _partial(_inp: object, **_kw: object) -> list[Path]:
        sdf = cf32.with_suffix(".sdf")
        sdf.write_bytes(b"x")
        return [sdf]  # produced sdf, but NOT the requested ogg

    monkeypatch.setattr(iq_views, "derive_views", _partial)
    rc = main(["--input", str(cf32), "--sample-rate", "48000", "--formats", "sdf,ogg"])
    assert rc == 1, "a requested-but-unproduced view must fail, not report silent success"
    assert "produced=sdf" in capsys.readouterr().out


# ---------------------------------------------------------------- OGG discriminator audio


def test_discriminator_chunk_boundaries_match_whole_array() -> None:
    """CRITICAL correctness property: chunking the CF32 at ANY (even random) boundary yields the
    SAME discriminator as one whole-array computation — phase must not reset at a boundary."""
    iq = _random_iq(20_003)
    # Reference computed independently of the chunked path under test.
    ref = np.empty(len(iq), dtype=np.float32)
    ref[0] = 0.0
    ref[1:] = np.angle(iq[1:] * np.conj(iq[:-1])) / np.pi
    for chunk in (1, 2, 3, 7, 500, 4096, len(iq) - 1, len(iq), len(iq) + 5):
        got = iq_views._discriminator(iq, chunk_samples=chunk)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=1e-6, err_msg=f"chunk={chunk}")
    # And a spread of random boundary sizes.
    rng = np.random.default_rng(1)
    for _ in range(20):
        got = iq_views._discriminator(iq, chunk_samples=int(rng.integers(1, len(iq) + 2)))
        np.testing.assert_allclose(got, ref, atol=1e-6)


# ------------------- OGG discriminator audio (soundfile/Vorbis — pip-only, no ffmpeg)
# OPERATOR DECISION 2026-07-15 (CA-INTEG-002): the encoder is the pip-installable
# `soundfile` wheel (bundled libsndfile + Vorbis). These tests run REAL encodes on
# every platform — the old ffmpeg-absent skip is gone with the ffmpeg dependency.

import soundfile as sf  # noqa: E402 — bundled-Vorbis wheel; a hard test dependency now


def test_resampler_is_chunk_size_invariant() -> None:
    """The streaming resampler must produce BIT-IDENTICAL output for any chunking of
    the same input — FIR state and decimation phase carry across chunks exactly."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50_011).astype(np.float32)
    whole = iq_views._StreamingResampler(96_000, 48_000)
    ref = np.concatenate([whole.process(x), whole.flush()])
    for chunk in (1_000, 4_096, 7_777, 50_000):
        r = iq_views._StreamingResampler(96_000, 48_000)
        parts = [r.process(x[off : off + chunk]) for off in range(0, len(x), chunk)]
        parts.append(r.flush())
        got = np.concatenate(parts)
        assert got.shape == ref.shape, f"chunk={chunk}"
        np.testing.assert_array_equal(got, ref, err_msg=f"chunk={chunk}")


def test_resampler_preserves_a_tone_and_rejects_exotic_rates() -> None:
    fs_in = 96_000
    t = np.arange(fs_in)
    tone = np.sin(2 * np.pi * 1_000 * t / fs_in).astype(np.float32)
    r = iq_views._StreamingResampler(fs_in, 48_000)
    out = np.concatenate([r.process(tone), r.flush()])
    assert abs(len(out) - 48_000) < 500  # ~1 s at 48 kHz (FIR flush adds a hair)
    spec = np.abs(np.fft.rfft(out[:48_000]))
    assert abs(int(np.argmax(spec[10:])) + 10 - 1_000) <= 2  # 1 kHz bin survives
    with pytest.raises(ValueError, match="unsupported capture rate"):
        iq_views._StreamingResampler(48_001, 48_000)  # upsample factor 48000 — refuse


@pytest.mark.parametrize("channels", [1, 2])
def test_ogg_real_encode_atomic_and_decodable(tmp_path: Path, channels: int) -> None:
    cf32 = tmp_path / "ok.cf32"
    n, fs = 96_000, 96_000.0  # one second
    _capture(cf32, n=n, fs=fs)
    written = derive_views(
        cf32, center_hz=0.0, sample_rate_hz=fs, formats=("ogg",), ogg_channels=channels
    )
    ogg, tmp = cf32.with_suffix(".ogg"), cf32.with_suffix(".ogg.tmp")
    assert ogg in written and ogg.exists() and ogg.stat().st_size > 0
    assert not tmp.exists()                       # temp renamed away, none left behind
    info = sf.info(str(ogg))
    assert info.samplerate == iq_views.OGG_SAMPLE_RATE_HZ
    assert info.channels == channels
    assert abs(info.frames / info.samplerate - 1.0) < 0.05  # ~1 s of audio
    if channels == 2:
        data, _rate = sf.read(str(ogg))
        # stereo is the SAME mono signal duplicated — deterministic, no fake width
        np.testing.assert_allclose(data[:, 0], data[:, 1], atol=1e-6)


def test_ogg_missing_soundfile_skips_with_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "soundfile", None)  # import raises ImportError
    cf32 = tmp_path / "nosf.cf32"
    _capture(cf32, n=20_000)
    with caplog.at_level(logging.WARNING, logger="iq_views"):
        out = iq_views.write_discriminator_ogg(
            cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_000.0
        )
    assert out is None
    assert "soundfile" in caplog.text
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()


def test_invalid_ogg_channels_skips_encode(tmp_path: Path) -> None:
    cf32 = tmp_path / "bad.cf32"
    _capture(cf32, n=20_000)
    out = iq_views.write_discriminator_ogg(
        cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_000.0, ogg_channels=3
    )
    assert out is None
    assert not cf32.with_suffix(".ogg").exists()


def test_unsupported_rate_skips_encode_with_no_artifact(tmp_path: Path) -> None:
    cf32 = tmp_path / "odd.cf32"
    _capture(cf32, n=20_000)
    out = iq_views.write_discriminator_ogg(
        cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_001.0
    )
    assert out is None
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()


def test_stale_tmp_is_never_renamed_into_the_final_ogg(tmp_path: Path) -> None:
    cf32 = tmp_path / "stale.cf32"
    _capture(cf32, n=96_000, fs=96_000.0)
    stale = cf32.with_suffix(".ogg.tmp")
    stale.write_bytes(b"NOT-AN-OGG-FROM-A-CRASHED-RUN")
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=96_000.0, formats=("ogg",))
    ogg = cf32.with_suffix(".ogg")
    assert ogg in written
    info = sf.info(str(ogg))  # the final file is a REAL fresh encode, not the stale bytes
    assert info.samplerate == iq_views.OGG_SAMPLE_RATE_HZ


def test_synthetic_gfsk_ogg_decodes_through_audio_analyze(tmp_path: Path) -> None:
    """End-to-end (REAL Vorbis encode, no external tools): synth an AX.25 GFSK capture,
    derive the OGG, decode the audio via soundfile, and confirm the bench analyzer
    recovers at least one CRC-valid frame from it."""
    import audio_analyze

    from gfsk_ax25 import ax25
    from gfsk_ax25 import framing as ax25_framing

    body = ax25.encode_ui(dest="CQ", src="DSN", info=b"OGG VIEW SELFTEST")
    bits = ax25_framing.encode(body, scramble=True, nrzi=True)
    fs, sps = 48_000.0, 5  # 9600 baud, five samples/symbol
    levels = np.concatenate([
        np.zeros(4_000, dtype=np.float32),
        np.repeat(bits.astype(np.float32) * 2.0 - 1.0, sps),
        np.zeros(4_000, dtype=np.float32),
    ])
    # FM-modulate: instantaneous frequency ∝ symbol level, so the discriminator returns the levels.
    inst_freq = levels * (fs * 0.45)  # phase step 0.9*pi/sample: unambiguous, robust to Vorbis loss
    iq = np.exp(1j * 2.0 * np.pi * np.cumsum(inst_freq) / fs).astype(np.complex64)
    cf32 = tmp_path / "synth.cf32"
    iq.tofile(cf32)

    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=fs, formats=("ogg",))
    ogg = cf32.with_suffix(".ogg")
    assert ogg in written and ogg.exists()

    audio, rate = sf.read(str(ogg), dtype="float32")
    frames = audio_analyze.decode_ax25_audio(audio, float(rate), 9600.0, window_s=1.0)
    assert any(frame == body for _, frame in frames)


# --------------------------------------------------------------------------- Phase 2C (DS-020..023)


def test_stale_ogg_in_reused_workspace_is_removed_before_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DS-020 (CA-FLOW-007 completion): the PNG branch removes its stale target first; the OGG
    branch did not — a reused workspace's old .ogg survived a skipped/failed encode and was swept
    up as THIS pass's audio. Skip is forced via the missing-soundfile path."""
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "soundfile", None)  # encode import raises -> writer skips
    cf32 = tmp_path / "retry.cf32"
    _capture(cf32, n=20_000)
    stale = cf32.with_suffix(".ogg")
    stale.write_bytes(b"OggS stale audio from the previous attempt")
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("ogg",))
    assert not stale.exists(), "the stale .ogg from the previous attempt survived"
    assert cf32.with_suffix(".ogg") not in written


def test_sdf_failure_leaves_no_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DS-021: SDF is written via tmp + atomic rename — an OSError mid-write must leave NO
    truncated final .sdf (and no .tmp remnant)."""
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)

    def _boom(_arr: np.ndarray) -> bytes:
        msg = "No space left on device"
        raise OSError(28, msg)

    monkeypatch.setattr(iq_views, "iq_to_sdf_bytes", _boom)
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("sdf",))
    assert written == []
    assert not cf32.with_suffix(".sdf").exists(), "a truncated final .sdf was left behind"
    assert not cf32.with_suffix(".sdf.tmp").exists(), "the .sdf.tmp remnant was left behind"


def test_csv_failure_leaves_no_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DS-021 (CSV half): same tmp+rename contract for the VSA CSV."""
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)

    def _boom(*_a: object, **_kw: object) -> None:
        msg = "No space left on device"
        raise OSError(28, msg)

    monkeypatch.setattr(iq_views, "write_vsa_csv", _boom)
    written = derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("csv",))
    assert written == []
    assert not cf32.with_suffix(".csv").exists()
    assert not cf32.with_suffix(".csv.tmp").exists()


def test_vsa_timestamp_is_the_capture_start_not_the_last_write(tmp_path: Path) -> None:
    """DS-022: TimeUtcString came from the cf32 mtime — the LAST write (LOS), shifting every VSA
    timestamp by the whole pass duration. It must be the capture START (mtime - duration)."""
    import datetime as dt
    import os

    fs = 48_000.0
    n = 480_000  # 10 s capture
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32, n=n, fs=fs)
    end = dt.datetime(2026, 7, 16, 12, 0, 10, tzinfo=dt.UTC)
    os.utime(cf32, (end.timestamp(), end.timestamp()))

    derive_views(cf32, center_hz=0.0, sample_rate_hz=fs, formats=("csv",), csv_seconds=0.01)
    header = cf32.with_suffix(".csv").read_text(encoding="ascii")
    line = next(ln for ln in header.splitlines() if ln.startswith("TimeUtcString"))
    stamp = line.split(",", 1)[1]
    assert stamp.startswith("2026-07-16T12:00:00"), (
        f"TimeUtcString is {stamp!r} — expected the capture START (end 12:00:10 minus 10 s)"
    )


def test_main_rejects_unknown_format_tokens(tmp_path: Path) -> None:
    """DS-023: an unknown --formats token is a config error (rc!=0), not a silent drop that
    vanishes from the produced-vs-requested accounting."""
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)
    rc = main([
        "--input", str(cf32), "--sample-rate", "48000",
        "--formats", "sdf,waterfal",  # typo'd token
    ])
    assert rc == 2
    assert not cf32.with_suffix(".sdf").exists(), "views were derived despite the config error"

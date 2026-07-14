"""Post-pass view derivation: PNG + CSV from a recorded .cf32 (pure numpy, no GNU Radio).

This is what gs-client runs AFTER the flowgraph exits, decoupled from the stop path."""

from __future__ import annotations

import logging
import shutil
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


class _FakeFfmpeg:
    """A stand-in for ``subprocess.Popen(ffmpeg ...)`` — records argv + fed bytes, and simulates a
    chosen exit code without a real encoder (ffmpeg is absent on the Windows dev box).

    ``output_bytes`` is what a "successful" (``make_output``) encode writes to the temp file; pass
    ``b""`` to model ffmpeg exiting 0 while producing an EMPTY file. ``write_exc`` raises that
    exception from ``stdin.write`` to model a cancellation (e.g. ``KeyboardInterrupt``) mid-stream.
    Each spawned process tracks ``killed`` / ``reaped`` so tests can assert the encoder was both
    signalled and awaited."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        make_output: bool = True,
        raise_on_write: bool = False,
        output_bytes: bytes = b"OggS\x00fake-vorbis",
        write_exc: type[BaseException] | None = None,
    ):
        self.calls: list[list[str]] = []
        self.instances: list = []
        outer = self

        class _FakeStdin:
            def __init__(self) -> None:
                self.chunks: list[bytes] = []
                self.closed = False

            def write(self, data: bytes) -> int:
                if write_exc is not None:
                    raise write_exc("cancelled mid-write")
                if raise_on_write:
                    raise BrokenPipeError("broken pipe")
                self.chunks.append(bytes(data))
                return len(data)

            def close(self) -> None:
                self.closed = True

        class _Popen:
            def __init__(self, argv, stdin=None, stdout=None, stderr=None, **kw) -> None:
                outer.calls.append(list(argv))
                outer.instances.append(self)
                self.argv = list(argv)
                self.stdin = _FakeStdin()
                self.returncode: int | None = None
                self.killed = False
                self.reaped = False
                self._out = Path(argv[-1])

            def communicate(self, timeout=None):
                self.reaped = True
                if make_output:
                    self._out.write_bytes(output_bytes)
                if self.returncode is None:  # a prior kill() already set the signal returncode
                    self.returncode = returncode
                return (b"", b"" if returncode == 0 else b"fake encoder error")

            def kill(self) -> None:
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                self.reaped = True
                return self.returncode

        self.popen = _Popen

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _FakeFfmpeg:
        monkeypatch.setattr(iq_views.subprocess, "Popen", self.popen)
        return self


def _fed_bytes(fake: _FakeFfmpeg) -> bytes:
    return b"".join(fake.instances[0].stdin.chunks)


def _output_ac(argv: list[str]) -> str:
    """The OUTPUT channel count = the value after the LAST ``-ac`` (input ``-ac`` is always 1)."""
    idxs = [i for i, tok in enumerate(argv) if tok == "-ac"]
    return argv[idxs[-1] + 1]


def test_derive_views_writes_png_and_csv(tmp_path: Path) -> None:
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)
    written = derive_views(
        cf32, center_hz=401_510_000.0, sample_rate_hz=48_000.0, formats=("png", "csv")
    )
    assert cf32.with_suffix(".png").exists()
    assert cf32.with_suffix(".csv").exists()
    assert set(written) == {cf32.with_suffix(".png"), cf32.with_suffix(".csv")}


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


def test_ffmpeg_argv_reports_input_rate_resample_and_channels(tmp_path: Path) -> None:
    tmp = tmp_path / "p.ogg.tmp"
    mono = iq_views._ffmpeg_ogg_argv("ffmpeg", 96_000.0, 1, tmp)
    assert mono[mono.index("-i") - 1] == "pipe:0" or "pipe:0" in mono
    assert mono[mono.index("-ar") + 1] == "96000"          # true input rate
    assert "48000" in mono                                  # resample target
    assert mono[mono.index("-c:a") + 1] == "libvorbis"      # Vorbis encode
    assert _output_ac(mono) == "1"
    assert "pan=stereo|c0=c0|c1=c0" not in mono
    # The OUTPUT muxer is pinned explicitly (the ``.ogg.tmp`` name is not inferrable) right before
    # the output path — ffmpeg would otherwise fail or guess a wrong container.
    assert mono[-3:] == ["-f", "ogg", str(tmp)]
    stereo = iq_views._ffmpeg_ogg_argv("ffmpeg", 48_000.0, 2, tmp)
    assert _output_ac(stereo) == "2"                        # explicit two output channels
    assert "pan=stereo|c0=c0|c1=c0" in stereo               # explicit mono->L/R duplication
    assert stereo[-3:] == ["-f", "ogg", str(tmp)]           # explicit output muxer here too


def test_ogg_absent_spawns_no_ffmpeg_and_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeFfmpeg().install(monkeypatch)
    cf32 = tmp_path / "noaudio.cf32"
    _capture(cf32, n=20_000)
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("sdf",), ffmpeg="ffmpeg")
    assert fake.calls == []                                 # no ffmpeg process
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()


def test_invalid_ogg_channels_skips_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # iq_views rejects a bad channel count rather than clamping; no ffmpeg is spawned.
    fake = _FakeFfmpeg().install(monkeypatch)
    cf32 = tmp_path / "bad.cf32"
    _capture(cf32, n=20_000)
    out = iq_views.write_discriminator_ogg(
        cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_000.0,
        ogg_channels=3, ffmpeg="ffmpeg",
    )
    assert out is None
    assert fake.calls == []
    assert not cf32.with_suffix(".ogg").exists()


@pytest.mark.parametrize("channels", [1, 2])
def test_ogg_success_atomic_rename_and_mono_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    fake = _FakeFfmpeg(returncode=0, make_output=True).install(monkeypatch)
    cf32 = tmp_path / "ok.cf32"
    n = 20_000
    _capture(cf32, n=n)
    written = derive_views(
        cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("ogg",),
        ogg_channels=channels, ffmpeg="ffmpeg",
    )
    ogg, tmp = cf32.with_suffix(".ogg"), cf32.with_suffix(".ogg.tmp")
    assert ogg in written and ogg.exists()
    assert not tmp.exists()                                 # temp renamed away, none left behind
    assert _output_ac(fake.calls[0]) == str(channels)
    # A SINGLE mono discriminator is always fed (float32, one value per input sample), regardless of
    # the requested channel count — duplication happens inside ffmpeg, not by feeding I/Q.
    assert len(_fed_bytes(fake)) == n * 4


def test_ogg_encoder_failure_leaves_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A partial temp is written then the encoder exits non-zero: no final AND no temp may survive,
    # and NO success line is logged — the failure must be reported truthfully.
    fake = _FakeFfmpeg(returncode=1, make_output=True).install(monkeypatch)
    cf32 = tmp_path / "fail.cf32"
    _capture(cf32, n=20_000)
    with caplog.at_level(logging.INFO, logger="iq_views"):
        written = derive_views(
            cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("ogg",), ffmpeg="ffmpeg"
        )
    assert fake.calls != []                                 # it did attempt to encode
    assert written == []
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()
    assert "derived OGG" not in caplog.text                 # no false success log


def test_ogg_empty_output_reports_failure_no_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # ffmpeg exits 0 but produces an EMPTY file — that is NOT a produced artifact. It must be
    # reported as a failure (no return, no success log), and the empty temp removed.
    fake = _FakeFfmpeg(returncode=0, make_output=True, output_bytes=b"").install(monkeypatch)
    cf32 = tmp_path / "emptyout.cf32"
    _capture(cf32, n=20_000)
    with caplog.at_level(logging.INFO, logger="iq_views"):
        written = derive_views(
            cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("ogg",), ffmpeg="ffmpeg"
        )
    assert fake.calls != []
    assert written == []
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()
    assert "derived OGG" not in caplog.text                 # empty output is not success


def test_ogg_cancellation_kills_reaps_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cancellation mid-encode (KeyboardInterrupt from the write loop) must: propagate, leave NO
    # .ogg and NO .ogg.tmp, and kill+reap the encoder (never orphan the ffmpeg child).
    fake = _FakeFfmpeg(make_output=False, write_exc=KeyboardInterrupt).install(monkeypatch)
    cf32 = tmp_path / "cancel.cf32"
    _capture(cf32, n=20_000)
    with pytest.raises(KeyboardInterrupt):
        iq_views.write_discriminator_ogg(
            cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_000.0, ffmpeg="ffmpeg"
        )
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()
    inst = fake.instances[0]
    assert inst.killed is True                              # encoder was signalled
    assert inst.reaped is True                              # ...and awaited (bounded reap)


def test_ogg_broken_pipe_leaves_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ffmpeg dies mid-stream (write raises BrokenPipe): still no partial .ogg / .ogg.tmp.
    fake = _FakeFfmpeg(returncode=1, make_output=False, raise_on_write=True).install(monkeypatch)
    cf32 = tmp_path / "bp.cf32"
    _capture(cf32, n=20_000)
    out = iq_views.write_discriminator_ogg(
        cf32, np.fromfile(cf32, dtype=np.complex64), sample_rate_hz=48_000.0, ffmpeg="ffmpeg"
    )
    assert out is None
    assert fake.calls != []
    assert not cf32.with_suffix(".ogg").exists()
    assert not cf32.with_suffix(".ogg.tmp").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed (Windows dev box) — real encode/decode not exercised",
)
def test_synthetic_gfsk_ogg_decodes_through_audio_analyze(tmp_path: Path) -> None:
    """End-to-end (real ffmpeg): synth an AX.25 GFSK capture, derive the OGG, and confirm
    tools/audio_analyze.py recovers at least one CRC-valid frame from it."""
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

    audio, meta = audio_analyze.load_audio(str(ogg))
    frames = audio_analyze.decode_ax25_audio(audio, meta["sample_rate"], 9600.0, window_s=1.0)
    assert any(frame == body for _, frame in frames)

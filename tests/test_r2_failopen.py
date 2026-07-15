"""Audit round 2 — the SILENT-SUCCESS defect class, swept out of gs-flowgraphs.

R2-02 (a recorder-only pass reporting a clean decode) was one instance of a pattern: **the
engine could not do the job it was asked to do, degraded silently, and the pass still
reported success.** An adversarial sweep found the rest of the family in this repo.

GNU Radio is not importable off-bench, so the *logic* lives in pure helpers and the app
coroutines are driven directly with stubs. Source-text assertions are used ONLY to pin that
a deleted code path has not come back — never as evidence that a behaviour works. (An
earlier version of this file did exactly that, and passed implementations that did nothing:
see tests/test_r2_engine_death.py and tests/test_r2_tx_safety.py, which replace them.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

from _fallback_select import no_decode_reason  # noqa: E402


class TestADemodWithNoDeframerIsDecodeDead:
    """The nastiest of the family: the demod builds, the graph LOOKS healthy, and the
    engine even logs a success-shaped 'our demod fsk@9600 …' — but the backend framing
    (AX.100 / USP / Mobitex / CCSDS Concatenated) has no local deframer, gr-satellites is
    gated off, and every drain returns nothing. Zero frames, no error, forever."""

    def test_demod_built_but_no_deframer_is_reported(self) -> None:
        why = no_decode_reason(
            has_decode_consumer=True,          # the demod DID build
            mode=("fsk", 9600.0),
            grsat_live=False,
            framing="AX.100",
            deframer_available=False,          # ...but nothing can deframe it
        )
        assert why, "a demod nothing can deframe must not look like a healthy decode"
        assert "no deframer" in why
        assert "AX.100" in why
        assert "GS_GRSAT_LIVE" in why

    def test_a_deframable_framing_is_not_degraded(self) -> None:
        assert no_decode_reason(
            has_decode_consumer=True,
            mode=("gfsk", 9600.0),
            grsat_live=False,
            framing="ax25",
            deframer_available=True,
        ) == ""

    def test_deframer_available_defaults_true_for_older_callers(self) -> None:
        assert no_decode_reason(
            has_decode_consumer=True, mode=("gfsk", 9600.0), grsat_live=True
        ) == ""


class TestTheEngineWiresTheDeframerCheck:
    def test_gnuradio_satellites_computes_deframer_availability(self) -> None:
        src = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")
        assert "deframer_available = " in src
        assert "deframer_available=deframer_available" in src


class TestAnEngineThatCannotRunACommandSaysSo:
    """The TX apps do ALL of their transmitting inside these handlers. A `transmit` that
    blew up was logged and then IGNORED — dispatch continued, the exit code was unaffected,
    and gs-client saw a pass that completed as if it had radiated.

    (These replace source-text greps. The first fix made the error event OPTIONAL and no
    app passed a writer, so it stayed silent — and my grep-test passed anyway. Behaviour
    is the only thing worth asserting.)"""

    @staticmethod
    def _drive(handlers: dict, *cmds: dict) -> tuple[str, list[dict]]:
        import asyncio
        import json

        from _spawn_contract import run_command_loop

        NL = b"\n"

        class _W:
            def __init__(self) -> None:
                self.buf = b""

            def write(self, data: bytes) -> None:
                self.buf += data

            async def drain(self) -> None:
                return None

        async def _run() -> tuple[str, list[dict]]:
            reader = asyncio.StreamReader()
            for c in cmds:
                reader.feed_data(json.dumps(c).encode() + NL)
            reader.feed_eof()
            w = _W()
            reason = await run_command_loop(reader, handlers, w)  # type: ignore[arg-type]
            events = [
                json.loads(ln) for ln in w.buf.decode().splitlines() if ln.strip()
            ]
            return reason, events

        return asyncio.run(_run())

    def test_a_failed_transmit_emits_an_error_and_ends_dispatch(self) -> None:
        seen: list[str] = []

        async def boom(_cmd: dict) -> None:
            raise RuntimeError("SDR write failed")

        async def later(_cmd: dict) -> None:
            seen.append("later")

        reason, events = self._drive(
            {"start": boom, "stop": later},
            {"cmd": "start"},          # raises
            {"cmd": "stop"},           # must NOT be dispatched — the engine is broken
        )
        assert reason == "handler-failed", "the app must exit nonzero, not report a clean stop"
        assert seen == [], "dispatch continued past a failed handler"
        err = [e for e in events if e.get("event") == "error"]
        assert err and err[0]["code"] == "handler-failed"
        assert err[0]["cmd"] == "start"

    def test_a_raising_stop_handler_still_ends_dispatch_cleanly(self) -> None:
        """P0-08 must survive: a teardown that throws still ends dispatch as a STOP —
        the caller's cleanup path owns the rest."""
        async def boom(_cmd: dict) -> None:
            raise RuntimeError("teardown died")

        reason, events = self._drive({"stop": boom}, {"cmd": "stop"})
        assert reason == "stop"
        assert any(e.get("event") == "error" for e in events), "still reported, just not fatal"

    def test_a_healthy_pass_reports_a_clean_stop(self) -> None:
        async def ok(_cmd: dict) -> None:
            return None

        reason, events = self._drive({"start": ok, "stop": ok}, {"cmd": "start"}, {"cmd": "stop"})
        assert reason == "stop"
        assert not [e for e in events if e.get("event") == "error"]


# The bidir engine's death is covered by tests/test_r2_engine_death.py, which DRIVES
# run_rx with a dying IQ source. The grep-test that used to live here asserted the string
# "reader_error.append(e)" was present — and passed an implementation that never raised.


class TestTheRecorderDoesNotInventArtifacts:
    def test_a_too_short_capture_yields_no_spectrogram_rows(self) -> None:
        """A zeros row became a uniform, perfectly plausible PNG — an operator reads that
        as 'the band was quiet'. An absent waterfall is honest; an invented one is not."""
        import numpy as np
        from _recorder import _spectrogram_dbfs

        spec = _spectrogram_dbfs(np.zeros(64, dtype=np.complex64), nfft=1024)
        assert spec.shape[0] == 0, "a capture shorter than one FFT window has NO waterfall"

    def test_a_real_capture_still_produces_rows(self) -> None:
        import numpy as np
        from _recorder import _spectrogram_dbfs

        iq = (np.random.default_rng(0).standard_normal(4096)
              + 1j * np.random.default_rng(1).standard_normal(4096)).astype(np.complex64)
        assert _spectrogram_dbfs(iq, nfft=1024).shape[0] > 0

    def test_the_sidecar_write_failure_is_logged_not_suppressed(self) -> None:
        src = (_APPS / "_recorder.py").read_text(encoding="utf-8")
        assert "IQ sidecar write FAILED" in src
        assert "with contextlib.suppress(OSError):\n        iq_path" not in src


class TestDroppedAudioIsVisible:
    def test_fm_rx_logs_its_drops(self) -> None:
        src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
        assert "audio queue FULL" in src
        assert "_DROP_LOG_EVERY" in src, "rate-limited, so an overflow cannot flood the journal"


class TestTheDeafRadioProofDoesNotDependOnRecording:
    """R2-15: the first-sample proof was derived ONLY from the recorder's cf32, so it was
    unavailable whenever IQ recording was disabled — and the RX apps then skipped the check
    and declared `ready` with no evidence at all. A deaf radio was undetectable on exactly
    the passes that keep no recording. The proof must not depend on an unrelated flag."""

    def test_a_recorder_still_provides_the_proof(self, tmp_path) -> None:
        from _recorder import first_sample_probe

        iq = tmp_path / "p.cf32"
        iq.write_bytes(b"\x00" * 16)

        class _Rec:
            iq_path = iq

        probe = first_sample_probe(_Rec())
        assert callable(probe)
        assert probe() > 0

    def test_without_a_recorder_the_gr_source_provides_it(self) -> None:
        from _recorder import first_sample_probe

        class _Src:
            def __init__(self) -> None:
                self.n = 0

            def nitems_written(self, _port: int) -> int:
                return self.n

        src = _Src()
        probe = first_sample_probe(None, source=src)
        assert callable(probe), "recording off must NOT mean 'no proof'"
        assert probe() == 0          # nothing delivered yet
        src.n = 4096
        assert probe() == 4096       # the SDR provably delivered

    def test_a_probe_that_raises_is_not_fatal(self) -> None:
        from _recorder import first_sample_probe

        class _Bad:
            def nitems_written(self, _port: int) -> int:
                raise RuntimeError("block died")

        probe = first_sample_probe(None, source=_Bad())
        assert probe() == 0, "a probe must never kill the pass"

    def test_the_rx_apps_pass_their_source(self) -> None:
        for app in ("cubesat_gfsk_ax25_rx.py", "satellite_rx.py"):
            src = (_APPS / app).read_text(encoding="utf-8")
            assert 'source=getattr(ctx, "src", None)' in src, app

    def test_the_fm_rx_app_passes_its_source_and_fails_closed(self) -> None:
        """CA-FLOW-002: the FM app was the one RX app still deriving the proof from the
        recorder alone — recording off meant probe None, the check was SKIPPED, and
        ready carried first_samples=null for a deaf source. It must pass its source AND
        refuse ready when no proof can be constructed at all."""
        src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
        assert "first_sample_probe(ctx.recorder, source=ctx.src)" in src
        assert '"engine-no-proof"' in src, "an unprovable stream must REFUSE ready"
        assert "if probe is None:" in src, "the probe-None case must fail closed, not skip"

    def test_no_proof_available_returns_none(self) -> None:
        from _recorder import first_sample_probe

        assert first_sample_probe(None) is None  # nothing to prove with — caller must refuse


class TestIqViewsDoesNotShipPhantomOrMislabelledProducts:
    def test_a_png_that_was_never_written_is_not_reported(self, tmp_path) -> None:
        """R2-21: write_waterfall_png SKIPS a capture shorter than one FFT window (it refuses
        to fabricate an all-zeros image). Reporting the path anyway told the caller a product
        existed when nothing was written."""
        import json

        from iq_views import derive_views

        cf32 = tmp_path / "short.cf32"
        cf32.write_bytes(b"\x00" * 8 * 64)          # 64 samples: far short of one FFT window
        (tmp_path / "short.cf32.json").write_text(
            json.dumps({"sample_rate_hz": 48000.0, "center_hz": 401e6}), encoding="utf-8"
        )
        written = derive_views(
            cf32, center_hz=401e6, sample_rate_hz=48000.0, formats=("png",)
        )
        assert written == [], "reported a PNG that does not exist"
        assert not (tmp_path / "short.png").exists()

    def test_views_derived_without_a_sidecar_are_labelled_unverified(self, tmp_path) -> None:
        """R2-20: without the sidecar the rate/centre are the orchestrator's REQUEST, not what
        the engine used — every axis is a guess. The views are still worth having; presenting
        their axes as fact is not."""
        import numpy as np
        from iq_views import derive_views

        cf32 = tmp_path / "nosidecar.cf32"
        iq = (np.random.default_rng(0).standard_normal(8192)
              + 1j * np.random.default_rng(1).standard_normal(8192)).astype(np.complex64)
        cf32.write_bytes(iq.tobytes())
        written = derive_views(
            cf32, center_hz=401e6, sample_rate_hz=48000.0, formats=("png",)
        )
        assert written, "the views are still produced — the data is useful"
        # the honesty is in the title; the code path is pinned so it cannot be dropped
        src = (_APPS / "iq_views.py").read_text(encoding="utf-8")
        assert "AXES UNVERIFIED" in src


class TestDroppedAudioReachesThePassResult:
    def test_the_fm_app_emits_an_error_event_for_dropped_audio(self) -> None:
        """R2-16: the counter was rate-limit-logged but never left the process — gs-client,
        the backend and the customer all saw an ordinary successful pass whose product was
        silently missing audio."""
        src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
        assert '"code": "audio-dropped"' in src
        assert "await _report_audio_loss()" in src

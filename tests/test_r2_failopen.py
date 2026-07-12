"""Audit round 2 — the SILENT-SUCCESS defect class, swept out of gs-flowgraphs.

R2-02 (a recorder-only pass reporting a clean decode) was one instance of a pattern: **the
engine could not do the job it was asked to do, degraded silently, and the pass still
reported success.** An adversarial sweep found the rest of the family in this repo.

GNU Radio is not importable off-bench, so the *logic* lives in pure helpers that ARE
testable, and the wiring is pinned at the source level. Where a fix is purely structural
(the bidir reader re-raising, the TX app reporting a real BurstResult), the source pin is
the honest limit of what a dev box can prove — those are re-verified at Gate 5/6.
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


class TestTheTxAppReportsWhatItActuallyDid:
    """`transmit_complete outcome="complete", samples=0` was HARDCODED in the gnuradio TX
    branch — a fabricated success, directly beneath the comment forbidding exactly that."""

    def test_tx_app_no_longer_hardcodes_a_zero_sample_success(self) -> None:
        src = (_APPS / "cubesat_gfsk_ax25_tx.py").read_text(encoding="utf-8")
        assert '{"event": "transmit_complete", "samples": 0, "outcome": "complete"}' not in src
        assert '"code": "tx-engine-failed"' in src, "an engine that raises must say so"
        assert 'getattr(burst, "accepted", 0)' in src, "report the ACCEPTED sample count"

    def test_the_gnuradio_engine_returns_a_burst_result(self) -> None:
        src = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
        assert "BurstResult(" in src
        assert "outcome=\"complete\" if total > 0 else \"error\"" in src


class TestAnEngineThatCannotRunACommandSaysSo:
    def test_command_loop_emits_an_error_event_on_a_raising_handler(self) -> None:
        """The TX apps do ALL of their transmitting inside these handlers. A `transmit`
        that blew up was logged and then ignored — dispatch continued, the exit code was
        unaffected, and gs-client saw a pass that completed as if it had radiated."""
        src = (_APPS / "_spawn_contract.py").read_text(encoding="utf-8")
        assert '"code": "handler-failed"' in src
        assert "status_writer: asyncio.StreamWriter | None = None" in src


class TestTheBidirEngineDoesNotSwallowItsOwnDeath:
    def test_reader_failure_is_captured_not_laundered_into_a_clean_eof(self) -> None:
        """The reader used to push the SAME `None` terminator a clean EOF uses, so a dead
        SDR ended the pass NORMALLY: zero frames, no error event, a completed pass."""
        src = (_APPS / "cubesat_gfsk_endurosat_bidir.py").read_text(encoding="utf-8")
        assert "reader_error: list[BaseException] = []" in src
        assert "reader_error.append(e)" in src


class TestTheRecorderDoesNotInventArtifacts:
    def test_a_too_short_capture_yields_no_spectrogram_rows(self) -> None:
        """A zeros row became a uniform, perfectly plausible PNG — an operator reads that
        as 'the band was quiet'. An absent waterfall is honest; an invented one is not."""
        import numpy as np
        from _recorder import _spectrogram_db

        spec = _spectrogram_db(np.zeros(64, dtype=np.complex64), nfft=1024)
        assert spec.shape[0] == 0, "a capture shorter than one FFT window has NO waterfall"

    def test_a_real_capture_still_produces_rows(self) -> None:
        import numpy as np
        from _recorder import _spectrogram_db

        iq = (np.random.default_rng(0).standard_normal(4096)
              + 1j * np.random.default_rng(1).standard_normal(4096)).astype(np.complex64)
        assert _spectrogram_db(iq, nfft=1024).shape[0] > 0

    def test_the_sidecar_write_failure_is_logged_not_suppressed(self) -> None:
        src = (_APPS / "_recorder.py").read_text(encoding="utf-8")
        assert "IQ sidecar write FAILED" in src
        assert "with contextlib.suppress(OSError):\n        iq_path" not in src


class TestDroppedAudioIsVisible:
    def test_fm_rx_logs_its_drops(self) -> None:
        src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
        assert "audio queue FULL" in src
        assert "_DROP_LOG_EVERY" in src, "rate-limited, so an overflow cannot flood the journal"

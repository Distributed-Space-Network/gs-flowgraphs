"""R2-18 — TX SAFETY: the burst must announce itself, or the PA never gets de-keyed.

The GNU Radio TX branch was a SECOND transmit path that opened its own soapy sink and ran
its own graph. It never emitted ``transmit_started``, so gs-client's safety FSM stayed in
KEYED_READY — PA enabled, T/R switched to TX — and the orchestrator's immediate de-key,
which fires only from KEYED, never ran. The PA stayed energized until LOS. It also counted
SOURCE items instead of samples the SDR accepted, skipped the shared payload selection, and
never validated the hardware rate.

These tests DRIVE THE REAL BURST COROUTINE. The previous "test" grepped the source for a
string; that is what let the defect survive a green suite.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

import cubesat_gfsk_ax25_tx as tx  # noqa: E402
from _soapy_tx import BurstResult  # noqa: E402

from gfsk_ax25 import endurosat  # noqa: E402


class _Writer:
    """Captures the NDJSON events the app writes to its status socket."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, data: bytes) -> None:
        import json

        for line in data.decode().splitlines():
            if line.strip():
                self.events.append(json.loads(line))

    async def drain(self) -> None:
        return None

    def kinds(self) -> list[str]:
        return [e.get("event", "") for e in self.events]


class _Args:
    sdr_args = "driver=null"
    sample_rate = 96_000
    center_freq_hz = 401_500_000
    uplink_file = None


def _profile() -> endurosat.LinkProfile:
    return endurosat.LinkProfile()


async def _run(monkeypatch: pytest.MonkeyPatch, *, accepted: int, total: int,
               outcome: str = "complete", engine: str = "dsp",
               modulate_raises: Exception | None = None) -> _Writer:
    w = _Writer()

    def _fake_sink(args, iq, params=None, on_first_accept=None,  # noqa: ANN001
                   should_abort=None, *, cs16=None):
        assert iq is not None and len(iq) > 0, "the sink must be handed real IQ"
        if accepted > 0 and on_first_accept is not None:
            on_first_accept()          # the SDR provably accepted a sample
        return BurstResult(accepted=accepted, total=total, outcome=outcome)

    monkeypatch.setattr(tx, "_sink_iq", _fake_sink)
    if engine == "dsp":
        monkeypatch.setattr(
            tx, "_build_frame_iq",
            lambda *_a, **_k: np.ones(256, dtype=np.complex64),
        )
    else:  # the gnuradio engine: modulator only, no SDR of its own
        mod = type(sys)("gnuradio_gfsk")

        def _modulate(*_a, **_k):
            if modulate_raises is not None:
                raise modulate_raises
            return np.ones(256, dtype=np.complex64)

        mod.modulate_gnuradio = _modulate  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "gnuradio_gfsk", mod)

    await tx.emit_burst(w, _Args(), {}, _profile(), engine)
    # the first-accept callback hops through call_soon_threadsafe -> create_task
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return w


class TestTheBurstAnnouncesItself:
    @pytest.mark.parametrize("engine", ["dsp", "gnuradio"])
    def test_transmit_started_is_emitted_before_completion(
        self, monkeypatch: pytest.MonkeyPatch, engine: str
    ) -> None:
        """THE SAFETY INVARIANT: without transmit_started the safety FSM never leaves
        KEYED_READY, so the PA is never de-keyed at burst end. Both engines must emit it."""
        w = asyncio.run(_run(monkeypatch, accepted=256, total=256, engine=engine))
        kinds = w.kinds()
        assert "transmit_started" in kinds, (
            f"engine {engine!r} never announced its burst — the PA would stay keyed"
        )
        assert "transmit_complete" in kinds
        assert kinds.index("transmit_started") < kinds.index("transmit_complete")

    @pytest.mark.parametrize("engine", ["dsp", "gnuradio"])
    def test_completion_reports_the_accepted_count_not_a_fabricated_one(
        self, monkeypatch: pytest.MonkeyPatch, engine: str
    ) -> None:
        w = asyncio.run(
            _run(monkeypatch, accepted=200, total=256, outcome="stalled", engine=engine)
        )
        done = next(e for e in w.events if e["event"] == "transmit_complete")
        assert done["samples"] == 200      # what the SDR ACCEPTED
        assert done["outcome"] == "stalled"

    def test_a_burst_the_sdr_never_accepted_does_not_announce_a_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """transmit_started means 'the stream took a sample' — not 'we tried'."""
        w = asyncio.run(_run(monkeypatch, accepted=0, total=256, outcome="error"))
        assert "transmit_started" not in w.kinds()
        done = next(e for e in w.events if e["event"] == "transmit_complete")
        assert done["samples"] == 0
        assert done["outcome"] == "error"


class TestAFailedBurstTakesThePaDown:
    """AUDIT ROUND 4 (P0): emitting `tx-failed` and RETURNING NORMALLY is not PA safety.
    The burst runs inside the `start` command handler — returning cleanly leaves the app
    alive and the pass running while the PA/T-R chain is still energized (KEYED_READY with
    no accepted sample; KEYED after one). It must RAISE, so the command loop's
    handler-failure path ends dispatch, the app exits nonzero, and gs-client forces the PA
    off and fails the pass."""

    def test_a_failed_burst_reports_and_then_RAISES(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = _Writer()

        async def _go() -> None:
            mod = type(sys)("gnuradio_gfsk")

            def _boom(*_a, **_k):
                raise RuntimeError("GNU Radio blew up")

            mod.modulate_gnuradio = _boom  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, "gnuradio_gfsk", mod)
            await tx.emit_burst(w, _Args(), {}, _profile(), "gnuradio")

        with pytest.raises(RuntimeError, match="GNU Radio blew up"):
            asyncio.run(_go())

        kinds = w.kinds()
        assert "error" in kinds, "the reason must still reach gs-client before we die"
        assert "transmit_complete" not in kinds, "a failed burst must not report completion"
        err = next(e for e in w.events if e["event"] == "error")
        assert err["code"] == "tx-failed"

    def test_a_sink_failure_also_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        w = _Writer()

        def _boom_sink(*_a, **_k):
            raise OSError("SoapySDR write failed")

        monkeypatch.setattr(tx, "_sink_iq", _boom_sink)
        monkeypatch.setattr(
            tx, "_build_frame_iq", lambda *_a, **_k: np.ones(256, dtype=np.complex64)
        )
        with pytest.raises(OSError, match="SoapySDR write failed"):
            asyncio.run(tx.emit_burst(w, _Args(), {}, _profile(), "dsp"))
        assert any(e.get("code") == "tx-failed" for e in w.events)


class TestBothEnginesShareTheOneSink:
    def test_the_gnuradio_engine_no_longer_owns_an_sdr_path(self) -> None:
        """Structural: the second transmit path is GONE, not patched. (Bench-only import,
        so the behaviour above is proven with a stub; this pins that the divergence cannot
        come back.)"""
        src = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
        assert "def transmit_gnuradio" not in src
        assert "def modulate_gnuradio" in src
        assert "make_sink(" not in src.split("def modulate_gnuradio")[1], (
            "the modulator must not open an SDR sink of its own"
        )

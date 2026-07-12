"""R2-02: a recorder-only pass must SAY it built no decoder.

The engine builds our numpy demod chain only when the pass carries BOTH a modulation and a
positive symbol rate. The backend omits ``symbol_rate_hz`` whenever the transmitter record
has a null/zero baud — so with ``GS_GRSAT_LIVE`` unset there is no decoder at all and the
graph degrades to source -> rotator -> decimator -> file-sink. It records IQ, produces ZERO
frames, and used to emit a perfectly ordinary ``ready`` event: the pass came back green,
with no frames and nothing saying why.

The reason lives in a PURE helper (no GNU Radio) precisely so it can be tested off-bench;
``gnuradio_satellites`` puts it on the ``ready`` event, and gs-client carries it into the
terminal PassResult (gs-client tests/unit/test_r2_02_recorder_only.py).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

from _fallback_select import no_decode_reason  # noqa: E402


class TestNoDecoderIsReported:
    def test_null_baud_with_grsat_off_is_the_repro(self) -> None:
        """THE bug: no demod params (null-baud transmitter) + gr-satellites gated off."""
        why = no_decode_reason(has_decode_consumer=False, mode=None, grsat_live=False)
        assert why, "a recorder-only graph must report WHY it cannot decode"
        assert "null/zero baud" in why
        assert "GS_GRSAT_LIVE is unset" in why

    def test_null_baud_with_grsat_on_does_not_blame_the_flag(self) -> None:
        why = no_decode_reason(has_decode_consumer=False, mode=None, grsat_live=True)
        assert why
        assert "GS_GRSAT_LIVE" not in why  # the flag was ON; the missing baud is the cause

    def test_a_demod_chain_that_failed_to_build_is_reported_distinctly(self) -> None:
        why = no_decode_reason(
            has_decode_consumer=False, mode=("gfsk", 9600.0), grsat_live=False, framing="ax25"
        )
        assert "failed to construct" in why
        assert "gfsk@9600" in why
        assert "ax25" in why


class TestADecodingGraphIsNotDegraded:
    @pytest.mark.parametrize("mode", [None, ("gfsk", 9600.0)])
    @pytest.mark.parametrize("grsat", [True, False])
    def test_any_decode_consumer_means_no_reason(
        self, mode: tuple[str, float] | None, grsat: bool
    ) -> None:
        """A graph that HAS a decoder must never be marked degraded, whatever built it."""
        assert no_decode_reason(has_decode_consumer=True, mode=mode, grsat_live=grsat) == ""


def test_the_engine_puts_the_flag_on_the_ready_event() -> None:
    """The wiring the gs-client fix depends on: satellite_rx must publish `decode_built`
    (and the reason) on `ready`. GNU Radio is not importable off-bench, so this is pinned
    at the source level — the LOGIC itself is executable above and in gs-client."""
    src = (_APPS / "satellite_rx.py").read_text(encoding="utf-8")
    assert re.search(r'"decode_built":\s*not no_decode_reason', src), (
        "satellite_rx must declare decode_built on its ready event"
    )
    assert '"no_decode_reason": no_decode_reason' in src

    engine = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")
    assert "no_decode_reason(" in engine, "the engine must compute the reason"
    assert "no_decode_reason=reason" in engine, "and pass it into the context it returns"

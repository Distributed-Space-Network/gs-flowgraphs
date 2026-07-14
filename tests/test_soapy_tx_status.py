"""Finding #22 regression: a burst the driver ACCEPTED then DISCARDED must not
report as a clean transmission.

writeStream returning ret>0 only proves the buffer entered the driver's DMA queue.
The XTRX bench probe (tools/probe_soapy_tx_write.py) documents libxtrx accepting a
buffer and then discarding it late ("TX DMA ... skip due to TO buffers"), surfaced
ONLY through readStreamStatus (UNDERFLOW / TIME_ERROR / END_ABRUPT). The pre-fix
write_burst never inspected it, so a dead uplink was indistinguishable from a good
one. write_burst now drains the stream status (bounded) after a fully-accepted burst
and downgrades a driver-reported discard to outcome="discarded".
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

SOAPY_TIMEOUT = -1
SOAPY_NOT_SUPPORTED = -5
SOAPY_UNDERFLOW = -7
SOAPY_TIME_ERROR = -6
SOAPY_END_ABRUPT = 8


@pytest.fixture(autouse=True)
def fake_soapysdr(monkeypatch):
    mod = types.ModuleType("SoapySDR")
    mod.SOAPY_SDR_END_BURST = 2
    mod.SOAPY_SDR_TIMEOUT = SOAPY_TIMEOUT
    mod.SOAPY_SDR_NOT_SUPPORTED = SOAPY_NOT_SUPPORTED
    mod.SOAPY_SDR_UNDERFLOW = SOAPY_UNDERFLOW
    mod.SOAPY_SDR_TIME_ERROR = SOAPY_TIME_ERROR
    mod.SOAPY_SDR_STREAM_ERROR = -2
    mod.SOAPY_SDR_CORRUPTION = -3
    mod.SOAPY_SDR_END_ABRUPT = SOAPY_END_ABRUPT
    monkeypatch.setitem(sys.modules, "SoapySDR", mod)
    return mod


from _soapy_tx import poll_tx_status, write_burst  # noqa: E402


class _Result:
    def __init__(self, ret: int, flags: int = 0) -> None:
        self.ret = ret
        self.flags = flags


class _Dev:
    """Fake device that accepts every write (echo) and replays a scripted list of
    readStreamStatus results (each an int ``ret`` or an (ret, flags) tuple)."""

    def __init__(self, *, mtu: int = 1000, status=None, no_status: bool = False) -> None:
        self._mtu = mtu
        self._status = list(status or [])
        self._no_status = no_status
        self.status_calls = 0

    def getStreamMTU(self, _stream) -> int:
        return self._mtu

    def writeStream(self, _stream, buffs, num_elems, *extra):
        # New call shape: exactly (stream, buffs, num_elems) — no flags/END_BURST.
        assert extra == (), "writeStream must be called with exactly three arguments"
        del buffs
        return _Result(int(num_elems))

    def readStreamStatus(self, _stream, *args, **kwargs):
        del args, kwargs
        if self._no_status:
            raise RuntimeError("status read failed")
        self.status_calls += 1
        item = self._status.pop(0) if self._status else SOAPY_TIMEOUT
        if isinstance(item, tuple):
            return _Result(int(item[0]), int(item[1]))
        return _Result(int(item))


def _buf(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.complex64)


class TestDiscardIsNotSuccess:
    def test_underflow_status_downgrades_to_discarded(self):
        # The exact XTRX failure: every sample accepted, then the driver reports an
        # underflow (buffers discarded as late). Pre-fix this was outcome="complete".
        dev = _Dev(mtu=1000, status=[SOAPY_UNDERFLOW])
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.outcome == "discarded"
        assert result.accepted == 2000  # writeStream took them all...
        assert not result.complete  # ...but the burst did NOT radiate cleanly
        assert "underflow" in result.detail

    def test_time_error_status_downgrades_to_discarded(self):
        dev = _Dev(mtu=1000, status=[SOAPY_TIME_ERROR])
        result = write_burst(dev, object(), _buf(1500), mtu=1000)
        assert result.outcome == "discarded"

    def test_end_abrupt_flag_downgrades_to_discarded(self):
        # A status event with ret==0 but the END_ABRUPT flag = discarded burst.
        dev = _Dev(mtu=1000, status=[(0, SOAPY_END_ABRUPT)])
        result = write_burst(dev, object(), _buf(1500), mtu=1000)
        assert result.outcome == "discarded"
        assert "END_ABRUPT" in result.detail

    def test_clean_status_stays_complete(self):
        # readStreamStatus reports nothing pending (TIMEOUT) → genuinely clean.
        dev = _Dev(mtu=1000, status=[SOAPY_TIMEOUT])
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.complete and result.outcome == "complete"

    def test_unexpected_negative_status_downgrades_to_discarded(self):
        # (3g) an UNEXPECTED negative status ret (not TIMEOUT/NOT_SUPPORTED, not a named
        # discard) is treated as a discard, not silently swallowed.
        dev = _Dev(mtu=1000, status=[-99])
        result = write_burst(dev, object(), _buf(1500), mtu=1000)
        assert result.outcome == "discarded"
        assert "unexpected negative" in result.detail

    def test_benign_event_then_timeout_stays_complete(self):
        # A benign status event (ret==0, no bad flag) followed by TIMEOUT is clean.
        dev = _Dev(mtu=1000, status=[(0, 0), SOAPY_TIMEOUT])
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.complete

    def test_status_read_error_leaves_complete_not_fabricated(self):
        # If status can't be read we must NOT invent a discard (false failure would
        # abort a good pass) — the burst stays "complete".
        dev = _Dev(mtu=1000, no_status=True)
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.complete

    def test_device_without_status_reads_stays_complete(self):
        class _NoStatusDev:
            def getStreamMTU(self, _s):
                return 1000

            def writeStream(self, _s, buffs, num_elems, *extra):
                del buffs, extra
                return _Result(int(num_elems))

        result = write_burst(_NoStatusDev(), object(), _buf(1000), mtu=1000)
        assert result.complete

    def test_poll_status_disabled_skips_the_check(self):
        dev = _Dev(mtu=1000, status=[SOAPY_UNDERFLOW])
        result = write_burst(dev, object(), _buf(1000), mtu=1000, poll_status=False)
        assert result.complete
        assert dev.status_calls == 0


class TestPollBounded:
    def test_poll_is_bounded_on_endless_benign_events(self):
        # A driver that keeps returning benign ret==0 events must not spin forever:
        # bounded by the poll cap. (A hard wall-clock guard proves it terminates.)
        import threading

        class _Endless:
            def readStreamStatus(self, _s, *a, **k):
                return _Result(0, 0)

        done: list[tuple[bool, str]] = []

        def run() -> None:
            done.append(poll_tx_status(_Endless(), object()))

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "poll_tx_status did not terminate — poll cap missing"
        assert done == [(False, "")]

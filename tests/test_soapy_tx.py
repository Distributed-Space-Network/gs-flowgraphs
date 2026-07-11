"""F-03 regressions for the shared production Soapy TX transport (P0-07).

Pre-fix production loops advanced only on ret>0 (zero-return = infinite spin),
used fixed 4096-sample chunks regardless of the driver MTU (oversized writes
can segfault native drivers), never placed END_BURST, and had no deadline or
cancellation. Every one of those is pinned here against a fake device.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

SOAPY_TIMEOUT = -1


@pytest.fixture(autouse=True)
def fake_soapysdr(monkeypatch):
    mod = types.ModuleType("SoapySDR")
    mod.SOAPY_SDR_END_BURST = 2
    mod.SOAPY_SDR_TIMEOUT = SOAPY_TIMEOUT
    monkeypatch.setitem(sys.modules, "SoapySDR", mod)
    return mod


from _soapy_tx import BurstResult, query_tx_mtu, write_burst  # noqa: E402


class _Result:
    def __init__(self, ret: int, flags: int = 0) -> None:
        self.ret = ret
        self.flags = flags


class _Dev:
    """Scriptable fake SoapySDR device. ``script`` yields per-call returns:
    an int (samples accepted / error code) or "echo" (accept whole chunk)."""

    def __init__(self, *, mtu: int = 1000, script=None, full_call: bool = True) -> None:
        self._mtu = mtu
        self._script = list(script or [])
        self._full_call = full_call
        self.writes: list[tuple[int, int]] = []  # (num_elems, flags)

    def getStreamMTU(self, _stream) -> int:
        return self._mtu

    def writeStream(self, _stream, buffs, num_elems, flags, *rest):
        if not self._full_call and rest:
            raise TypeError("binding takes (stream, buffs, numElems, flags)")
        del buffs
        self.writes.append((int(num_elems), int(flags)))
        action = self._script.pop(0) if self._script else "echo"
        if action == "echo":
            return _Result(int(num_elems))
        return _Result(int(action))


def _buf(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.complex64)


class TestMtu:
    def test_query_uses_driver_value(self):
        assert query_tx_mtu(_Dev(mtu=1360), object()) == 1360

    def test_query_falls_back_on_error_or_nonsense(self):
        class _NoMtu:
            def getStreamMTU(self, _s):
                raise RuntimeError("unsupported")

        assert query_tx_mtu(_NoMtu(), object(), fallback=4096) == 4096
        assert query_tx_mtu(_Dev(mtu=0), object(), fallback=2048) == 2048


class TestWriteBurst:
    def test_chunks_never_exceed_mtu_and_end_burst_on_last_data_chunk(self):
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), _buf(2500), mtu=1000)
        assert result.complete and result.accepted == 2500
        assert [n for n, _ in dev.writes] == [1000, 1000, 500]
        # END_BURST rides the LAST DATA chunk only (a separate 0-length
        # END_BURST write blocks XTRX/LMS drivers).
        assert [f for _, f in dev.writes] == [0, 0, 2]

    def test_partial_accepts_advance_by_actual_ret(self):
        dev = _Dev(mtu=1000, script=[300, "echo", "echo", "echo"])
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.complete
        # first write accepted 300 of 1000; the burst still totals 2000
        assert result.accepted == 2000

    def test_zero_return_is_bounded_not_an_infinite_spin(self):
        # P0-07 repro: the pre-fix loop did `i += ret if ret > 0 else 0`.
        dev = _Dev(mtu=1000, script=[0] * 100)
        result = write_burst(dev, object(), _buf(2000), mtu=1000, max_stalls=5)
        assert result.outcome == "stalled"
        assert result.accepted == 0
        assert len(dev.writes) == 6  # bounded: max_stalls + the tripping write

    def test_timeout_returns_are_bounded(self):
        dev = _Dev(mtu=1000, script=[SOAPY_TIMEOUT] * 100)
        result = write_burst(dev, object(), _buf(2000), mtu=1000, max_stalls=3)
        assert result.outcome == "stalled"

    def test_error_return_aborts_with_explicit_outcome(self):
        dev = _Dev(mtu=1000, script=["echo", -7])
        result = write_burst(dev, object(), _buf(2000), mtu=1000)
        assert result.outcome == "error"
        assert result.accepted == 1000
        assert "-7" in result.detail

    def test_total_deadline_ends_a_slow_burst(self):
        dev = _Dev(mtu=10, script=[1] * 10_000)  # 1 sample per write: crawls
        result = write_burst(dev, object(), _buf(5_000), mtu=10, deadline_s=0.0)
        assert result.outcome == "deadline"

    def test_cancellation_ends_the_burst(self):
        calls = {"n": 0}

        def abort() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        dev = _Dev(mtu=100)
        result = write_burst(dev, object(), _buf(10_000), mtu=100, should_abort=abort)
        assert result.outcome == "cancelled"
        assert 0 < result.accepted < 10_000

    def test_first_accept_fires_exactly_once(self):
        fired: list[int] = []
        dev = _Dev(mtu=1000)
        write_burst(
            dev, object(), _buf(3000), mtu=1000, on_first_accept=lambda: fired.append(1)
        )
        assert fired == [1]

    def test_no_first_accept_when_nothing_accepted(self):
        fired: list[int] = []
        dev = _Dev(mtu=1000, script=[-7])
        result = write_burst(
            dev, object(), _buf(3000), mtu=1000, on_first_accept=lambda: fired.append(1)
        )
        assert fired == [] and result.outcome == "error"

    def test_binding_without_timeout_args_falls_back(self):
        dev = _Dev(mtu=1000, full_call=False)
        result = write_burst(dev, object(), _buf(1500), mtu=1000)
        assert result.complete
        assert [n for n, _ in dev.writes] == [1000, 500]

    def test_empty_buffer_is_a_complete_noop(self):
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), _buf(0), mtu=1000)
        assert result == BurstResult(accepted=0, total=0, outcome="complete")
        assert dev.writes == []

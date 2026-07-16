"""F-03 regressions for the shared production Soapy TX transport (P0-07).

The send behaviour is the KNOWN-GOOD shape distilled from the XTRX bench probe
(``tools/probe_soapy_tx_write.py``): ONE call ``writeStream(stream, [chunk],
num_elems)`` — three arguments, no flags / timestamp / timeout overload and no
``END_BURST`` (a separate 0-length END_BURST write blocks XTRX/LMS); a REQUIRED
positive MTU (never a guessed fallback); a ``min(1024, MTU)`` chunk; a flat CS16
buffer sliced two int16 per complex sample; advance only by the positive accepted
count; reject a return greater than requested; bound zero (no-progress) returns;
treat any negative result as an error. Each is pinned here against a fake device.

This file was rewritten: the pre-fix behaviour it used to pin (a 4096 fallback
MTU, flagged/END_BURST writes, the 6-arg call with a per-binding fallback,
timeout-as-stall) is exactly the wrong behaviour, so those tests are gone.

Soapy is exercised ONLY through a fake device — no real XTRX is touched.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

SOAPY_TIMEOUT = -1


@pytest.fixture(autouse=True)
def fake_soapysdr(monkeypatch):
    # write_burst/query_tx_mtu touch no SoapySDR symbols; poll_tx_status imports
    # them lazily. A minimal fake module keeps the whole path importable/testable.
    mod = types.ModuleType("SoapySDR")
    mod.SOAPY_SDR_TIMEOUT = SOAPY_TIMEOUT
    mod.SOAPY_SDR_NOT_SUPPORTED = -5
    monkeypatch.setitem(sys.modules, "SoapySDR", mod)
    return mod


from _soapy_tx import (  # noqa: E402
    TxGainConfigError,
    named_tx_gains,
    query_tx_mtu,
    to_cs16,
    verify_named_tx_gains,
    write_burst,
)


class _Result:
    def __init__(self, ret: int) -> None:
        self.ret = ret


class _Dev:
    """Scriptable fake SoapySDR device. ``script`` yields per-call returns:
    an int (samples accepted / error code) or "echo" (accept the whole chunk).

    ``writeStream`` deliberately declares ONLY ``(stream, buffs, num_elems)`` plus
    a ``*extra`` catch-all, so any flags/timestamp/timeout argument the production
    code might pass shows up as an over-length call (arg_lens) instead of a silent
    TypeError — the "exactly three arguments" invariant is then assertable."""

    def __init__(self, *, mtu: int = 1000, script=None) -> None:
        self._mtu = mtu
        self._script = list(script or [])
        self.writes: list[int] = []           # num_elems per call (COMPLEX count)
        self.blocks: list[np.ndarray] = []    # the buffer slice per call
        self.arg_lens: list[int] = []         # positional args writeStream received

    def getStreamMTU(self, _stream) -> int:
        return self._mtu

    def writeStream(self, _stream, buffs, num_elems, *extra):
        self.arg_lens.append(3 + len(extra))  # stream + buffs + num_elems + any extras
        self.writes.append(int(num_elems))
        self.blocks.append(np.asarray(buffs[0]))
        action = self._script.pop(0) if self._script else "echo"
        ret = int(num_elems) if action == "echo" else int(action)
        return _Result(ret)


def _cbuf(n: int) -> np.ndarray:
    """A complex64 burst (CF32 sink path): one element per complex sample."""
    return np.zeros(n, dtype=np.complex64)


def _flat_cs16(n_complex: int) -> np.ndarray:
    """A flat CS16 burst [I0,Q0,I1,Q1,...] with distinct values so slicing is checkable."""
    return np.arange(2 * n_complex, dtype=np.int16)


class TestMtu:
    def test_query_uses_driver_value(self):
        assert query_tx_mtu(_Dev(mtu=1360), object()) == 1360

    def test_nonpositive_mtu_is_rejected_not_replaced(self):
        # Spec: require a positive MTU; NEVER substitute a guessed fallback (an
        # oversized/fabricated write can segfault a native driver).
        with pytest.raises(ValueError):
            query_tx_mtu(_Dev(mtu=0), object())
        with pytest.raises(ValueError):
            query_tx_mtu(_Dev(mtu=-1), object())

    def test_unsupported_mtu_call_propagates_fail_closed(self):
        class _NoMtu:
            def getStreamMTU(self, _s):
                raise RuntimeError("unsupported")

        # No fallback: the driver error propagates so the caller fails closed.
        with pytest.raises(RuntimeError):
            query_tx_mtu(_NoMtu(), object())


class TestCallShape:
    def test_writestream_always_receives_exactly_three_args(self):
        dev = _Dev(mtu=1000)
        write_burst(dev, object(), _cbuf(2500), mtu=1000)
        assert dev.arg_lens == [3, 3, 3]  # no flags / timestamp / timeout / END_BURST

    def test_chunk_is_min_1024_and_mtu(self):
        # MTU larger than 1024 → capped at 1024.
        dev = _Dev(mtu=4096)
        write_burst(dev, object(), _cbuf(2500), mtu=4096)
        assert dev.writes == [1024, 1024, 452]
        # MTU smaller than 1024 → the MTU bounds the chunk.
        dev = _Dev(mtu=600)
        write_burst(dev, object(), _cbuf(1000), mtu=600)
        assert dev.writes == [600, 400]


class TestWriteBurst:
    def test_chunks_never_exceed_mtu_and_no_flags(self):
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), _cbuf(2500), mtu=1000)
        assert result.complete and result.accepted == 2500
        assert dev.writes == [1000, 1000, 500]

    def test_partial_accepts_advance_by_actual_ret(self):
        dev = _Dev(mtu=1000, script=[300, "echo", "echo", "echo"])
        result = write_burst(dev, object(), _cbuf(2000), mtu=1000)
        assert result.complete
        # first write accepted 300 of 1000; the burst still totals 2000. The
        # remainder stays a full 1000-chunk until fewer than 1000 samples are left.
        assert result.accepted == 2000
        assert dev.writes == [1000, 1000, 700]

    def test_zero_return_is_bounded_not_an_infinite_spin(self):
        # P0-07 repro: the pre-fix loop did `i += ret if ret > 0 else 0`.
        dev = _Dev(mtu=1000, script=[0] * 100)
        result = write_burst(dev, object(), _cbuf(2000), mtu=1000, max_stalls=5)
        assert result.outcome == "stalled"
        assert result.accepted == 0
        assert len(dev.writes) == 6  # bounded: max_stalls + the tripping write

    def test_negative_return_is_an_error_not_a_stall(self):
        # New rule: any negative Soapy result (incl. TIMEOUT) is an error, matching
        # the bench probe's ret<=0 = failure. It must NOT be retried like a zero.
        dev = _Dev(mtu=1000, script=[SOAPY_TIMEOUT])
        result = write_burst(dev, object(), _cbuf(2000), mtu=1000, max_stalls=20)
        assert result.outcome == "error"
        assert result.accepted == 0
        assert len(dev.writes) == 1  # errored on the first write, no retry loop

    def test_error_return_aborts_with_explicit_outcome(self):
        dev = _Dev(mtu=1000, script=["echo", -7])
        result = write_burst(dev, object(), _cbuf(2000), mtu=1000)
        assert result.outcome == "error"
        assert result.accepted == 1000
        assert "-7" in result.detail

    def test_return_greater_than_requested_is_rejected(self):
        # A driver cannot accept more than offered; a return past the request is
        # corrupt, not progress.
        dev = _Dev(mtu=1000, script=[1500])
        result = write_burst(dev, object(), _cbuf(2000), mtu=1000)
        assert result.outcome == "error"
        assert "exceeds requested" in result.detail
        assert result.accepted == 0

    def test_total_deadline_ends_a_slow_burst(self):
        dev = _Dev(mtu=10, script=[1] * 10_000)  # 1 sample per write: crawls
        result = write_burst(dev, object(), _cbuf(5_000), mtu=10, deadline_s=0.0)
        assert result.outcome == "deadline"

    def test_cancellation_stops_further_writes(self):
        calls = {"n": 0}

        def abort() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        dev = _Dev(mtu=100)
        result = write_burst(dev, object(), _cbuf(10_000), mtu=100, should_abort=abort)
        assert result.outcome == "cancelled"
        assert 0 < result.accepted < 10_000
        # authority revoked → no further writes queued after the abort trips
        assert len(dev.writes) == 3

    def test_first_accept_fires_exactly_once_on_first_positive(self):
        fired: list[int] = []
        dev = _Dev(mtu=1000)
        write_burst(
            dev, object(), _cbuf(3000), mtu=1000, on_first_accept=lambda: fired.append(1)
        )
        assert fired == [1]

    def test_no_first_accept_and_no_success_when_nothing_accepted(self):
        fired: list[int] = []
        dev = _Dev(mtu=1000, script=[-7])
        result = write_burst(
            dev, object(), _cbuf(3000), mtu=1000, on_first_accept=lambda: fired.append(1)
        )
        # zero accepted samples can NEVER be reported as a successful transmission
        assert fired == []
        assert result.outcome == "error"
        assert not result.complete
        assert result.accepted == 0

    def test_empty_buffer_is_an_error_not_a_silent_success(self):
        # (3g) an empty burst is a failure, never a fabricated "complete" — a caller that
        # reached the write path with nothing to send has already failed.
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), _cbuf(0), mtu=1000)
        assert result.outcome == "error" and not result.complete
        assert result.accepted == 0
        assert dev.writes == []


class TestFlatCs16:
    def test_two_int16_per_complex_sample(self):
        n_complex = 2500
        flat = _flat_cs16(n_complex)
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), flat, mtu=1000)
        assert result.complete and result.accepted == n_complex
        assert result.total == n_complex
        # num_elems count COMPLEX samples, not int16 elements
        assert dev.writes == [1000, 1000, 500]
        # each block is exactly 2*num_elems int16, and the concatenation of the
        # blocks reproduces the original flat buffer (no skip, no duplication)
        for num, block in zip(dev.writes, dev.blocks, strict=True):
            assert block.dtype == np.int16
            assert len(block) == 2 * num
        assert np.array_equal(np.concatenate(dev.blocks), flat)

    def test_partial_write_slices_at_two_int16_per_complex(self):
        n_complex = 2000
        flat = _flat_cs16(n_complex)
        dev = _Dev(mtu=1000, script=[300, "echo", "echo"])
        result = write_burst(dev, object(), flat, mtu=1000)
        assert result.complete and result.accepted == n_complex
        assert dev.writes == [1000, 1000, 700]
        # after accepting 300 complex, the next slice starts at int16 offset 600
        # (== 2*300): proves the flat buffer is indexed as buf[2*i : 2*(i+n)]
        assert np.array_equal(dev.blocks[0], flat[0:2000])
        assert np.array_equal(dev.blocks[1], flat[600:2600])
        assert np.array_equal(dev.blocks[2], flat[2600:4000])


class TestToCs16:
    """to_cs16 is the shared pre-key packer both TX sinks use to produce the flat
    CS16 buffer write_burst streams (the XTRX probe's exact packing)."""

    def test_packs_interleaved_two_int16_per_complex(self):
        iq = np.array([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
        out = to_cs16(iq)
        assert out.dtype == np.int16
        assert out.size == 2 * iq.size  # [I0,Q0,I1,Q1,...]
        assert (out[0], out[1]) == (0, 0)                 # 0+0j
        assert (out[2], out[3]) == (32767, 0)             # +1 → +full-scale I
        assert (out[4], out[5]) == (-32767, 0)            # -1 → -full-scale I
        assert (out[6], out[7]) == (0, int(np.rint(0.5 * 32767.0)))  # 0+0.5j

    def test_clips_out_of_range_to_the_rail_not_wrap(self):
        # |value|>1 must clip to the int16 rail, never wrap to the opposite sign.
        out = to_cs16(np.array([2.0 - 2.0j], dtype=np.complex64))
        assert (out[0], out[1]) == (32767, -32767)

    def test_output_is_contiguous(self):
        out = to_cs16(np.zeros(10, dtype=np.complex64))
        assert out.flags.c_contiguous and out.size == 20


class TestNamedTxGains:
    """(3c) TX drive must be a named per-element gain (PAD); the overall setGain overload
    is never used, so 'no named gain' is a hard configuration error."""

    def test_named_gains_returned(self):
        assert named_tx_gains({"sdr_gains": {"PAD": 52.0, "IAMP": 6}}) == {"PAD": 52.0, "IAMP": 6.0}

    def test_no_named_gain_is_a_config_error(self):
        with pytest.raises(TxGainConfigError):
            named_tx_gains({})
        with pytest.raises(TxGainConfigError):
            named_tx_gains({"sdr_gains": {}})

    def test_overall_gain_only_is_still_a_config_error(self):
        # An explicit overall gain does NOT satisfy the requirement — the overall setGain
        # overload is XTRX-unsafe and is never the fallback.
        with pytest.raises(TxGainConfigError):
            named_tx_gains({"sdr_gain_db": 30.0})

    def test_named_gain_WITHOUT_pad_is_a_config_error(self):
        # RE-AUDIT (P2): PAD is required SPECIFICALLY. A named gain that is NOT PAD (e.g. IAMP
        # alone, a digital preamp) does not set the XTRX TX OUTPUT drive, so it must be refused —
        # before this fix any named gain passed.
        with pytest.raises(TxGainConfigError):
            named_tx_gains({"sdr_gains": {"IAMP": 6.0}})
        # PAD present (any case) is accepted, even alongside others.
        assert named_tx_gains({"sdr_gains": {"pad": 40.0}}) == {"pad": 40.0}

    def test_non_numeric_entries_are_ignored(self):
        with pytest.raises(TxGainConfigError):
            named_tx_gains({"sdr_gains": {"PAD": "loud", 7: 3, "IAMP": True}})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_gain_is_a_config_error(self, bad):
        # DS-016 (P1): a NaN/inf PAD would be written to the driver AND defeats the readback check
        # (abs(readback - NaN) is never > tolerance). It must be refused before setGain.
        with pytest.raises(TxGainConfigError, match="non-finite"):
            named_tx_gains({"sdr_gains": {"PAD": bad}})


class TestVerifyNamedTxGains:
    """DS-016 (P1): verify_named_tx_gains must not trust a non-finite REQUESTED value even if the
    readback is finite — abs(got - NaN) is nan and never exceeds tolerance, so a NaN request would
    otherwise pass verification at whatever drive the driver actually latched."""

    class _GainDev:
        def __init__(self, readback: float) -> None:
            self._readback = readback

        def getGain(self, _direction, _channel, _name) -> float:
            return self._readback

    def test_finite_request_and_matching_readback_passes(self):
        dev = self._GainDev(52.0)
        assert verify_named_tx_gains(dev, 1, {"PAD": 52.0}) == {"PAD": 52.0}

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_request_is_refused_even_with_finite_readback(self, bad):
        dev = self._GainDev(52.0)  # driver latched a plausible value
        with pytest.raises(TxGainConfigError, match="non-finite"):
            verify_named_tx_gains(dev, 1, {"PAD": bad})

    def test_write_burst_consumes_to_cs16_output_as_flat_cs16(self):
        # End-to-end: to_cs16 → write_burst treats it as flat CS16 (num_elems are
        # COMPLEX samples, two int16 each).
        flat = to_cs16(np.zeros(2500, dtype=np.complex64))
        dev = _Dev(mtu=1000)
        result = write_burst(dev, object(), flat, mtu=1000)
        assert result.complete and result.accepted == 2500
        assert dev.writes == [1000, 1000, 500]

"""Production TX-sink SETUP + keyed-window shape, exercised with a FAKE Soapy device only.

The XTRX bench probe (tools/probe_soapy_tx_write.py) proved a specific send shape on real
hardware. These tests pin that the production bidir sink now matches it AND the Codex follow-up
requirements:

* the STAGED buffer is the FINAL flat CS16 (Doppler/resample/pack are pre-key) — transmit_burst
  does NO DSP, it only opens the TX stream and writes;
* ``setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16)`` with NO explicit ``[0]`` channel list;
* (3d) getStreamMTU is queried BEFORE activateStream;
* NO sleep between activate and the first write;
* (3e) if the RX stream cannot be deactivated, TX is REFUSED (never keyed on an un-broken stream);
* (3f) on every exit each cleanup step (TX deactivate, TX close, RX restore) is independent;
* (3g) an empty buffer is an error, never a silent success;
* (3c) the named PAD gain is required — the overall setGain overload is refused.

No real XTRX is touched. ``_SoapyBidirIo.transmit_burst`` is driven directly (its heavy ``__init__``
is bypassed); the ax25 gain path is driven through the real ``configure_tx_sink``.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import cubesat_gfsk_ax25_tx as txapp
import cubesat_gfsk_endurosat_bidir as bidir
import numpy as np
import pytest
from _soapy_tx import TxGainConfigError

# Real SoapySDR format constants are strings; directions are ints.
_TX = 1
_RX = 0
_CS16 = "CS16"
_CF32 = "CF32"
_RX_STREAM = ("rx-stream",)


@pytest.fixture(autouse=True)
def fake_soapysdr(monkeypatch):
    """poll_tx_status (inside write_burst) does ``import SoapySDR`` for the status
    constants — provide a minimal fake so the whole path runs with no real driver."""
    mod = types.ModuleType("SoapySDR")
    mod.SOAPY_SDR_TX = _TX
    mod.SOAPY_SDR_RX = _RX
    mod.SOAPY_SDR_CS16 = _CS16
    mod.SOAPY_SDR_CF32 = _CF32
    mod.SOAPY_SDR_TIMEOUT = -1
    mod.SOAPY_SDR_NOT_SUPPORTED = -5
    mod.SOAPY_SDR_UNDERFLOW = -7
    mod.SOAPY_SDR_TIME_ERROR = -6
    mod.SOAPY_SDR_STREAM_ERROR = -2
    mod.SOAPY_SDR_CORRUPTION = -3
    mod.SOAPY_SDR_END_ABRUPT = 8
    monkeypatch.setitem(sys.modules, "SoapySDR", mod)
    return mod


class _FakeSdrDev:
    """Records the ordered Soapy call sequence a TX burst makes. Accepts every write (echo)
    unless ``write_ret`` scripts a return code. ``deactivate_fail`` (a predicate on the stream)
    makes deactivateStream raise, to drive the RX-break-fail (3e) and independent-cleanup (3f)
    paths."""

    def __init__(self, *, mtu: int = 1024, write_ret=None, deactivate_fail=None) -> None:
        self._mtu = mtu
        self._write_ret = list(write_ret or [])
        self._deactivate_fail = deactivate_fail
        self.ops: list[tuple] = []
        self.setup_args: list[tuple] = []
        self.writes: list[tuple[int, np.dtype, int]] = []  # (num_elems, dtype, int16-count)
        self._n = 0

    def setupStream(self, direction, fmt, *extra):
        self._n += 1
        stream = ("stream", direction, self._n)
        self.setup_args.append((direction, fmt, extra))
        self.ops.append(("setup", stream, direction, fmt, extra))
        return stream

    def getStreamMTU(self, _stream) -> int:
        self.ops.append(("mtu", _stream))
        return self._mtu

    def activateStream(self, stream, *args):
        self.ops.append(("activate", stream))

    def deactivateStream(self, stream):
        self.ops.append(("deactivate", stream))
        if self._deactivate_fail is not None and self._deactivate_fail(stream):
            raise RuntimeError(f"deactivate failed for {stream!r}")

    def closeStream(self, stream):
        self.ops.append(("close", stream))

    def writeStream(self, stream, buffs, num, *extra):
        block = np.asarray(buffs[0])
        self.writes.append((int(num), block.dtype, block.size))
        self.ops.append(("write", stream, int(num), block.dtype, len(extra)))
        ret = self._write_ret.pop(0) if self._write_ret else int(num)
        return SimpleNamespace(ret=int(ret))

    def readStreamStatus(self, stream, *args, **kwargs):
        self.ops.append(("status", stream))
        return SimpleNamespace(ret=-1, flags=0)  # TIMEOUT → clean, one bounded check


def _bidir_io(dev, *, hw_rate: float = 96_000.0):
    """A _SoapyBidirIo bound to a fake device, WITHOUT its hardware __init__."""
    io = object.__new__(bidir._SoapyBidirIo)
    io._dev = dev
    io._TX = _TX
    io._CS16 = _CS16
    io._hw_rate = hw_rate
    io._lock = threading.Lock()
    io.tx_active = threading.Event()
    io.rx_suspended = threading.Event()
    io._stop = threading.Event()  # DS-017 teardown signal (rx_chunks checks it each loop)
    io._rx_stream = _RX_STREAM
    io._rx_settle_until = 0.0
    return io


def _cs16(n_complex: int) -> np.ndarray:
    """A FINAL flat CS16 burst [I0,Q0,I1,Q1,...] with distinct values so slicing is checkable."""
    return np.arange(2 * n_complex, dtype=np.int16)


def _no_sleep(monkeypatch):
    """Spy on the shared time module: any sleep during a burst is recorded so 'no sleep between
    activate and first write' is assertable (the path sleeps nowhere, so the list stays empty)."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


def _tx_stream(dev):
    return next(op[1] for op in dev.ops if op[0] == "setup" and op[2] == _TX)


# --------------------------------------------------------------- bidir _SoapyBidirIo


def test_bidir_transmit_opens_cs16_no_channel_list(monkeypatch):
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev(mtu=512)
    io = _bidir_io(dev)

    result = io.transmit_burst(_cs16(1000))

    assert result.complete and result.accepted == 1000
    tx_setups = [a for a in dev.setup_args if a[0] == _TX]
    # exactly one TX stream, CS16 wire format, and NO explicit channel list (extra == ())
    assert tx_setups == [(_TX, _CS16, ())]


def test_bidir_transmit_streams_the_cached_cs16_with_no_dsp(monkeypatch):
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev(mtu=400)
    io = _bidir_io(dev)
    staged = _cs16(1000)

    result = io.transmit_burst(staged)

    assert result.complete
    assert dev.writes, "no write happened"
    # every write is flat CS16 int16, two int16 per complex sample, chunked by write_burst
    for num, dtype, int16_count in dev.writes:
        assert dtype == np.dtype(np.int16)
        assert int16_count == 2 * num
        assert num <= 400  # min(1024, MTU) chunking → write_burst is in use
    assert sum(num for num, _, _ in dev.writes) == 1000  # complex count == cs16.size // 2


def test_bidir_transmit_queries_mtu_before_activate(monkeypatch):
    # (3d) getStreamMTU must be queried BEFORE activateStream, not after.
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev()
    io = _bidir_io(dev)

    io.transmit_burst(_cs16(300))

    tx = _tx_stream(dev)
    mtu_idx = next(i for i, op in enumerate(dev.ops) if op[0] == "mtu")
    activate_idx = next(
        i for i, op in enumerate(dev.ops) if op[0] == "activate" and op[1] == tx
    )
    assert mtu_idx < activate_idx


def test_bidir_transmit_does_not_sleep_between_activate_and_write(monkeypatch):
    slept = _no_sleep(monkeypatch)
    dev = _FakeSdrDev()
    io = _bidir_io(dev)

    io.transmit_burst(_cs16(500))

    assert slept == [], "no sleep is allowed in the keyed write window (stale XTRX buffers)"
    tx = _tx_stream(dev)
    activate_idx = next(
        i for i, op in enumerate(dev.ops) if op[0] == "activate" and op[1] == tx
    )
    write_idx = next(i for i, op in enumerate(dev.ops) if op[0] == "write")
    assert write_idx == activate_idx + 1


def test_bidir_transmit_breaks_rx_first_and_leaves_it_suspended(monkeypatch):
    # HARDWARE-SAFETY EXTENSION: RX deactivated FIRST, TX opened/activated/written, TX
    # torn down — and RX left STOPPED (rx_suspended set). The external T/R switch may
    # still be on TX and the PA energized when the burst returns; only gs-client's
    # resume_rx (after de-key + quiet proof + RX selection + settle) brings RX back.
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev()
    io = _bidir_io(dev)

    io.transmit_burst(_cs16(300))

    assert dev.ops[0] == ("deactivate", _RX_STREAM)
    assert ("activate", _RX_STREAM) not in dev.ops, (
        "the burst self-reactivated RX before transmit_complete/resume_rx"
    )
    assert io.rx_suspended.is_set()
    tx = _tx_stream(dev)
    order = [op[0] for op in dev.ops]
    assert order.index("write") < order.index("close")
    assert ("close", tx) in dev.ops
    # exactly ONE bounded readStreamStatus outcome check (write_burst's), no extra poll
    assert sum(1 for op in dev.ops if op[0] == "status") == 1


def test_resume_rx_reactivates_and_clears_suspension(monkeypatch):
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev()
    io = _bidir_io(dev)
    io.transmit_burst(_cs16(300))
    assert io.rx_suspended.is_set()

    before = time.monotonic()
    io.resume_rx()

    assert dev.ops[-1] == ("activate", _RX_STREAM)
    assert not io.rx_suspended.is_set()
    assert io._rx_settle_until >= before  # reader discards the reactivation transient


def test_resume_rx_failure_propagates_and_keeps_rx_suspended(monkeypatch):
    # A failing resume must RAISE (handler-failed -> nonzero exit -> pass fails); RX
    # stays suspended — there is no best-effort resume.
    _no_sleep(monkeypatch)

    class _RefusingDev(_FakeSdrDev):
        def activateStream(self, stream, *args):
            raise RuntimeError("activate refused")

    dev = _RefusingDev()
    io = _bidir_io(dev)
    io.rx_suspended.set()
    with pytest.raises(RuntimeError, match="activate refused"):
        io.resume_rx()
    assert io.rx_suspended.is_set()


def test_bidir_transmit_refuses_when_rx_deactivate_fails(monkeypatch):
    # (3e) if RX cannot be cleanly broken, TX is REFUSED — never keyed on an un-broken stream.
    # HARDWARE-SAFETY EXTENSION: the ambiguous deactivate no longer self-reactivates —
    # the command arrives with the external switch already on TX, so RX stays SUSPENDED
    # until gs-client's resume_rx handshake.
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev(deactivate_fail=lambda s: s == _RX_STREAM)
    io = _bidir_io(dev)

    result = io.transmit_burst(_cs16(300))

    assert result.outcome == "error"
    assert "RX break failed" in result.detail
    # No TX stream was ever opened / written (TX refused)...
    assert not any(op[0] in ("setup", "write") for op in dev.ops)
    # ...and RX was NOT self-reactivated: it stays suspended for the handshake.
    assert ("deactivate", _RX_STREAM) in dev.ops
    assert ("activate", _RX_STREAM) not in dev.ops
    assert io.rx_suspended.is_set()
    assert not io.tx_active.is_set()


def test_bidir_transmit_cleanup_is_independent_when_tx_deactivate_fails(monkeypatch):
    # (3f) TX deactivate failing must NOT skip TX close; RX stays suspended either way.
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev(
        deactivate_fail=lambda s: isinstance(s, tuple) and len(s) == 3 and s[0] == "stream"
    )
    io = _bidir_io(dev)

    result = io.transmit_burst(_cs16(300))

    tx = _tx_stream(dev)
    assert result.complete  # the write itself succeeded; only teardown deactivate failed
    assert ("deactivate", tx) in dev.ops  # attempted...
    assert ("close", tx) in dev.ops       # ...and close still attempted despite it
    assert ("activate", _RX_STREAM) not in dev.ops
    assert io.rx_suspended.is_set()


def test_bidir_transmit_write_error_still_leaves_rx_suspended(monkeypatch):
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev(write_ret=[-7])  # driver error on the first write
    io = _bidir_io(dev)

    result = io.transmit_burst(_cs16(300))

    assert result.outcome == "error"  # truthful, not a fabricated success
    # a write error must NOT self-reactivate RX either — the handshake owns the resume
    assert ("activate", _RX_STREAM) not in dev.ops
    assert io.rx_suspended.is_set()
    assert not io.tx_active.is_set()


def test_bidir_transmit_empty_output_is_error_and_never_touches_tr(monkeypatch):
    # (3g) empty output is an ERROR (not a silent success) and refuses before any T/R activity.
    _no_sleep(monkeypatch)
    dev = _FakeSdrDev()
    io = _bidir_io(dev)

    result = io.transmit_burst(np.zeros(0, dtype=np.int16))

    assert result.outcome == "error" and result.accepted == 0
    assert dev.ops == []
    assert not io.tx_active.is_set()


# --------------------------------------------------------------- ax25 configure_tx_sink gain


class _GainDev:
    """Fake device recording setGain overloads: named = (dir, ch, name, value),
    overall = (dir, ch, value)."""

    def __init__(self) -> None:
        self.set_gain_calls: list[tuple] = []

    def setAntenna(self, *a): ...
    def setGainMode(self, *a): ...
    def setFrequencyCorrection(self, *a): ...
    def setBandwidth(self, *a): ...

    def setGain(self, *a):
        self.set_gain_calls.append(a)
        if len(a) == 4:  # (dir, ch, name, value) — remember for the readback echo
            self._named = getattr(self, "_named", {})
            self._named[(a[0], a[1], a[2])] = float(a[3])

    def getGain(self, direction, channel, name):
        # TX-CHAIN EXTENSION: faithful echo of the last named set, unless the test
        # scripts a clamp/failure via `gain_readback`.
        rb = getattr(self, "gain_readback", None)
        if rb is not None:
            if isinstance(rb, Exception):
                raise rb
            return rb
        return getattr(self, "_named", {})[(direction, channel, name)]


def test_configure_tx_sink_uses_named_pad_not_overall_setgain(monkeypatch):
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()

    txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": 40.0}}, 2_000_000.0)

    # the named PAD element was set...
    named = [a for a in dev.set_gain_calls if len(a) == 4 and a[2] == "PAD"]
    assert named == [(_TX, 0, "PAD", 40.0)]
    # ...and the overall (value-only) setGain overload was NEVER called (XTRX-unsafe).
    overall = [a for a in dev.set_gain_calls if len(a) == 3]
    assert overall == []


def test_configure_tx_sink_without_named_pad_refuses(monkeypatch):
    # (3c) no named per-element gain is a CONFIGURATION error — refuse, do not fall back to
    # the overall setGain overload or transmit deaf.
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()

    with pytest.raises(TxGainConfigError):
        txapp.configure_tx_sink(dev, _TX, {}, 2_000_000.0)
    assert dev.set_gain_calls == []  # nothing was configured before the refusal


def test_configure_tx_sink_refuses_a_clamped_pad_readback(monkeypatch):
    # TX-CHAIN EXTENSION: the driver clamping the requested PAD (readback != request)
    # must fail the spawn BEFORE ready/key — never radiate at an unintended drive.
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()
    dev.gain_readback = -12.0  # driver clamped the requested -40 to -12

    with pytest.raises(TxGainConfigError, match="does not match"):
        txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": -40.0}}, 2_000_000.0)


def test_configure_tx_sink_refuses_an_unreadable_pad_readback(monkeypatch):
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()
    dev.gain_readback = RuntimeError("getGain not supported")

    with pytest.raises(TxGainConfigError, match="readback FAILED"):
        txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": -40.0}}, 2_000_000.0)


def test_configure_tx_sink_refuses_a_non_finite_pad_readback(monkeypatch):
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()
    dev.gain_readback = float("nan")

    with pytest.raises(TxGainConfigError, match="non-finite"):
        txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": -40.0}}, 2_000_000.0)


def test_verified_pad_readback_passes_within_tolerance(monkeypatch):
    # A faithful (or ≤0.5 dB quantized) readback is accepted.
    monkeypatch.delenv("GS_SDR_TX_GAINS", raising=False)
    monkeypatch.delenv("GS_SDR_TX_GAIN_DB", raising=False)
    dev = _GainDev()

    applied = txapp.configure_tx_sink(dev, _TX, {"sdr_tx_gains": {"PAD": -40.0}}, 2_000_000.0)
    assert applied  # configured and verified without raising


# ------------------------------------------------- ax25 pre-key CS16 builder (3a / 3g)


def test_ax25_prepare_tx_cs16_returns_none_for_file_sink():
    # (3a) the file/bench sink writes modem-rate cf32 in _sink_iq — no pre-packed CS16.
    args = SimpleNamespace(sdr_args="file:/tmp/uplink.cf32", sample_rate=96_000)
    assert txapp._prepare_tx_cs16(args, np.ones(64, dtype=np.complex64)) is None


def test_ax25_prepare_tx_cs16_builds_final_cs16_for_hardware(monkeypatch):
    # (3a) the hardware sink gets the FINAL flat CS16 built PRE-KEY.
    monkeypatch.setattr(
        txapp, "sdr_env",
        lambda: {"capture_rate_hz": 2_048_000.0, "ppm": 0.0, "dc_removal": False},
    )
    args = SimpleNamespace(sdr_args="driver=xtrx", sample_rate=96_000)
    out = txapp._prepare_tx_cs16(args, np.ones(100, dtype=np.complex64))
    assert out is not None and out.dtype == np.int16 and out.size % 2 == 0


def test_ax25_prepare_tx_cs16_rejects_empty_and_non_finite(monkeypatch):
    # (3g) an empty or non-finite hardware-rate waveform is refused PRE-KEY (fails the spawn).
    monkeypatch.setattr(
        txapp, "sdr_env",
        lambda: {"capture_rate_hz": 2_048_000.0, "ppm": 0.0, "dc_removal": False},
    )
    args = SimpleNamespace(sdr_args="driver=xtrx", sample_rate=96_000)
    with pytest.raises(ValueError):
        txapp._prepare_tx_cs16(args, np.zeros(0, dtype=np.complex64))
    with pytest.raises(ValueError):
        txapp._prepare_tx_cs16(args, np.full(100, np.nan + 0j, dtype=np.complex64))


# --------------------------------------------------------------- SWEEP-1 (#5) rx_chunks fail-closed


class _ScriptedRxDev:
    """readStream returns a scripted sequence of ret codes, then raises _RxScriptEnd once the
    script is exhausted — so a NON-raising rx_chunks generator terminates the test deterministically
    instead of spinning forever."""

    def __init__(self, rets: list[int]) -> None:
        self._rets = list(rets)

    def readStream(self, _stream, _buffs, _num, timeoutUs: int = 0):  # noqa: N803 (Soapy kwarg name)
        del timeoutUs
        if not self._rets:
            raise _RxScriptEnd
        return SimpleNamespace(ret=self._rets.pop(0))


class _RxScriptEnd(Exception):
    pass


def test_rx_chunks_raises_engine_failure_on_persistent_readstream_error(monkeypatch, fake_soapysdr):
    """SWEEP-1 (#5): a readStream returning a hard error code (STREAM_ERROR) on every read is a
    dead/unplugged radio; rx_chunks must RAISE EngineFailure (so run_rx fails the pass) after the
    consecutive-error bound, instead of warn-and-continue spinning forever (deaf-RX hang)."""
    fake_soapysdr.SOAPY_SDR_OVERFLOW = -4  # rx_chunks imports OVERFLOW + TIMEOUT
    monkeypatch.setattr(bidir, "_MAX_CONSECUTIVE_RX_ERRORS", 5)
    err = fake_soapysdr.SOAPY_SDR_STREAM_ERROR  # -2
    io = _bidir_io(_ScriptedRxDev([err] * 20))
    with pytest.raises(bidir.EngineFailure, match="dead/unplugged"):
        for _ in io.rx_chunks():
            pass


def test_rx_chunks_error_count_resets_on_timeout(monkeypatch, fake_soapysdr):
    """SWEEP-1 (#5): a TIMEOUT (or overflow) between errors resets the consecutive-error counter, so
    an isolated blip never fails the pass. 4 errors + timeout + 4 errors (bound=5) must NOT raise
    EngineFailure — the script exhausts and we hit the sentinel instead. (Deaf window huge so the
    SWEEP-2 time deadline does not fire either.)"""
    fake_soapysdr.SOAPY_SDR_OVERFLOW = -4
    monkeypatch.setattr(bidir, "_MAX_CONSECUTIVE_RX_ERRORS", 5)
    monkeypatch.setattr(bidir, "_RX_DEAF_TIMEOUT_S", 1e9)
    err = fake_soapysdr.SOAPY_SDR_STREAM_ERROR  # -2
    tmo = fake_soapysdr.SOAPY_SDR_TIMEOUT  # -1
    io = _bidir_io(_ScriptedRxDev([err, err, err, err, tmo, err, err, err, err]))
    with pytest.raises(_RxScriptEnd):  # exhausted WITHOUT an EngineFailure
        for _ in io.rx_chunks():
            pass


def test_rx_chunks_raises_on_persistent_timeout_deaf_radio(monkeypatch, fake_soapysdr):
    """SWEEP-2 (gap#1): the far more common deaf-radio mode returns SOAPY_SDR_TIMEOUT forever
    (stalled RX DMA / dead LNA), which RESETS the hard-error counter every read — so the SWEEP-1
    backstop never fired. A time-based no-progress deadline must fail the pass when NOTHING is
    delivered for the deaf window. With a negative window, the first timeout past t0 raises."""
    fake_soapysdr.SOAPY_SDR_OVERFLOW = -4
    # Negative window so the first timeout past t0 trips deterministically (a fast test loop may not
    # advance the coarse monotonic clock past 0 within a few iterations).
    monkeypatch.setattr(bidir, "_RX_DEAF_TIMEOUT_S", -1.0)
    tmo = fake_soapysdr.SOAPY_SDR_TIMEOUT  # -1
    io = _bidir_io(_ScriptedRxDev([tmo] * 50))
    with pytest.raises(bidir.EngineFailure, match="deaf/stalled"):
        for _ in io.rx_chunks():
            pass


def test_rx_chunks_parks_while_rx_is_suspended(monkeypatch, fake_soapysdr):
    """HARDWARE-SAFETY EXTENSION: while rx_suspended is set (post-burst, awaiting the
    resume_rx handshake) the reader must PARK — no readStream calls, and the deaf
    deadline must not fire (the pause is intentional, not a dead radio)."""
    fake_soapysdr.SOAPY_SDR_OVERFLOW = -4
    monkeypatch.setattr(bidir, "_RX_DEAF_TIMEOUT_S", -1.0)  # any read would trip instantly

    class _Park(Exception):
        pass

    calls = {"n": 0}

    def _sleep(_s):
        calls["n"] += 1
        if calls["n"] >= 5:
            raise _Park

    monkeypatch.setattr(bidir.time, "sleep", _sleep)

    class _MustNotRead:
        def readStream(self, *_a, **_k):
            raise AssertionError("readStream called while RX was suspended")

    io = _bidir_io(_MustNotRead())
    io.rx_suspended.set()
    with pytest.raises(_Park):  # parked through 5 sleep ticks without a single read
        for _ in io.rx_chunks():
            pass


# --------------------------------------------------------------------------- DS-018 (Phase 2C)


def test_close_runs_each_cleanup_step_independently():
    """DS-018: one suppress wrapped BOTH deactivateStream and closeStream, so a raising deactivate
    SKIPPED the close and leaked the stream handle. Each step must be attempted independently."""
    calls: list[str] = []

    class _Dev:
        def deactivateStream(self, _s):
            calls.append("deactivate")
            msg = "device wedged"
            raise RuntimeError(msg)

        def closeStream(self, _s):
            calls.append("close")

    io = _bidir_io(_Dev())
    io.close()
    assert calls == ["deactivate", "close"], "closeStream was skipped after deactivate raised"
    assert io._stop.is_set()  # close() also signals the reader (DS-017)

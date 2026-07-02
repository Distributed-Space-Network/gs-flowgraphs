"""Round-2 adversarial-review regressions (docs/J follow-up batch).

HIGH-1  carry-fragment crash: a burst ending 1..75 samples before a drain
        boundary left a >=2 ms ON fragment in the sub-frame carry; it re-gated
        as a "burst" whose segment was SHORTER than the demod's 64-symbol
        moving-mean kernel, np.convolve('same') broadcast-failed, the exception
        escaped asyncio.to_thread and gather(return_exceptions=True) swallowed
        it — live decode silently dead for the rest of the pass.
HIGH-2  window-local percentile floor: a continuous packet train filling >90 %
        of a window pushed the 10th percentile to signal level (constant GFSK
        envelope), the gate rose to 4x signal, and the whole train was silently
        discarded (deferral concentrated it into the NEXT window, making it
        worse). Fixed by a persistent noise-floor estimate a dense window can
        never raise. MED-2 rode along: the ``mag.max()*0.08`` relative term
        masked any burst weaker than 1/12.5 of the strongest in its window.
MED-1   ax25 tail-SUBTRACT dedup assumed the demod is suffix-local; it is not
        (capture-global RMS + centered moving mean), producing ~2-3 % duplicate
        frames under mixed-amplitude chunking. Replaced with positional dedup
        (payload + absolute sample position).
LOW-2   plan/engine race divergence for the absent-framing case.
LOW-3   is covered in test_soapy.py (gains precedence).
LOW-4   GR engine final drain at stop (static lock, GR not importable here).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from pathlib import Path

import compose
import cubesat_gfsk_ax25_rx as rxapp
import framings
import numpy as np

from gfsk_ax25 import ax25, endurosat, gfsk
from gfsk_ax25 import endurosat_link as el

_APPS = Path(__file__).resolve().parent.parent / "apps"
_TX_SR = 153_600.0  # 16 samples/symbol at 9600 (endurosat chip link)
_AX_SR = 99_840.0  # 8 samples/symbol at 12480 (AX.25 profile)


# ----------------------------------------------------------------------
# HIGH-1(a): segments shorter than the demod kernel must yield no bits, never raise
# ----------------------------------------------------------------------


def test_demodulate_input_shorter_than_kernel_yields_no_bits():
    params = gfsk.GfskParams(sample_rate_hz=_TX_SR, symbol_rate_hz=9600.0)
    kernel = int(round(params.sps * 64))
    for n in (2, 100, kernel - 1, kernel):  # all previously raised ValueError
        assert gfsk.demodulate(np.ones(n, np.complex64), params).size == 0
    # At/above the kernel the normal path runs (no exception).
    gfsk.demodulate(np.ones(kernel + 64, np.complex64), params)


def test_carry_fragment_near_boundary_never_crashes_and_decodes_once():
    # The exact reproduced crash zone: trailing quiet k=1..75 leaves a >=307-
    # sample ON fragment in the 383-sample carry; the next drain re-gated it as
    # a burst whose segment (~767-842 samples) was shorter than the 1024-sample
    # kernel -> ValueError killed decode_new. Now: no crash, and the frame is
    # emitted exactly once (from its own window — the fragment yields nothing).
    payload = b"edge-tail-frame"
    burst = el.transmit(payload, _TX_SR)
    for k in range(1, 76):
        dec = el.StreamDecoder(_TX_SR)
        dec.push(
            np.concatenate(
                [np.zeros(4000, np.complex64), burst, np.zeros(k, np.complex64)]
            ).astype(np.complex64)
        )
        out = dec.decode_new()
        dec.push(np.zeros(6000, np.complex64))
        out += dec.decode_new()  # pre-fix: ValueError escaped here
        out += dec.flush()
        assert out == [payload], f"k={k}: {out!r}"


# ----------------------------------------------------------------------
# HIGH-1(b): the RX app's decode loop must survive (and log) a decoder exception
# ----------------------------------------------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf += data

    async def drain(self) -> None:
        return None


class _FakeSockets:
    def __init__(self) -> None:
        self.status_writer = _FakeWriter()
        self.data_writer = _FakeWriter()


class _RaiseOnceDecoder:
    """Delegates to a real decoder but raises on the FIRST decode_new call —
    before consuming the buffered chunks, so nothing is lost if the loop
    survives (which is exactly what the fix must guarantee)."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.raised = False

    def push(self, chunk) -> None:
        self._inner.push(chunk)

    def decode_new(self):
        if not self.raised:
            self.raised = True
            msg = "injected decoder failure"
            raise ValueError(msg)
        return self._inner.decode_new()

    def flush(self):
        return self._inner.flush()


class _FlushRaisesDecoder:
    def __init__(self, inner) -> None:
        self._inner = inner

    def push(self, chunk) -> None:
        self._inner.push(chunk)

    def decode_new(self):
        return self._inner.decode_new()

    def flush(self):
        msg = "injected flush failure"
        raise ValueError(msg)


def _drive_engine(monkeypatch, wrapper, chunks, *, pause_s: float) -> tuple[list, bytes]:
    """Run the dsp engine with an endurosat decoder wrapped by ``wrapper`` and a
    source that yields ``chunks`` with pauses (so the decode loop gets ticks)."""
    import argparse

    real = el.StreamDecoder

    def wrapped(*a, **kw):
        return wrapper(real(*a, **kw))

    monkeypatch.setattr(rxapp.endurosat_link, "StreamDecoder", wrapped)
    monkeypatch.setattr(rxapp, "_DECODE_PERIOD_S", 0.05)

    def fake_source(args, params=None):
        for c in chunks:
            yield c
            time.sleep(pause_s)

    monkeypatch.setattr(rxapp, "_open_iq_source", fake_source)
    args = argparse.Namespace(sample_rate=_TX_SR, sdr_args="", center_freq_hz=401_500_000)
    socks = _FakeSockets()
    started = asyncio.Event()
    started.set()
    stop = asyncio.Event()
    profile = rxapp._profile_from_params({})
    asyncio.run(
        rxapp._run_dsp_engine(
            args, socks, {"framing": "endurosat"}, started, stop, profile, {"hz": 0.0}
        )
    )
    import json

    events = [
        json.loads(line) for line in socks.status_writer.buf.decode().splitlines() if line.strip()
    ]
    return events, bytes(socks.data_writer.buf)


def _burst(payload: bytes) -> np.ndarray:
    return np.concatenate(
        [np.zeros(2000, np.complex64), el.transmit(payload, _TX_SR), np.zeros(2000, np.complex64)]
    ).astype(np.complex64)


def test_decode_loop_survives_decoder_exception_and_logs(monkeypatch, caplog):
    # Pre-fix: the first decode_new exception ended the decode task; gather(...,
    # return_exceptions=True) swallowed it with NO log and every later frame of
    # the pass was lost. Now the loop logs and keeps decoding: both frames
    # (pushed before and after the failure) must still be emitted.
    p1, p2 = b"before-failure", b"after-failure!"
    with caplog.at_level(logging.ERROR, logger="cubesat_gfsk_ax25_rx"):
        events, data = _drive_engine(
            monkeypatch, _RaiseOnceDecoder, [_burst(p1), _burst(p2)], pause_s=0.3
        )
    assert any("decode_new failed" in r.getMessage() for r in caplog.records)
    got = [e for e in events if e["event"] == "frame_received"]
    assert len(got) == 2 and data == p1 + p2  # loop survived; no frame lost


def test_final_flush_failure_is_logged_not_swallowed(monkeypatch, caplog):
    p1 = b"flushed-window"
    with caplog.at_level(logging.ERROR, logger="cubesat_gfsk_ax25_rx"):
        events, _ = _drive_engine(monkeypatch, _FlushRaisesDecoder, [_burst(p1)], pause_s=0.3)
    # The engine finished cleanly (ready event emitted, no exception escaped)
    # and the flush failure is VISIBLE in the log.
    assert any(e["event"] == "ready" for e in events)
    assert any("final flush decode failed" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------
# HIGH-2 (+ MED-2): persistent noise floor vs dense windows / masked weak bursts
# ----------------------------------------------------------------------


def _train(n_frames: int = 8) -> tuple[list[bytes], np.ndarray]:
    payloads = [bytes([65 + i]) * 100 for i in range(n_frames)]
    return payloads, np.concatenate([el.transmit(p, _TX_SR) for p in payloads])


def test_dense_packet_train_decodes_fully_across_small_drains():
    # The reproduced HIGH-2 collapse: 8 back-to-back frames fill >90 % of every
    # 50 k-sample window; the window-local 10th percentile hit signal level and
    # 1/8 frames survived (the AirMAC bulk-download profile). Now 8/8, once.
    payloads, train = _train()
    iq = np.concatenate(
        [np.zeros(3000, np.complex64), train, np.zeros(30_000, np.complex64)]
    ).astype(np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    out: list[bytes] = []
    for i in range(0, len(iq), 50_000):
        dec.push(iq[i : i + 50_000])
        out += dec.decode_new()
    out += dec.flush()
    assert sorted(out) == sorted(payloads)  # all 8, exactly once


def test_weak_burst_deferred_across_boundary_not_masked_by_strong():
    # MED-2 / probe T1: a weak burst deferred across the drain boundary lands in
    # the same window as a 12.5x+ stronger one; the old mag.max()*0.08 gate term
    # masked it. Both frames must decode.
    for amp in (0.02, 0.06):
        weak, strong = b"weak-deferred", b"strong-later"
        wb = (amp * el.transmit(weak, _TX_SR)).astype(np.complex64)
        sb = el.transmit(strong, _TX_SR).astype(np.complex64)
        quiet = np.zeros(4000, np.complex64)
        half = len(wb) // 2
        dec = el.StreamDecoder(_TX_SR)
        dec.push(np.concatenate([quiet, wb[:half]]))
        out = dec.decode_new()
        dec.push(np.concatenate([wb[half:], quiet, sb, quiet]))
        out += dec.decode_new() + dec.flush()
        assert sorted(out) == sorted([weak, strong]), f"amp={amp}: {out!r}"


def test_weak_and_strong_burst_in_one_drain_both_decode():
    # MED-2 / probe T2: same masking WITHIN a single window.
    weak, strong = b"weak-same-drain", b"strong-same"
    quiet = np.zeros(4000, np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    dec.push(
        np.concatenate(
            [quiet, 0.02 * el.transmit(weak, _TX_SR), quiet, el.transmit(strong, _TX_SR), quiet]
        ).astype(np.complex64)
    )
    out = dec.decode_new() + dec.flush()
    assert sorted(out) == sorted([weak, strong])


def test_noise_floor_is_never_raised_by_a_dense_window():
    # The structural HIGH-2 property at the unit level: once seeded from noise, a
    # window whose quietest block is SIGNAL (a wall-to-wall packet train) must
    # leave the floor untouched — never pulled up toward signal level (which is
    # what collapsed the gate and discarded the train). Fed directly so no quiet
    # carry from a previous window can contaminate the reading.
    rng = np.random.default_rng(3)
    noise_mag = np.abs(
        (0.01 * (rng.standard_normal(30_000) + 1j * rng.standard_normal(30_000))).astype(
            np.complex64
        )
    )
    dec = el.StreamDecoder(_TX_SR)
    dec._update_floor(noise_mag)
    seeded = dec._noise_floor
    assert seeded is not None and 0 < seeded < 0.03  # seeded at noise level
    signal_mag = np.abs(_train(4)[1].astype(np.complex64))  # constant ~1.0 envelope
    thr = dec._update_floor(signal_mag)
    assert dec._noise_floor == seeded  # dense window must NOT raise the floor
    assert thr == seeded * 4.0  # gate stays at 4x the NOISE floor, not signal


def test_pass_starting_mid_transmission_defers_then_decodes():
    # Pathological start: signal from sample 0, no noise reference to seed the
    # floor. The decoder defers un-gated and decodes the carried train once the
    # first quiet gap seeds the floor — nothing is discarded.
    payloads, train = _train()
    iq = np.concatenate([train, np.zeros(30_000, np.complex64)]).astype(np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    out: list[bytes] = []
    for i in range(0, len(iq), 50_000):
        dec.push(iq[i : i + 50_000])
        out += dec.decode_new()
    out += dec.flush()
    assert sorted(out) == sorted(payloads)


def test_all_signal_capture_flushes_as_one_burst():
    # No quiet EVER (flush before any gap): with no floor the window is decoded
    # as ONE burst at flush — recovering what the whole-capture demod recovers,
    # NOT silently discarded against a signal-level gate (the HIGH-2 failure).
    _, train = _train(3)
    train = train.astype(np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    dec.push(train)
    assert dec.decode_new() == []  # deferred un-gated (no floor yet)
    out = dec.flush()
    assert out  # decoded as one burst, not discarded
    assert out == el.receive(train, _TX_SR)  # == the best a single-blob demod does


# ----------------------------------------------------------------------
# MED-1: ax25 StreamDecoder positional dedup (payload + absolute position)
# ----------------------------------------------------------------------


def _ui(info: bytes) -> bytes:
    return ax25.encode_ui(dest="DSN0", src="ES1", info=info)


def _cn(rng, n: int, amp: float) -> np.ndarray:
    """Complex Gaussian noise, ``amp`` per quadrature."""
    return (amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def _ax25_stream(drains) -> list[bytes]:
    dec = endurosat.StreamDecoder(_AX_SR)
    out: list[bytes] = []
    for d in drains:
        dec.push(d.astype(np.complex64))
        out += dec.decode_new()
    return out + dec.flush()


def test_ax25_no_duplicates_on_the_reproduced_dup_seeds():
    # Exact reproduction of the reviewer's A5 dup seeds (1, 5, 21): the OLD
    # tail-subtract dedup re-emitted an already-emitted frame when the tail-
    # alone re-decode (different capture-global RMS/moving-mean context) missed
    # it. Positional dedup must emit each frame exactly once — no dup, no loss.
    for seed in (1, 5, 21):
        rng = np.random.default_rng(1000 + seed)
        bodies = [_ui(f"pkt-{seed}-{i}".encode()) for i in range(5)]
        parts = []
        namp = 0.01
        for b in bodies:
            a = float(rng.uniform(0.05, 1.0))
            s = (a * endurosat.transmit(b, _AX_SR)).astype(np.complex64)
            sn = s + _cn(rng, len(s), namp)
            # Two extra draws mirror the reviewer's probe rng stream exactly.
            _ = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
            _ = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
            parts.append(sn)
            g = int(rng.integers(1500, 8000))
            parts.append(_cn(rng, g, namp))
        iq = np.concatenate(parts)
        cuts = sorted(rng.integers(1, len(iq), rng.integers(3, 14)).tolist())
        cuts = [0, *cuts, len(iq)]
        out = _ax25_stream([iq[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)])
        assert Counter(out) == Counter(bodies), f"seed={seed}"


def test_ax25_repeat_beacon_across_boundary_emits_per_instance():
    # docs/10 section 7 invariant under the NEW dedup: an identical beacon whose
    # first copy still sits in the carried tail when the second arrives is TWO
    # frames at two positions — both emit, neither twice. (gap=500 is excluded:
    # the whole-capture demod itself loses one copy there — a demod-level
    # limitation shared by every decoder generation, not a dedup effect.)
    body = _ui(b"repeat-beacon")
    sig = endurosat.transmit(body, _AX_SR)
    for gap in (250, 750, 1500, 3000):
        d1 = np.concatenate([np.zeros(1500, np.complex64), sig, np.zeros(gap, np.complex64)])
        d2 = np.concatenate([sig, np.zeros(2500, np.complex64)])
        out = _ax25_stream([d1, d2])
        assert out == [body, body], f"gap={gap}: {len(out)} copies"


def test_ax25_emitted_frame_parked_in_tail_never_re_emits():
    # A frame that stays inside the carried tail across many small noise drains
    # re-decodes at the SAME absolute position every time -> dropped every time.
    rng = np.random.default_rng(11)
    body = _ui(b"parked-in-tail")
    dec = endurosat.StreamDecoder(_AX_SR)
    dec.push(
        np.concatenate(
            [endurosat.transmit(body, _AX_SR), np.zeros(2000, np.complex64)]
        ).astype(np.complex64)
    )
    assert dec.decode_new() == [body]
    for _ in range(12):
        dec.push(_cn(rng, 3000, 0.02))
        assert dec.decode_new() == []
    assert dec.flush() == []
    # Bounded memory: the position registry pruned to what the tail can re-yield.
    assert len(dec._emitted) <= 4


def test_ax25_fuzz_no_dups_and_no_loss_beyond_whole_capture():
    # Fresh seeds (distinct from the probes): random frames, 5x amplitude
    # spread, noise, random drain cuts. Positional-dedup invariants: (a) NEVER a
    # duplicate (emitted multiplicity <= sent), and (b) NO loss beyond the
    # whole-capture decode's own demod-level losses — the incremental decoder
    # must not throw away anything the reference recovers. (The demod itself
    # loses some low-amplitude frames; that is not a dedup regression, so the
    # reference bounds the acceptable loss.)
    namp = 0.01
    for seed in range(12):
        rng = np.random.default_rng(7000 + seed)
        bodies = [_ui(f"fz-{seed}-{i}".encode()) for i in range(5)]
        if seed % 2 == 0:
            bodies.append(bodies[0])  # genuine repeat beacon
        parts = []
        for b in bodies:
            a = float(rng.uniform(0.2, 1.0))
            s = (a * endurosat.transmit(b, _AX_SR)).astype(np.complex64)
            parts.append(s + _cn(rng, len(s), namp))
            g = int(rng.integers(1500, 8000))
            parts.append(_cn(rng, g, namp))
        iq = np.concatenate(parts)
        cuts = sorted(rng.integers(1, len(iq), rng.integers(3, 14)).tolist())
        cuts = [0, *cuts, len(iq)]
        out = _ax25_stream([iq[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)])
        sent = Counter(bodies)
        got = Counter(out)
        ref = Counter(endurosat.receive(iq, _AX_SR))  # whole-capture reference
        for k in sent:
            assert got[k] <= sent[k], f"seed={seed} dup of {k!r}"  # (a) no dup
            stream_loss = sent[k] - got[k]
            whole_loss = max(0, sent[k] - ref[k])
            assert stream_loss <= whole_loss, f"seed={seed} extra loss of {k!r}"  # (b)


# ----------------------------------------------------------------------
# LOW-2: absent-framing plan agrees with the engine's autodetecting fallbacks
# ----------------------------------------------------------------------


def test_absent_framing_plan_reports_winnable_race():
    # The engine builds modulation fallbacks with framing=None and
    # framings.deframe AUTODETECTS (CRC-gated set) — a CRC hit wins the race via
    # race_winner. The plan must say the same, from the same registry facts.
    plan = compose.plan_decode({"modulation": "gfsk", "symbol_rate_hz": 9600}, catalogued=True)
    assert plan.our_engine and plan.race and plan.race_ours_can_win
    auto = framings.autodetect_framings()
    assert auto and all(framings.is_crc_gated(f) for f in auto)  # the premise
    # and the winner logic agrees for any autodetected framing:
    for f in auto:
        assert compose.race_winner([f], grsat_produced=True) == "ours"


def test_absent_framing_plan_without_grsat_is_our_engine():
    plan = compose.plan_decode({"modulation": "gfsk", "symbol_rate_hz": 9600}, catalogued=False)
    assert plan.our_engine and plan.decodable
    assert not plan.grsatellites and not plan.race  # nothing to race against


def test_absent_framing_and_unknown_modulation_still_not_decodable():
    # Autodetect needs OUR demod to produce bits: no modem -> still record-only.
    plan = compose.plan_decode({"modulation": "smoke", "symbol_rate_hz": 1200}, catalogued=False)
    assert not plan.our_modem and not plan.decodable


# ----------------------------------------------------------------------
# LOW-4: GR engine final drain at stop (GNU Radio not importable — static lock)
# ----------------------------------------------------------------------


def test_gnuradio_engine_has_final_drain_at_stop():
    # The dsp engine flushes its decoder at stop; the GR engine must equally
    # drain the sink once more AFTER ctx.stop()/wait() (frames from the last
    # <=2 s of the pass — the LOS end) and emit through the same dedup path.
    src = (_APPS / "cubesat_gfsk_ax25_rx.py").read_text(encoding="utf-8")
    fn = src[src.index("async def _run_gnuradio_engine") : src.index("async def amain")]
    fin = fn[fn.rindex("finally:") :]
    assert "ctx.stop()" in fin and "ctx.drain_bits()" in fin and "_emit_frame" in fin
    assert fin.index("ctx.stop()") < fin.index("ctx.drain_bits()")  # drain the flushed graph
    assert "_decode(tail)" in fin  # same tail-carry dedup as the loop body

"""Regression tests for the docs/10 full-system-review fixes in the stream decoders
and the cubesat RX app (HIGH-1 frame loss, MED-3 incremental decode + bounded RAM +
off-loop decode, LOW-3 framing normalization, LOW-7 data_format label).

HIGH-1 background: the old StreamDecoders re-decoded the WHOLE growing capture each
call and deduped by slicing past an emitted COUNT, assuming the frame list was
prefix-stable. It was not: in endurosat_link the burst gate threshold included
``mag.max()*0.08`` over the whole capture, so a later strong burst re-baselined the
list (an earlier weak burst fell below the gate) while the count did not — one real
frame was silently lost forever, including at flush(). A rising-SNR
horizon->culmination pass IS that profile, on the DEFAULT engine for
framing=endurosat.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import cubesat_gfsk_ax25_rx as rxapp
import numpy as np

from gfsk_ax25 import ax25, endurosat
from gfsk_ax25 import endurosat_link as el

_TX_SR = 153_600.0  # 16 samples/symbol at 9600 (endurosat chip link)
_AX_SR = 99_840.0  # 8 samples/symbol at 12480 (AX.25 profile)


# ----------------------------------------------------------------------
# HIGH-1 — endurosat_link.StreamDecoder (the production EnduroSat dsp path)
# ----------------------------------------------------------------------


def test_rising_snr_never_loses_earlier_weak_frames():
    # EXACT reproduced loss scenario: a weak burst decoded in drain 1, then two
    # strong bursts in drain 2. Old behavior: the whole-capture threshold rose to
    # max*0.08, the weak burst fell out of the (re-baselined) frame list while
    # ``_emitted`` stayed at 1, and only ONE of the two strong frames was emitted
    # — the other was lost forever, including at flush(). Zero loss required.
    weak = b"weak-early-frame"
    strong_a = b"strong-frame-A"
    strong_b = b"strong-frame-B"
    quiet = np.zeros(4000, np.complex64)
    drain1 = np.concatenate([quiet, 0.05 * el.transmit(weak, _TX_SR), quiet]).astype(np.complex64)
    drain2 = np.concatenate(
        [quiet, el.transmit(strong_a, _TX_SR), quiet, el.transmit(strong_b, _TX_SR), quiet]
    ).astype(np.complex64)

    dec = el.StreamDecoder(_TX_SR)
    dec.push(drain1)
    first = dec.decode_new()
    assert first == [weak]  # the weak burst was visible (emitted) on its own drain
    dec.push(drain2)
    out = first + dec.decode_new() + dec.flush()
    assert sorted(out) == sorted([weak, strong_a, strong_b])  # zero loss
    assert len(out) == len(set(out)) == 3  # and no duplicates


def test_repeat_beacons_at_different_positions_each_emit():
    # docs/10 section 7 positional-dedup semantics: identical payloads in
    # different bursts are DISTINCT frames — a payload-set dedup must never
    # suppress them, and none may be emitted twice for one burst.
    beacon = b"identical-beacon-payload"
    quiet = np.zeros(4000, np.complex64)
    burst = el.transmit(beacon, _TX_SR)
    dec = el.StreamDecoder(_TX_SR)
    dec.push(np.concatenate([quiet, burst, quiet]).astype(np.complex64))
    out = dec.decode_new()
    dec.push(np.concatenate([burst, quiet, burst, quiet]).astype(np.complex64))
    out += dec.decode_new()
    out += dec.flush()
    assert out == [beacon] * 3


def test_burst_straddling_drain_boundary_decodes_exactly_once():
    # A burst cut mid-air by the drain boundary is deferred (never decoded
    # truncated) and decoded whole, once, when the rest arrives.
    payload = b"straddler"
    quiet = np.zeros(3000, np.complex64)
    burst = el.transmit(payload, _TX_SR)
    iq = np.concatenate([quiet, burst, quiet]).astype(np.complex64)
    cutpt = 3000 + len(burst) // 2  # split mid-burst
    dec = el.StreamDecoder(_TX_SR)
    dec.push(iq[:cutpt])
    out = dec.decode_new()
    assert out == []  # deferred, not decoded truncated
    dec.push(iq[cutpt:])
    out += dec.decode_new()
    out += dec.flush()
    assert out == [payload]


def test_endurosat_retained_iq_stays_bounded():
    # MED-3: the decoder must not retain the whole pass. Push many drains of
    # quiet + bursts; after every decode_new the carried buffer stays tiny
    # (sub-frame carry), never the accumulated capture.
    payload = b"bounded-memory-frame"
    quiet = np.zeros(5000, np.complex64)
    drain = np.concatenate([quiet, el.transmit(payload, _TX_SR), quiet]).astype(np.complex64)
    dec = el.StreamDecoder(_TX_SR)
    out: list[bytes] = []
    for _ in range(20):
        dec.push(drain.copy())
        out += dec.decode_new()
        assert len(dec._pending) <= dec._max_defer + dec._carry + 1
        assert len(dec._pending) < len(drain)  # NOT accumulating the capture
    out += dec.flush()
    assert out == [payload] * 20  # every burst emitted exactly once


def test_continuous_carrier_forced_cut_bounds_the_buffer():
    # A continuous above-gate region longer than the defer ceiling must be
    # force-decoded rather than carried forever (RAM bound on interference).
    dec = el.StreamDecoder(_TX_SR)
    dec._max_defer = 20_000  # tighten the ceiling for the test
    quiet = np.zeros(6000, np.complex64)
    tone = np.ones(40_000, np.complex64)  # ON region far beyond the ceiling
    dec.push(np.concatenate([quiet, tone]).astype(np.complex64))
    assert dec.decode_new() == []  # a bare carrier carries no valid frame
    assert len(dec._pending) <= dec._carry + 1


# ----------------------------------------------------------------------
# HIGH-1 sibling + MED-3(a) — endurosat.StreamDecoder (dsp ax25 backup engine)
# ----------------------------------------------------------------------


def _ui(info: bytes) -> bytes:
    return ax25.encode_ui(dest="DSN0", src="ES1", info=info)


def test_ax25_incremental_decode_matches_whole_capture_and_bounds_memory():
    # (i) Feeding many sequential drains must yield exactly what the old
    # whole-capture decode yields on the same input — including a genuine repeat
    # beacon (same payload, different position), which must emit BOTH times.
    # (ii) The retained buffer stays bounded (tail carry, not the whole pass).
    bodies = [_ui(f"pkt-{i}".encode()) for i in range(4)]
    bodies.append(bodies[0])  # repeat beacon
    gap = np.zeros(3000, np.complex64)
    parts: list[np.ndarray] = []
    for b in bodies:
        parts += [endurosat.transmit(b, _AX_SR), gap]
    iq = np.concatenate(parts).astype(np.complex64)

    reference = endurosat.receive(iq, _AX_SR)  # the old whole-capture decode
    assert sorted(reference) == sorted(bodies)  # sanity: all frames decodable

    dec = endurosat.StreamDecoder(_AX_SR)
    out: list[bytes] = []
    step = 8192  # many small sequential drains
    for i in range(0, len(iq), step):
        dec.push(iq[i : i + step])
        out += dec.decode_new()
        assert len(dec._tail) <= dec._tail_max  # (ii) retained IQ bounded
        assert not dec._chunks  # nothing left queued after a decode
    out += dec.flush()
    assert out == reference  # (i) identical results, same order, repeat kept


def test_ax25_rising_snr_profile_no_loss_no_dup():
    # The audit-required rising-SNR profile on the sibling decoder: a weak frame
    # in drain 1 followed by strong frames in drain 2 must all be emitted once.
    weak = _ui(b"weak")
    s1 = _ui(b"strong-1")
    s2 = _ui(b"strong-2")
    gap = np.zeros(3000, np.complex64)
    dec = endurosat.StreamDecoder(_AX_SR)
    dec.push(np.concatenate([0.02 * endurosat.transmit(weak, _AX_SR), gap]).astype(np.complex64))
    out = dec.decode_new()
    assert out == [weak]
    dec.push(
        np.concatenate(
            [endurosat.transmit(s1, _AX_SR), gap, endurosat.transmit(s2, _AX_SR), gap]
        ).astype(np.complex64)
    )
    out += dec.decode_new()
    out += dec.flush()
    assert sorted(out) == sorted([weak, s1, s2])
    assert len(out) == 3


def test_endurosat_empty_chunks_are_harmless():
    dec = el.StreamDecoder(_TX_SR)
    dec.push(np.empty(0, np.complex64))
    assert dec.decode_new() == []
    assert dec.flush() == []
    dec_ax = endurosat.StreamDecoder(_AX_SR)
    dec_ax.push(np.empty(0, np.complex64))
    assert dec_ax.decode_new() == []
    assert dec_ax.flush() == []


def test_ax25_decode_new_without_new_samples_is_empty():
    body = _ui(b"once-only")
    dec = endurosat.StreamDecoder(_AX_SR)
    dec.push(
        np.concatenate(
            [endurosat.transmit(body, _AX_SR), np.zeros(3000, np.complex64)]
        ).astype(np.complex64)
    )
    assert dec.decode_new() == [body]
    assert dec.decode_new() == []  # no new samples -> nothing re-emitted
    assert dec.flush() == []


# ----------------------------------------------------------------------
# App level: LOW-3 record-only + LOW-7 data_format + MED-3(b) off-loop decode
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


def _run_engine(cap_path, sample_rate, params):
    import argparse

    args = argparse.Namespace(
        sample_rate=sample_rate,
        sdr_args=f"file:{cap_path}",
        center_freq_hz=endurosat.CENTER_FREQUENCY_HZ,
    )
    socks = _FakeSockets()
    started = asyncio.Event()
    started.set()
    stop = asyncio.Event()
    profile = rxapp._profile_from_params(params)
    asyncio.run(
        rxapp._run_dsp_engine(args, socks, params, started, stop, profile, {"hz": 0.0})
    )
    events = [
        json.loads(line) for line in socks.status_writer.buf.decode().splitlines() if line.strip()
    ]
    return events, bytes(socks.data_writer.buf)


def test_dsp_engine_verbatim_airmac_label_decodes_endurosat(tmp_path):
    # gs-client passes the backend framing label VERBATIM; the app must route it
    # through framings.normalize_framing (docs/10 P0-2 single normalization
    # point) — "EnduroSat AirMAC" runs the endurosat chip-packet decoder. Also
    # exercises the off-event-loop decode path end to end (MED-3(b)) and the
    # LOW-7 data_format label.
    cap = tmp_path / "airmac.cf32"
    payload = bytes(range(24))
    iq = np.concatenate(
        [np.zeros(2000, np.complex64), el.transmit(payload, 96_000.0), np.zeros(2000, np.complex64)]
    ).astype(np.complex64)
    iq.tofile(cap)

    events, data = _run_engine(cap, 96_000, {"framing": "EnduroSat AirMAC"})
    ready = next(e for e in events if e["event"] == "ready")
    # docs/13: both LIGHT framings run live (the label is a hint). The endurosat capture decodes;
    # ax25 stays silent (CRC-gated), so every emitted frame is tagged endurosat.
    assert ready["framing"] == "ax25,endurosat"
    assert ready["framing_hint"] == "endurosat"  # backend label, normalized (hint only)
    assert ready["data_format"] == "raw_bytes"  # explicit gs-client map key (LOW-7)
    frames = [e for e in events if e["event"] == "frame_received"]
    assert [base64.b64decode(f["frame"]["bytes_b64"]) for f in frames] == [payload]
    assert all(f["framing"] == "endurosat" for f in frames)  # only the endurosat deframer matched
    assert data == payload


def test_dsp_engine_nonlight_label_does_not_suppress_light_framings(tmp_path):
    # docs/13: the backend framing label is a HINT, not a filter. A non-light label ("USP", which
    # normalizes to no LOCAL framing) no longer forces record-only — the light framings
    # (ax25+endurosat) still run, so an EnduroSat capture is captured regardless of the label.
    # Non-light framings (USP/ccsds/kiss) are decoded POST-PASS on the recorded .cf32.
    cap = tmp_path / "unknown.cf32"
    payload = bytes(range(24))
    iq = np.concatenate(
        [np.zeros(2000, np.complex64), el.transmit(payload, 96_000.0), np.zeros(2000, np.complex64)]
    ).astype(np.complex64)
    iq.tofile(cap)

    events, data = _run_engine(cap, 96_000, {"framing": "USP"})
    ready = next(e for e in events if e["event"] == "ready")
    assert ready["framing"] == "ax25,endurosat"  # both light framings run despite the USP label
    assert ready["framing_hint"] == "none"  # "USP" normalizes to no LOCAL framing
    frames = [e for e in events if e["event"] == "frame_received"]
    assert [base64.b64decode(f["frame"]["bytes_b64"]) for f in frames] == [payload]
    assert all(f["framing"] == "endurosat" for f in frames)
    assert data == payload


def test_dsp_engine_ax25_label_still_captures_endurosat_traffic(tmp_path):
    # THE customer scenario: a pass LABELLED "ax25" whose real traffic is EnduroSat. Run-both means
    # the endurosat deframer runs alongside ax25, so the traffic is still captured (tagged
    # endurosat) in the SAME pass — the ax25 label no longer suppresses it.
    cap = tmp_path / "mislabeled.cf32"
    payload = bytes(range(24))
    iq = np.concatenate(
        [np.zeros(2000, np.complex64), el.transmit(payload, 96_000.0), np.zeros(2000, np.complex64)]
    ).astype(np.complex64)
    iq.tofile(cap)

    events, data = _run_engine(cap, 96_000, {"framing": "ax25"})
    ready = next(e for e in events if e["event"] == "ready")
    assert ready["framing"] == "ax25,endurosat"
    assert ready["framing_hint"] == "ax25"  # the (wrong) backend label — a hint, not a filter
    frames = [e for e in events if e["event"] == "frame_received"]
    assert [base64.b64decode(f["frame"]["bytes_b64"]) for f in frames] == [payload]
    assert all(f["framing"] == "endurosat" for f in frames)  # decoded despite the ax25 label
    assert data == payload


def test_unknown_framing_warns_exactly_once(caplog):
    label = "Mobitex-NX"  # fresh label (not used by other tests) for the warn-dedup
    rxapp._warned_framings.discard(label)
    with caplog.at_level(logging.WARNING, logger="cubesat_gfsk_ax25_rx"):
        assert rxapp._select_framing({"framing": label}) is None
        assert rxapp._select_framing({"framing": label}) is None
    warnings = [r for r in caplog.records if label in r.getMessage()]
    assert len(warnings) == 1  # one WARNING per unknown label, not per call

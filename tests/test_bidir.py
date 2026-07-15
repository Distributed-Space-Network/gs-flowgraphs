"""Bidirectional EnduroSat flowgraph: uplink-burst build, downlink demod, and the event contract.

No SDR / GNU Radio — the cf32 file (via FileBidirIo) stands in for the shared radio, exactly as the
RX/TX app tests do. Async coroutines are driven with asyncio.run (the repo has no pytest-asyncio).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json

import cubesat_gfsk_endurosat_bidir as bidir
import numpy as np
import pytest

from gfsk_ax25 import endurosat_link as el

_SR = 96_000.0  # multiple of 9600 → 10 samples/symbol


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


def _events(socks: _FakeSockets) -> list[dict]:
    return [json.loads(x) for x in socks.status_writer.buf.decode().splitlines() if x.strip()]


def _guard(iq: np.ndarray, n: int = 2000) -> np.ndarray:
    """Settling room around the burst — receive()/StreamDecoder are always fed that way."""
    return np.concatenate([np.zeros(n, np.complex64), iq, np.zeros(n, np.complex64)])


def _rx_args(tmp_path: object) -> argparse.Namespace:
    return argparse.Namespace(
        sample_rate=_SR, output_dir=str(tmp_path), record_iq=False, record_formats=""
    )


# --------------------------------------------------------------------------- TX build


def test_build_uplink_iq_raw_verbatim_decodes():
    # Rev A: the uplink file is ALREADY a framed EnduroSat packet; build_uplink_iq modulates it
    # VERBATIM (raw, no re-wrap), and the proven receiver still recovers the inner payload.
    inner = b"AIRMAC-uplink-command-blob"
    iq = _guard(bidir.build_uplink_iq(el.frame_bytes(inner), _SR, {}))
    assert inner in el.receive(iq, _SR)


def test_build_uplink_iq_no_wrap_no_truncate():
    # Rev A: build_uplink_iq no longer wraps or truncates — it modulates the bytes VERBATIM, so the
    # IQ length is exactly len*8*sps (no added preamble/sync/len/CRC), even for a >128 B train.
    sps = int(round(_SR / el.DEFAULT_SYMBOL_RATE_HZ))
    payload = bytes((np.arange(200) % 256).astype(np.uint8).tolist())  # > the old 128 B cap
    assert len(bidir.build_uplink_iq(payload, _SR, {})) == len(payload) * 8 * sps


def _longest_zero_run(iq: np.ndarray) -> int:
    silent = (np.abs(iq) == 0.0).astype(np.int8)
    if not silent.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([np.int8(0)], silent, [np.int8(0)]))))
    return int((edges[1::2] - edges[0::2]).max())


def test_build_uplink_iq_zero_gap_renders_silence():
    # A real uplink file: complete EnduroSat packets concatenated with zero-byte pads. The zero-gap
    # feature must actually insert IQ SILENCE for the pad run — NOT merely decode both packets
    # (deframe scans the whole bitstream, so 'both decode' holds with or without the gap: vacuous).
    a, b = b"first-command", b"second-command"
    train = el.frame_bytes(a) + b"\x00" * 40 + el.frame_bytes(b)
    sps = int(round(_SR / el.DEFAULT_SYMBOL_RATE_HZ))
    on = bidir.build_uplink_iq(train, _SR, {"uplink_zero_gap_bytes": 32})
    off = bidir.build_uplink_iq(train, _SR, {})  # verbatim: the pad is full-power FSK-0 carrier
    assert _longest_zero_run(on) >= 40 * 8 * sps  # the 40-byte pad became zero-amplitude silence
    assert _longest_zero_run(off) == 0  # verbatim modulation is |iq|==1 throughout (no silence)
    assert len(on) == len(off)  # silence preserves the pad's on-air duration
    got = bidir.demod_capture(_guard(on), _SR, {})
    assert a in got and b in got  # and both packets still decode across the silence gap


def test_build_uplink_iq_over_cap_raises():
    with pytest.raises(ValueError, match="cap"):
        bidir.build_uplink_iq(bytes(bidir._UPLINK_MAX_BYTES + 1), _SR, {})


def test_build_uplink_iq_empty_payload_is_empty_iq():
    # Empty file → empty IQ, consistently on BOTH paths (regression: the plain/min_gap<=0 path used
    # to raise via np.convolve while the zero-gap path returned empty).
    assert len(bidir.build_uplink_iq(b"", _SR, {})) == 0
    assert len(bidir.build_uplink_iq(b"", _SR, {"uplink_zero_gap_bytes": 32})) == 0


def test_demod_capture_roundtrip():
    inner = b"downlink-frame-payload"
    iq = _guard(bidir.build_uplink_iq(el.frame_bytes(inner), _SR, {}))
    assert inner in bidir.demod_capture(iq, _SR, {})


def test_decoder_kwargs_symbol_rate_aliases():
    assert bidir._decoder_kwargs({})["symbol_rate_hz"] == el.DEFAULT_SYMBOL_RATE_HZ
    assert bidir._decoder_kwargs({"baud": 4800})["symbol_rate_hz"] == 4800.0
    assert bidir._decoder_kwargs({"symbol_rate_hz": 2400})["symbol_rate_hz"] == 2400.0
    kw = bidir._decoder_kwargs({"mod_index": 0.7, "bt": 0.3})
    assert kw["mod_index"] == 0.7 and kw["bt"] == 0.3


# --------------------------------------------------------------------------- FileBidirIo


def test_file_bidir_io_rx_and_tx(tmp_path):
    rx = tmp_path / "rx.cf32"
    tx = tmp_path / "tx.cs16"
    iq = _guard(bidir.build_uplink_iq(b"hello", _SR, {}))
    rx.write_bytes(iq.astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(rx), str(tx))
    chunks = list(io.rx_chunks())
    assert sum(len(c) for c in chunks) == len(iq)
    # (3a) transmit_burst now takes the FINAL flat CS16 buffer, built pre-key by the controller.
    cs16 = bidir.to_cs16(bidir.build_uplink_iq(b"cmd", _SR, {}))
    result = io.transmit_burst(cs16)
    assert result.complete and result.accepted == cs16.size // 2
    assert io.sent_samples == result.accepted
    assert tx.stat().st_size == result.accepted * 4  # flat CS16 = 2 int16 * 2 bytes / complex


def test_file_bidir_io_no_rx_file_is_empty():
    io = bidir.FileBidirIo(None, None)
    assert list(io.rx_chunks()) == []
    result = io.transmit_burst(np.zeros(20, np.int16))  # 10 complex, discarded, still counted
    assert result.accepted == 10 and result.complete


# --------------------------------------------------------------------------- payload resolution


def test_uplink_payload_from_cmd_precedence(tmp_path):
    args = argparse.Namespace(output_dir=str(tmp_path))
    b64 = base64.b64encode(b"from-command").decode()
    assert bidir._uplink_payload_from_cmd({"bytes_b64": b64}, args, {}) == b"from-command"

    pf = tmp_path / "pl.bin"
    pf.write_bytes(b"from-file")
    assert bidir._uplink_payload_from_cmd({"payload_file": str(pf)}, args, {}) == b"from-file"
    assert bidir._uplink_payload_from_cmd({}, args, {"uplink_file": str(pf)}) == b"from-file"

    (tmp_path / "uplink.bin").write_bytes(b"from-default")
    assert bidir._uplink_payload_from_cmd({}, args, {}) == b"from-default"

    empty = argparse.Namespace(output_dir=str(tmp_path / "nope"))
    assert bidir._uplink_payload_from_cmd({}, empty, {}) == b""


# --------------------------------------------------------------------------- TX controller events


def test_resolve_sample_rate_snaps_to_integer_sps():
    # HIGH regression: the spawn-contract default --sample-rate is 2_000_000, which is NOT a 9600
    # multiple (208.33 sps) → gfsk.modulate would RAISE. resolve_sample_rate snaps it so every build
    # has integer sps.
    a2m = argparse.Namespace(sample_rate=2_000_000)
    r = bidir.resolve_sample_rate(a2m, {})
    assert r % el.DEFAULT_SYMBOL_RATE_HZ == 0
    assert abs(r - 2_000_000) < el.DEFAULT_SYMBOL_RATE_HZ  # nearest multiple
    assert bidir.resolve_sample_rate(argparse.Namespace(sample_rate=96_000), {}) == 96_000
    assert bidir.resolve_sample_rate(argparse.Namespace(sample_rate=0), {}) == 96_000  # unset
    assert bidir.resolve_sample_rate(a2m, {"baud": 4800}) % 4800 == 0  # custom symbol rate
    # the resolved rate is actually modulatable (the whole point) — must not raise:
    assert len(bidir.build_uplink_iq(b"cmd", r, {})) > 0


async def _prepare_then_transmit(tx, socks, frame_id, payload, params=None):
    """The real orchestrator sequence: stage while cold, THEN key, THEN burst the cached IQ."""
    ok = await tx.prepare(socks, frame_id, payload, tx._sample_rate, params or {})
    if not ok:
        return 0
    return await tx.transmit(socks, frame_id)


def test_tx_controller_stages_then_bursts(tmp_path):
    socks = _FakeSockets()
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io, sample_rate=_SR)
    sent = asyncio.run(_prepare_then_transmit(tx, socks, "f1", b"uplink"))
    evs = _events(socks)
    # tx_prepared is emitted PRE-KEY: it is the orchestrator's licence to energize the PA.
    assert [e["event"] for e in evs] == ["tx_prepared", "transmit_started", "transmit_complete"]
    assert evs[0]["frame_id"] == "f1"
    assert evs[0]["samples"] == len(bidir.build_uplink_iq(b"uplink", _SR, {}))
    assert evs[0]["payload_bytes"] == len(b"uplink")
    assert evs[2]["samples"] == sent == len(bidir.build_uplink_iq(b"uplink", _SR, {}))
    assert not tx.tx_active.is_set()  # cleared after the burst
    assert not tx.has_staged_burst  # one-shot: consumed


def test_tx_prepared_rate_is_consistent_with_hardware_samples_at_upsample_factor():
    """RE-AUDIT regression: on a real SDR the staged CS16 is upsampled to the hardware rate
    (factor > 1), so tx_prepared `samples` is the HARDWARE complex count. `sample_rate` must be the
    HARDWARE rate too, or the orchestrator's pre-key proof (samples / sample_rate == duration_s)
    fails for every burst and keying is REFUSED. FileBidirIo hides this at factor 1; drive > 1.
    """
    class _Factor22Io:  # a real-SDR-shaped IO: prepare() only reads tx_upsample_factor
        tx_upsample_factor = 22

    socks = _FakeSockets()
    tx = bidir._TxController(_Factor22Io(), sample_rate=_SR)
    payload = b"uplink-doppler-check"
    assert asyncio.run(tx.prepare(socks, "f22", payload, _SR, {})) is True
    ev = _events(socks)[0]
    assert ev["event"] == "tx_prepared"
    modem = len(bidir.build_uplink_iq(payload, _SR, {}))
    assert ev["samples"] == modem * 22, "samples is the hardware complex count"
    assert ev["sample_rate"] == int(_SR * 22), "sample_rate must be the HARDWARE rate"
    assert ev["predicted_samples"] == modem, "modem-rate count preserved separately"
    # THE INVARIANT the orchestrator's pre-key proof checks:
    assert abs(ev["samples"] / ev["sample_rate"] - ev["duration_s"]) < 0.01


def test_transmit_REFUSES_to_build_while_keyed(tmp_path):
    """THE INVARIANT. transmit() runs with the PA hot. If nothing was staged it must REFUSE — not
    quietly build the burst, which is the exact hazard the handshake removes."""
    socks = _FakeSockets()
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io, sample_rate=_SR)

    sent = asyncio.run(tx.transmit(socks, "never-staged"))  # no prepare!

    evs = _events(socks)
    assert [e["event"] for e in evs] == ["transmit_complete"]
    assert evs[0]["samples"] == sent == 0
    assert evs[0]["outcome"] == "error"
    assert "will NOT build a burst while keyed" in evs[0]["detail"]
    assert not (tmp_path / "tx.cf32").exists()  # nothing went on the air
    assert not tx.tx_active.is_set()


def test_a_staged_burst_is_not_flown_against_a_DIFFERENT_frame(tmp_path):
    """A stale stage must never be radiated under a later frame's id — that would transmit frame A
    while the orchestrator, the audit log and the operator all believe frame B went out."""
    socks = _FakeSockets()
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io, sample_rate=_SR)

    asyncio.run(tx.prepare(socks, "frame-A", b"payload-A", _SR, {}))
    sent = asyncio.run(tx.transmit(socks, "frame-B"))  # the orchestrator keyed for B

    evs = _events(socks)
    assert sent == 0
    assert evs[-1]["event"] == "transmit_complete"
    assert evs[-1]["outcome"] == "error"
    assert "frame-A" in evs[-1]["detail"]  # names what WAS staged
    assert not (tmp_path / "tx.cf32").exists()


def test_a_FAILED_stage_disarms_a_previously_staged_burst(tmp_path):
    """If frame B fails to stage, frame A's IQ must not survive in the cache — otherwise the
    orchestrator, told 'B failed', might still key for a retry and radiate A."""
    socks = _FakeSockets()
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io, sample_rate=_SR)

    assert asyncio.run(tx.prepare(socks, "frame-A", b"payload-A", _SR, {})) is True
    assert tx.has_staged_burst
    assert asyncio.run(tx.prepare(socks, "frame-B", b"", _SR, {})) is False  # empty → rejected
    assert not tx.has_staged_burst, "the rejected stage left frame A armed in the cache"


def test_an_unflyable_burst_is_rejected_BEFORE_the_key():
    """ROUND 10. This used to be 'a build failure still emits transmit_complete' — true, but it
    described a build failing with the PA already hot, recovered by a forced disarm. Now the same
    unflyable burst (100 kHz / 9600 baud = 10.42 sps, non-integer) is caught at STAGE time, and the
    orchestrator never keys at all."""
    socks = _FakeSockets()
    tx = bidir._TxController(bidir.FileBidirIo(None, None), sample_rate=100_000.0)

    ok = asyncio.run(tx.prepare(socks, "f1", b"payload", 100_000.0, {}))

    assert ok is False
    evs = _events(socks)
    assert [e["event"] for e in evs] == ["tx_prepare_failed"]
    assert evs[0]["code"] == "non-integer-sps"
    assert evs[0]["detail"]  # the reason travels with the event
    assert evs[0]["frame_id"] == "f1"
    assert not tx.has_staged_burst
    assert not tx.tx_active.is_set()  # nothing was ever keyed


# --------------------------------------------------------------------------- RX demod loop


def test_run_rx_emits_frame_from_downlink_file(tmp_path):
    cap = tmp_path / "downlink.cf32"
    payload = b"beacon-telemetry-xyz"
    iq = _guard(el.transmit(payload, _SR))  # a FRAMED downlink the RX StreamDecoder can decode
    cap.write_bytes(iq.astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(cap), None)
    socks = _FakeSockets()
    stop = asyncio.Event()
    tx = bidir._TxController(io, sample_rate=_SR)
    asyncio.run(
        bidir.run_rx(
            _rx_args(tmp_path), socks, {}, io, stop_requested=stop, doppler={"hz": 0.0}, tx=tx
        )
    )
    evs = _events(socks)
    frames = [e for e in evs if e["event"] == "frame_received"]
    assert any(base64.b64decode(f["frame"]["bytes_b64"]) == payload for f in frames)
    assert payload in bytes(socks.data_writer.buf)  # raw frame on the data socket
    assert any(e["event"] == "signal" for e in evs)  # at least one RSSI hint


def test_run_rx_survives_dead_status_socket(tmp_path):
    # Round-2 deadlock regression: a dead status socket (every drain raises) must NOT hang run_rx or
    # leave the reader parked on a full queue — status writes are suppressed and teardown completes.
    cap = tmp_path / "downlink.cf32"
    payload = b"telemetry-under-a-dead-status-socket"
    cap.write_bytes(_guard(el.transmit(payload, _SR)).astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(cap), None)

    class _DeadWriter(_FakeWriter):
        async def drain(self) -> None:
            raise ConnectionResetError("status socket dead")

    socks = _FakeSockets()
    socks.status_writer = _DeadWriter()
    socks.data_writer = _DeadWriter()
    stop = asyncio.Event()
    tx = bidir._TxController(io, sample_rate=_SR)

    async def _run() -> None:
        await asyncio.wait_for(
            bidir.run_rx(
                _rx_args(tmp_path), socks, {}, io, stop_requested=stop, doppler={"hz": 0.0}, tx=tx
            ),
            timeout=10.0,
        )

    asyncio.run(_run())  # returns (not TimeoutError) despite every socket write failing


def test_run_rx_terminates_on_stop_with_unbounded_source(tmp_path):
    # Teardown regression: with a never-ending source (like a live SDR), run_rx must terminate
    # promptly when stop_requested is set — NOT depend on a source EOF / the None sentinel.
    class _InfiniteIo:
        def rx_chunks(self):
            while True:
                yield np.zeros(1024, np.complex64)

        def transmit_burst(self, iq):
            return len(iq)

        def close(self):
            return None

    io = _InfiniteIo()
    socks = _FakeSockets()
    stop = asyncio.Event()
    tx = bidir._TxController(io, sample_rate=_SR)

    async def _run() -> None:
        task = asyncio.create_task(
            bidir.run_rx(
                _rx_args(tmp_path), socks, {}, io, stop_requested=stop, doppler={"hz": 0.0}, tx=tx
            )
        )
        await asyncio.sleep(0.3)  # let it stream a while
        stop.set()
        await asyncio.wait_for(task, timeout=10.0)  # must terminate, not hang

    asyncio.run(_run())


def test_run_rx_no_downlink_completes_clean(tmp_path):
    io = bidir.FileBidirIo(None, None)
    socks = _FakeSockets()
    stop = asyncio.Event()
    tx = bidir._TxController(io, sample_rate=_SR)
    asyncio.run(
        bidir.run_rx(
            _rx_args(tmp_path), socks, {}, io, stop_requested=stop, doppler={"hz": 0.0}, tx=tx
        )
    )
    assert not [e for e in _events(socks) if e["event"] == "frame_received"]


def test_run_rx_skips_rx_while_tx_active(tmp_path):
    # With tx_active latched, the reader parks — no chunks demod, no frames — proving RX yields the
    # device during an uplink burst.
    cap = tmp_path / "downlink.cf32"
    payload = b"should-not-be-decoded-during-tx"
    # A FRAMED (decodable) downlink, so the ONLY reason for no frames is the tx_active park.
    cap.write_bytes(_guard(el.transmit(payload, _SR)).astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(cap), None)
    socks = _FakeSockets()
    stop = asyncio.Event()
    tx = bidir._TxController(io, sample_rate=_SR)
    tx.tx_active.set()  # simulate an in-flight burst for the whole run
    asyncio.run(
        bidir.run_rx(
            _rx_args(tmp_path), socks, {}, io, stop_requested=stop, doppler={"hz": 0.0}, tx=tx
        )
    )
    assert not [e for e in _events(socks) if e["event"] == "frame_received"]


# --------------------------------------------------------------------------- TX Doppler pre-comp


def test_tx_doppler_hz_scales_and_flips_sign():
    # Uplink Doppler = downlink Doppler * (uplink/downlink), OPPOSITE sign (pre-compensation).
    # +5 kHz downlink Doppler on a 401 MHz downlink / 449.9 MHz uplink → -5.609 kHz uplink pre-comp.
    got = bidir.tx_doppler_hz(5000.0, 401_500_000.0, 449_900_000.0)
    assert got == pytest.approx(-5000.0 * (449_900_000.0 / 401_500_000.0))
    assert got < 0.0  # opposite sign


def test_tx_doppler_hz_same_freq_is_negated():
    # A same-frequency link (uplink == downlink) just negates the value (scale factor 1).
    assert bidir.tx_doppler_hz(3000.0, 437_000_000.0, 437_000_000.0) == pytest.approx(-3000.0)


def test_tx_doppler_hz_guards_zero_and_bad_downlink():
    assert bidir.tx_doppler_hz(0.0, 401_500_000.0, 449_900_000.0) == 0.0  # no Doppler
    assert bidir.tx_doppler_hz(5000.0, 0.0, 449_900_000.0) == 0.0  # divide-by-zero guard


def test_apply_nco_shifts_frequency():
    # A pure DC tone shifted by +f0 lands its spectral peak at +f0.
    fs = 96_000.0
    f0 = 6000.0
    dc = np.ones(4096, dtype=np.complex64)
    shifted = bidir.apply_nco(dc, f0, fs)
    spec = np.abs(np.fft.fft(shifted))
    peak_hz = np.fft.fftfreq(len(shifted), d=1.0 / fs)[int(np.argmax(spec))]
    assert peak_hz == pytest.approx(f0, abs=fs / len(shifted))


def test_apply_nco_zero_offset_is_identity():
    iq = bidir.build_uplink_iq(b"cmd", _SR, {})
    out = bidir.apply_nco(iq, 0.0, _SR)
    assert np.array_equal(out, iq)


def test_tx_controller_applies_doppler_precomp(tmp_path):
    # (3a) Doppler is now applied at STAGE time and the cache is the FINAL flat CS16 — so the
    # transmitted burst is the raw IQ rotated by the uplink pre-comp, packed to CS16.
    doppler = {"hz": 8000.0}
    tx_path = tmp_path / "tx.cs16"
    io = bidir.FileBidirIo(None, str(tx_path))
    tx = bidir._TxController(
        io, sample_rate=_SR, doppler=doppler, downlink_hz=401_500_000.0,
        uplink_hz=449_900_000.0,
    )
    socks = _FakeSockets()
    asyncio.run(_prepare_then_transmit(tx, socks, "f1", b"uplink-cmd"))
    raw = bidir.build_uplink_iq(b"uplink-cmd", _SR, {})
    tx_dop = bidir.tx_doppler_hz(8000.0, 401_500_000.0, 449_900_000.0)
    expected_cs16 = bidir.to_cs16(bidir.apply_nco(raw, tx_dop, _SR))
    on_air = np.frombuffer(tx_path.read_bytes(), dtype=np.int16)
    assert np.array_equal(on_air, expected_cs16)                 # Doppler applied, then packed
    assert not np.array_equal(on_air, bidir.to_cs16(raw))        # NOT the un-shifted burst


def test_build_final_cs16_rejects_a_non_finite_waveform(monkeypatch):
    # (3g) non-finite IQ in the built (pre-key) waveform is a rejection, not a keyed failure.
    tx = bidir._TxController(bidir.FileBidirIo(None, None), sample_rate=_SR)
    monkeypatch.setattr(
        bidir, "build_uplink_iq",
        lambda *_a, **_k: np.full(64, np.nan + 0j, dtype=np.complex64),
    )
    with pytest.raises(bidir.UplinkRejected) as e:
        tx._build_final_cs16(b"x", _SR, {}, 1)
    assert e.value.code == "non-finite-waveform"


def test_build_final_cs16_rejects_an_empty_waveform(monkeypatch):
    # (3g) an empty built waveform is a rejection.
    tx = bidir._TxController(bidir.FileBidirIo(None, None), sample_rate=_SR)
    monkeypatch.setattr(bidir, "build_uplink_iq", lambda *_a, **_k: np.zeros(0, dtype=np.complex64))
    with pytest.raises(bidir.UplinkRejected) as e:
        tx._build_final_cs16(b"x", _SR, {}, 1)
    assert e.value.code == "empty-waveform"


def test_tx_controller_no_doppler_is_verbatim(tmp_path):
    # Default controller (no doppler dict / no freqs) stages the raw verbatim IQ, packed to CS16.
    tx_path = tmp_path / "tx.cs16"
    io = bidir.FileBidirIo(None, str(tx_path))
    tx = bidir._TxController(io, sample_rate=_SR)
    asyncio.run(_prepare_then_transmit(tx, _FakeSockets(), "f1", b"uplink-cmd"))
    on_air = np.frombuffer(tx_path.read_bytes(), dtype=np.int16)
    raw = bidir.build_uplink_iq(b"uplink-cmd", _SR, {})
    assert np.array_equal(on_air, bidir.to_cs16(raw))


def test_version(capsys):
    assert bidir.main(["--version"]) == 0
    assert "0." in capsys.readouterr().out


# --------------------------------------------------------------- ROUND 10: the uplink is BOUNDED
#
# The bidirectional path had ONE guard: a 64 kB byte cap. But bytes are not what a burst costs —
# gfsk.modulate() does np.repeat(symbols, sps), so sps and the total sample count decide the
# allocation. And none of these parameters are ours: symbol_rate comes from the backend's
# transmitter catalogue, which has offered baud=10.


def test_a_sane_burst_survives_validation():
    samples = bidir.validate_uplink(b"hello", _SR, {})
    assert samples == len(b"hello") * 8 * int(_SR // 9600)
    assert len(bidir.build_uplink_iq(b"hello", _SR, {})) == samples


def test_an_empty_payload_is_rejected():
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"", _SR, {})
    assert e.value.code == "empty-payload"


def test_an_oversize_payload_is_rejected():
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"x" * (bidir._UPLINK_MAX_BYTES + 1), _SR, {})
    assert e.value.code == "payload-too-large"


def test_the_np_repeat_ALLOCATION_BOMB_is_refused():
    """A 64 kB payload is 'small'. At 9600 baud and a 100 MHz sample rate it is 10416 sps, and
    np.repeat then asks numpy for ~43 GB — inside the keyed window, on the old code. The byte cap
    does not see this at all; only an sps/sample-count bound does."""
    payload = b"x" * bidir._UPLINK_MAX_BYTES
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(payload, 96_000_000.0, {})  # 10_000 sps
    assert e.value.code == "sps-too-large"


def test_a_burst_past_the_sample_ceiling_is_refused():
    """Under the sps cap AND under the duration cap, but past the total-sample cap. 25000 B at 9600
    baud = 20.8 s (< 30 s) and, at the 1024 sps ceiling, 25000*8*1024 = 204.8M samples (> 200M)."""
    payload = b"x" * 25_000
    sr = 9600.0 * bidir._MAX_SPS  # exactly at the sps cap
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(payload, sr, {})
    assert e.value.code == "iq-too-large"


@pytest.mark.parametrize("baud", [10, 1, 0.5, 1199])
def test_a_sub_protocol_baud_is_refused(baud):
    """The REST backend's transmitter catalogue has offered baud=10. At 10 baud a 1 kB payload is a
    13-minute transmission — with the PA keyed the whole time."""
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"cmd", _SR, {"baud": baud})
    assert e.value.code in ("symbol-rate-unusable", "non-integer-sps")


@pytest.mark.parametrize(
    ("params", "rate"),
    [
        ({"mod_index": float("nan")}, _SR),
        ({"bt": float("inf")}, _SR),
        ({}, float("inf")),
        ({}, float("nan")),
    ],
)
def test_non_finite_parameters_are_refused(params, rate):
    """NaN/inf propagate silently through the DSP and produce an all-NaN burst — which the SDR will
    happily key up and transmit as noise across the band."""
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"cmd", rate, params)
    assert e.value.code in ("non-finite-parameter", "symbol-rate-unusable", "sample-rate-unusable")


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_the_RX_demod_falls_back_on_a_non_finite_baud(bad):
    """ROUND 10: symbol_rate_hz_of() gated on ``v > 0``, and ``inf > 0`` is True — so an infinite
    baud was handed back as a usable symbol rate. It now demands a FINITE rate. For the DEMOD path
    (RX), a garbage rate falls back to the catalogue default rather than poisoning the DSP — the
    receiver must keep going."""
    assert bidir._decoder_kwargs({"baud": bad})["symbol_rate_hz"] == 9600.0


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0.0, -1.0, "garbage", None])
def test_a_garbage_TX_baud_is_REJECTED_not_coerced(bad):
    """ROUND 11 (P1). A garbage baud in a TX COMMAND is a different thing from a garbage baud on the
    RX demod. The RX demod falls back and keeps receiving; a STATION TOLD TO TRANSMIT at baud=NaN
    must be REFUSED, not quietly retuned to 9600 and keyed. The round-10 fallback was correct for RX
    and wrong here."""
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"cmd", _SR, {"baud": bad})
    assert e.value.code == "symbol-rate-unusable"


@pytest.mark.parametrize("bad", [0.0, -1.0, 20.0])
def test_an_unusable_mod_index_is_refused(bad):
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"cmd", _SR, {"mod_index": bad})
    assert e.value.code == "modulation-unusable"


def test_the_app_ADVERTISES_that_it_requires_the_handshake():
    """`ready` means the DOWNLINK is live; it never meant an uplink was flyable. The app must say so
    out loud, or an orchestrator will go on keying first and asking later."""
    import inspect

    src = inspect.getsource(bidir.amain)
    assert '"tx_prepare_required": True' in src
    assert '"event": "ready"' in src


def test_prepare_transmit_is_a_REGISTERED_command():
    import inspect

    src = inspect.getsource(bidir.amain)
    assert '"prepare_transmit": _on_prepare_transmit' in src


def test_apply_nco_is_identical_across_the_chunk_boundary():
    """ROUND 10: apply_nco is chunked (bounded working set inside the keyed window). The output must
    be bit-for-bit what the whole-array float64-phase computation produced — a burst that straddles
    _NCO_CHUNK must not glitch at the seam."""
    rng = np.random.default_rng(0)
    n = bidir._NCO_CHUNK + 12345  # straddles exactly one chunk boundary
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    freq, sr = 8000.0, 124800.0

    got = bidir.apply_nco(iq, freq, sr)

    # The reference: the exact whole-array form apply_nco replaced.
    ph = 2.0 * np.pi * freq * np.arange(n) / sr
    expected = (iq * np.exp(1j * ph)).astype(np.complex64)

    assert got.dtype == np.complex64
    assert np.array_equal(got, expected), "the chunked NCO differs from the whole-array computation"


# --------------------------------------------------------- ROUND 11 (P0-5): the RF DURATION bound


def test_a_multi_minute_burst_is_REFUSED_before_keying():
    """THE POST-KEY RADIATION HAZARD. The round-10 caps bound the MODEM IQ, but RF duration =
    payload_bits / baud is independent of the sample rate, and the burst is upsampled to the
    hardware rate AFTER the PA is keyed. A 64 kB payload at the 1200-baud floor is ~437 s of RF —
    past gs-client's 60 s completion timeout, so gs-client gives up while the PA keeps radiating for
    6 more minutes and the post-key upsample allocates ~7 GB. It must be refused while cold."""
    # 64 kB at 1200 baud, at a sample rate that keeps sps in-bounds (1200*80 = 96000).
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(b"x" * 65_536, 96_000.0, {"baud": 1200})
    assert e.value.code == "burst-too-long"


def test_a_short_burst_within_the_duration_cap_is_accepted():
    # ~1024 bytes at 9600 baud = 0.85 s of RF — well inside the cap.
    samples = bidir.validate_uplink(b"x" * 1024, _SR, {})
    assert samples > 0
    # And the duration bound matches payload_bits / baud.
    assert bidir._MAX_BURST_SECONDS > (1024 * 8) / 9600.0


def test_the_duration_bound_is_independent_of_sample_rate():
    """The same payload+baud is the same air time at ANY sample rate — the bound must not be dodged
    by lowering the sample rate."""
    payload = b"x" * 8000  # 8000*8 / 1200 = 53.3 s > 30 s cap, regardless of sample rate
    for sr in (96_000.0, 12_000.0, 2_400_000.0):
        # keep sps integer: sr must be a multiple of 1200
        if sr % 1200 != 0:
            continue
        with pytest.raises(bidir.UplinkRejected) as e:
            bidir.validate_uplink(payload, sr, {"baud": 1200})
        assert e.value.code in ("burst-too-long", "sps-too-large")


# ------------------------------------------------- ROUND 11 (P0-4): stop aborts an in-flight burst


def test_stop_during_a_burst_is_processed_and_aborts_it():
    """THE DEFECT. run_command_loop dispatches serially: if the transmit handler AWAITED the whole
    burst (up to minutes of RF), the loop could not read the next command, so a `stop` on the
    control socket was never dequeued and stop_requested was never set — the abort machinery, wired
    and polling, was unreachable. Round 11 spawns the burst as a task; the loop stays free, reads
    `stop`, sets the flag, and the burst polls it and cancels mid-flight."""
    import cubesat_gfsk_endurosat_bidir as b
    from _spawn_contract import run_command_loop

    async def _run() -> tuple[str, list[dict]]:
        stop_requested = asyncio.Event()

        class _BlockingIo:
            """transmit_burst blocks — as a real multi-second burst would — until should_abort()."""
            def __init__(self) -> None:
                self.tx_active = None
                self.started = asyncio.Event()

            def transmit_burst(self, iq, *, on_first_accept=None, should_abort=None):
                import time as _t
                self._loop_started = True
                # Poll should_abort the way _soapy_tx.write_burst does between chunks.
                for _ in range(500):
                    if should_abort is not None and should_abort():
                        return b.BurstResult(
                            accepted=0, total=len(iq), outcome="cancelled", detail="stop")
                    _t.sleep(0.005)
                return b.BurstResult(
                    accepted=len(iq), total=len(iq), outcome="complete", detail="")

            def close(self) -> None: ...

        io = _BlockingIo()
        tx = b._TxController(io, sample_rate=_SR, should_abort=stop_requested.is_set)
        socks = _FakeSockets()
        socks.control_reader = None  # unused by these handlers

        tx_tasks: list[asyncio.Task] = []

        async def _on_prepare(cmd):
            await tx.prepare(socks, str(cmd.get("frame_id", "")), b"hello", _SR, {})

        async def _on_transmit(cmd):  # mirrors amain: SPAWN, do not await
            t = asyncio.create_task(tx.transmit(socks, str(cmd.get("frame_id", ""))))
            tx_tasks.append(t)

        async def _on_stop(_cmd):
            stop_requested.set()

        handlers = {"prepare_transmit": _on_prepare, "transmit_frame": _on_transmit,
                    "stop": _on_stop}

        reader = asyncio.StreamReader()
        for obj in ({"cmd": "prepare_transmit", "frame_id": "f1"},
                    {"cmd": "transmit_frame", "frame_id": "f1"},
                    {"cmd": "stop", "reason": "operator"}):
            reader.feed_data((json.dumps(obj) + "\n").encode())
        reader.feed_eof()

        reason = await asyncio.wait_for(
            run_command_loop(reader, handlers, socks.status_writer), timeout=5.0
        )
        # The loop returned on 'stop' WITHOUT waiting the full burst — proof it was not blocked.
        await asyncio.wait_for(asyncio.gather(*tx_tasks, return_exceptions=True), timeout=5.0)
        return reason, _events(socks)

    reason, evs = asyncio.run(_run())
    assert reason == "stop"
    complete = [e for e in evs if e["event"] == "transmit_complete"]
    assert complete and complete[-1]["outcome"] == "cancelled", (
        f"the in-flight burst was not aborted by stop: {evs}"
    )


# --------------------------------- ROUND 12 (11th audit, P0-5): hardware-rate memory bound


def test_the_HARDWARE_rate_allocation_is_bounded_before_keying():
    """The modem-sample cap is not what gets allocated: the burst is upsampled to the hardware rate
    (samples * factor) and resample_poly materializes THAT buffer AFTER keying. A burst under the
    modem cap can still be ~0.5 GB of hardware IQ. validate_uplink now bounds the hardware-rate
    count given the TX upsample factor, and rejects cold."""
    # A burst under the modem cap (~8.2M modem samples) but x16 upsample = 131M hardware > 64M cap.
    payload = b"x" * 1024  # 1024*8 = 8192 bits
    # sps 1000 (sample_rate = 1000 * 9600) → 8192 * 1000 = 8.192M modem samples (under 200M, ~7s RF)
    modem = bidir.validate_uplink(payload, 9600.0 * 1000, {})  # factor default 1 → accepted
    assert modem == 8192 * 1000
    with pytest.raises(bidir.UplinkRejected) as e:
        bidir.validate_uplink(payload, 9600.0 * 1000, {}, hardware_factor=16)
    assert e.value.code == "hardware-iq-too-large"


def test_a_burst_within_the_hardware_ceiling_is_accepted():
    samples = bidir.validate_uplink(b"cmd", _SR, {}, hardware_factor=22)  # small burst, x22
    assert samples > 0


def test_a_staged_burst_aborts_before_the_write():
    """P0-5 / (3a): the big allocation (resample) is now PRE-KEY; a stop before the write still
    cancels without radiating. transmit_burst receives the FINAL flat CS16 buffer."""
    io = bidir.FileBidirIo(None, None)
    # FileBidirIo returns cancelled if should_abort() is True — before any write.
    result = io.transmit_burst(np.zeros(2000, np.int16), should_abort=lambda: True)
    assert result.outcome == "cancelled"
    assert result.accepted == 0


# ------------------- HARDWARE-SAFETY EXTENSION: the post-burst resume_rx handshake


def test_the_app_ADVERTISES_the_rx_resume_handshake():
    """A burst leaves RX STOPPED (the external T/R switch may still be on TX and the PA
    energized when transmit_complete is emitted). The app must say so in `ready`, register
    the resume command, and ack it — or an orchestrator would assume RX came back."""
    import inspect

    src = inspect.getsource(bidir.amain)
    assert '"rx_resume_required": True' in src
    assert '"resume_rx": _on_resume_rx' in src
    assert '"event": "rx_resumed"' in src


def test_resume_rx_round_trips_and_acks_rx_resumed():
    """The handshake, protocol-shape (mirrors amain's handler like the stop test above):
    gs-client — having de-keyed, proven the PA quiet, selected RX and settled 2 s — sends
    resume_rx; the app resumes the io and acks rx_resumed. FileBidirIo is the explicit
    SIMULATED implementation: the protocol round-trips identically with no device."""
    from _spawn_contract import run_command_loop, send_event

    async def _run():
        io = bidir.FileBidirIo(None, None)
        socks = _FakeSockets()

        async def _on_resume_rx(_cmd):
            await asyncio.to_thread(io.resume_rx)
            await send_event(socks.status_writer, {"event": "rx_resumed"})

        async def _on_stop(_cmd):
            return None

        handlers = {"resume_rx": _on_resume_rx, "stop": _on_stop}
        reader = asyncio.StreamReader()
        for obj in ({"cmd": "resume_rx"}, {"cmd": "stop"}):
            reader.feed_data((json.dumps(obj) + "\n").encode())
        reader.feed_eof()
        reason = await asyncio.wait_for(
            run_command_loop(reader, handlers, socks.status_writer), timeout=5.0
        )
        return reason, _events(socks)

    reason, evs = asyncio.run(_run())
    assert reason == "stop"
    assert any(e["event"] == "rx_resumed" for e in evs)


def test_a_failing_resume_rx_fails_the_command_loop_and_never_acks():
    """Failure keeps RX stopped and fails the pass: a raising resume makes
    run_command_loop report handler-failed (amain converts that to a NONZERO exit), and
    no rx_resumed ack is ever emitted — there is no best-effort resume."""
    from _spawn_contract import run_command_loop, send_event

    class _RefusingIo:
        def resume_rx(self):
            msg = "activate refused"
            raise RuntimeError(msg)

    async def _run():
        io = _RefusingIo()
        socks = _FakeSockets()

        async def _on_resume_rx(_cmd):
            await asyncio.to_thread(io.resume_rx)
            await send_event(socks.status_writer, {"event": "rx_resumed"})

        handlers = {"resume_rx": _on_resume_rx}
        reader = asyncio.StreamReader()
        reader.feed_data((json.dumps({"cmd": "resume_rx"}) + "\n").encode())
        reader.feed_eof()
        reason = await asyncio.wait_for(
            run_command_loop(reader, handlers, socks.status_writer), timeout=5.0
        )
        return reason, _events(socks)

    reason, evs = asyncio.run(_run())
    assert reason == "handler-failed"
    assert not any(e.get("event") == "rx_resumed" for e in evs)


# ------------------------ CA-FLOW-001: `ready` must PROVE the downlink is alive


class _AmainSockets:
    def __init__(self) -> None:
        self.status_writer = _FakeWriter()
        self.data_writer = _FakeWriter()
        self.control_reader = asyncio.StreamReader()

    async def aclose(self) -> None:
        return None


def _amain_args(tmp_path, rx: str):
    return argparse.Namespace(
        params_file=None, sample_rate=_SR, center_freq_hz=401_500_000.0,
        sdr_args=f"file:{rx}", output_dir=str(tmp_path), record_iq=False,
        record_formats="",
    )


def _run_amain(monkeypatch, args, *, feed_eof: bool = False) -> tuple[int, list[dict]]:
    box: dict = {}

    async def _fake_connect(_args):
        # StreamReader must be constructed INSIDE the running loop (3.12).
        socks = _AmainSockets()
        if feed_eof:
            socks.control_reader.feed_eof()
        box["socks"] = socks
        return socks

    monkeypatch.setattr(bidir, "connect_spawn_sockets", _fake_connect)
    rc = asyncio.run(asyncio.wait_for(bidir.amain(args), timeout=20.0))
    return rc, _events(box["socks"])


def test_empty_input_emits_no_ready_and_exits_nonzero(tmp_path, monkeypatch):
    """THE REPRO: FileBidirIo with a missing rx file returns immediately (the 'normal
    empty-file return'). The old app emitted `ready` before the RX task even existed
    and the command loop stayed alive behind a dead engine."""
    args = _amain_args(tmp_path, rx=str(tmp_path / "missing.cf32"))
    rc, evs = _run_amain(monkeypatch, args, feed_eof=True)
    assert rc == 1
    assert not any(e.get("event") == "ready" for e in evs), "ready without a live downlink"
    assert any(e.get("code") == "rx-not-alive" for e in evs)


def test_one_sample_permits_exactly_one_ready_and_early_end_fails(tmp_path, monkeypatch):
    """One delivered sample licenses exactly ONE ready — and the engine ENDING
    normally before a requested stop (finite file exhausted) must still fail the
    pass (engine-ended), never linger behind a live command loop."""
    rx = tmp_path / "live.cf32"
    np.zeros(200_000, np.complex64).tofile(str(rx))
    args = _amain_args(tmp_path, rx=str(rx))
    rc, evs = _run_amain(monkeypatch, args)  # no stop is ever sent
    readies = [e for e in evs if e.get("event") == "ready"]
    assert len(readies) == 1, f"expected exactly one ready, got {len(readies)}"
    assert any(e.get("code") == "engine-ended" for e in evs), (
        "premature normal engine completion did not fail the pass"
    )
    assert rc == 1


def test_deaf_source_times_out_without_ready(tmp_path, monkeypatch):
    """A source that never delivers a sample (and never ends) must time out the
    first-sample bound: no ready, explicit rx-not-alive, nonzero exit."""
    import threading as _threading
    import time as _time

    monkeypatch.setattr(bidir, "_READY_FIRST_SAMPLE_TIMEOUT_S", 0.3)

    class _DeafIo(bidir.FileBidirIo):
        def __init__(self) -> None:
            super().__init__(None, None)
            self._stopped = _threading.Event()

        def rx_chunks(self):
            while not self._stopped.is_set():
                _time.sleep(0.005)
            return
            yield  # pragma: no cover — makes this a generator

        def close(self) -> None:
            self._stopped.set()

    monkeypatch.setattr(bidir, "_open_bidir_io", lambda _a, _p: _DeafIo())
    args = _amain_args(tmp_path, rx="ignored")
    rc, evs = _run_amain(monkeypatch, args, feed_eof=True)
    assert rc == 1
    assert not any(e.get("event") == "ready" for e in evs)
    deaf = [e for e in evs if e.get("code") == "rx-not-alive"]
    assert deaf and "deaf" in deaf[0].get("detail", "")

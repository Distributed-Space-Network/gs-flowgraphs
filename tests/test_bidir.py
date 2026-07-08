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
    tx = tmp_path / "tx.cf32"
    iq = _guard(bidir.build_uplink_iq(b"hello", _SR, {}))
    rx.write_bytes(iq.astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(rx), str(tx))
    chunks = list(io.rx_chunks())
    assert sum(len(c) for c in chunks) == len(iq)
    sent = io.transmit_burst(bidir.build_uplink_iq(b"cmd", _SR, {}))
    assert sent > 0 and io.sent_samples == sent
    assert tx.stat().st_size == sent * 8  # complex64


def test_file_bidir_io_no_rx_file_is_empty():
    io = bidir.FileBidirIo(None, None)
    assert list(io.rx_chunks()) == []
    assert io.transmit_burst(np.zeros(10, np.complex64)) == 10  # discarded, still counted


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


def test_tx_controller_emits_started_and_complete(tmp_path):
    socks = _FakeSockets()
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io)
    sent = asyncio.run(tx.transmit(socks, b"uplink", _SR, {}))
    evs = _events(socks)
    assert [e["event"] for e in evs] == ["transmit_started", "transmit_complete"]
    assert evs[1]["samples"] == sent == len(bidir.build_uplink_iq(b"uplink", _SR, {}))
    assert not tx.tx_active.is_set()  # cleared after the burst


def test_tx_controller_still_emits_complete_when_build_fails():
    # MED regression: a build/burst failure must STILL emit transmit_complete (samples=0), or the
    # orchestrator's half-duplex loop hangs waiting for it. A non-integer-sps rate makes the
    # modulator raise inside the controller.
    socks = _FakeSockets()
    tx = bidir._TxController(bidir.FileBidirIo(None, None))
    sent = asyncio.run(tx.transmit(socks, b"payload", 100_000.0, {}))  # 10.42 sps → raises
    evs = _events(socks)
    assert [e["event"] for e in evs] == ["transmit_started", "transmit_complete"]
    assert evs[1]["samples"] == sent == 0
    assert not tx.tx_active.is_set()


# --------------------------------------------------------------------------- RX demod loop


def test_run_rx_emits_frame_from_downlink_file(tmp_path):
    cap = tmp_path / "downlink.cf32"
    payload = b"beacon-telemetry-xyz"
    iq = _guard(el.transmit(payload, _SR))  # a FRAMED downlink the RX StreamDecoder can decode
    cap.write_bytes(iq.astype(np.complex64).tobytes())
    io = bidir.FileBidirIo(str(cap), None)
    socks = _FakeSockets()
    stop = asyncio.Event()
    tx = bidir._TxController(io)
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
    tx = bidir._TxController(io)

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
    tx = bidir._TxController(io)

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
    tx = bidir._TxController(io)
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
    tx = bidir._TxController(io)
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
    # With a live downlink Doppler + split frequencies, the transmitted burst is the verbatim IQ
    # rotated by the uplink pre-comp — NOT the raw IQ. Same length, different samples.
    doppler = {"hz": 8000.0}
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(
        io, doppler=doppler, downlink_hz=401_500_000.0, uplink_hz=449_900_000.0
    )
    socks = _FakeSockets()
    asyncio.run(tx.transmit(socks, b"uplink-cmd", _SR, {}))
    raw = bidir.build_uplink_iq(b"uplink-cmd", _SR, {})
    expected = bidir.apply_nco(
        raw, bidir.tx_doppler_hz(8000.0, 401_500_000.0, 449_900_000.0), _SR
    )
    on_air = np.frombuffer((tmp_path / "tx.cf32").read_bytes(), dtype=np.complex64)
    assert len(on_air) == len(raw)
    assert not np.array_equal(on_air, raw.astype(np.complex64))  # Doppler was applied
    assert np.allclose(on_air, expected, atol=1e-4)


def test_tx_controller_no_doppler_is_verbatim(tmp_path):
    # Default controller (no doppler dict / no freqs) transmits the raw verbatim IQ unchanged.
    io = bidir.FileBidirIo(None, str(tmp_path / "tx.cf32"))
    tx = bidir._TxController(io)
    asyncio.run(tx.transmit(_FakeSockets(), b"uplink-cmd", _SR, {}))
    on_air = np.frombuffer((tmp_path / "tx.cf32").read_bytes(), dtype=np.complex64)
    raw = bidir.build_uplink_iq(b"uplink-cmd", _SR, {}).astype(np.complex64)
    assert np.array_equal(on_air, raw)


def test_version(capsys):
    assert bidir.main(["--version"]) == 0
    assert "0." in capsys.readouterr().out

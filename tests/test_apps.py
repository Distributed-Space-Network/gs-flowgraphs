"""App-level tests: engine selection, --version, and a TX->file->RX round trip.

Exercises the flowgraph apps' I/O helpers and engine plumbing without any SDR or
GNU Radio (the cf32 file source/sink stands in for the radio)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json

import cubesat_gfsk_ax25_rx as rxapp
import cubesat_gfsk_ax25_tx as txapp

from gfsk_ax25 import ax25, endurosat

_SR = 99_840  # integer sps for the modulator


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


def test_emit_frame_decodes_ui_and_writes_raw_bytes():
    # Regression: _emit_frame must decode the AX.25 UI header (dest/src) and put
    # the raw frame on the data socket. Guards the ax25 import in the rx app.
    body = ax25.encode_ui(dest="DSN0", src="ES1", info=b"telemetry")
    socks = _FakeSockets()
    asyncio.run(rxapp._emit_frame(socks, body))

    event = json.loads(socks.status_writer.buf.decode().strip())
    assert event["event"] == "frame_received"
    assert event["frame"]["dest"] == "DSN0"
    assert event["frame"]["src"] == "ES1"
    assert event["frame"]["len"] == len(body)
    assert bytes(socks.data_writer.buf) == body


def test_version(capsys):
    assert rxapp.main(["--version"]) == 0
    assert txapp.main(["--version"]) == 0
    assert "0." in capsys.readouterr().out


def test_engine_selection_precedence(monkeypatch):
    ns = argparse.Namespace(engine="")
    monkeypatch.delenv("GS_FLOWGRAPH_ENGINE", raising=False)
    assert rxapp._select_engine(ns, {}) == "dsp"  # default
    assert rxapp._select_engine(ns, {"engine": "gnuradio"}) == "gnuradio"  # params
    monkeypatch.setenv("GS_FLOWGRAPH_ENGINE", "gnuradio")
    assert rxapp._select_engine(ns, {}) == "gnuradio"  # env over default
    assert rxapp._select_engine(argparse.Namespace(engine="dsp"), {}) == "dsp"  # flag wins
    assert rxapp._select_engine(argparse.Namespace(engine="bogus"), {}) == "dsp"  # fallback


def test_dsp_engine_decodes_capture_from_file(tmp_path):
    # Drives the whole dsp RX engine (reader thread -> NCO -> StreamDecoder ->
    # flush -> _emit_frame) against a cf32 capture, no SDR. Guards the engine
    # wiring and the shared-doppler refactor.
    cap = tmp_path / "capture.cf32"
    body = ax25.encode_ui(dest="DSN0", src="ES1", info=b"engine-path-frame")
    profile = endurosat.LinkProfile()
    iq = endurosat.transmit(body, _SR, profile=profile)
    tx_args = argparse.Namespace(sample_rate=_SR, sdr_args=f"file:{cap}")
    txapp._sink_iq(tx_args, iq)

    rx_args = argparse.Namespace(
        sample_rate=_SR, sdr_args=f"file:{cap}", center_freq_hz=endurosat.CENTER_FREQUENCY_HZ
    )
    socks = _FakeSockets()
    started = asyncio.Event()
    started.set()
    stop = asyncio.Event()
    doppler = {"hz": 0.0}
    asyncio.run(rxapp._run_dsp_engine(rx_args, socks, {}, started, stop, profile, doppler))

    events = [
        json.loads(line)
        for line in socks.status_writer.buf.decode().splitlines()
        if line.strip()
    ]
    assert any(e["event"] == "ready" and e["engine"] == "dsp" for e in events)
    frames = [e for e in events if e["event"] == "frame_received"]
    assert any(e["frame"].get("src") == "ES1" for e in frames)
    assert bytes(socks.data_writer.buf) == body


def test_tx_file_to_rx_file_roundtrip(tmp_path):
    iq_path = tmp_path / "uplink.cf32"
    payload = b"CMD set-beacon-interval 30s"
    profile = endurosat.LinkProfile()
    params = {"dest": "ES1", "src": "DSN0", "uplink_b64": base64.b64encode(payload).decode()}

    tx_args = argparse.Namespace(
        sample_rate=_SR, output_dir=str(tmp_path), sdr_args=f"file:{iq_path}",
        center_freq_hz=endurosat.CENTER_FREQUENCY_HZ, engine="dsp",
    )
    iq = txapp._build_frame_iq(tx_args, params, profile)
    txapp._sink_iq(tx_args, iq)
    assert iq_path.exists() and iq_path.stat().st_size > 0

    rx_args = argparse.Namespace(
        sample_rate=_SR, sdr_args=f"file:{iq_path}",
        center_freq_hz=endurosat.CENTER_FREQUENCY_HZ, engine="dsp",
    )
    dec = endurosat.StreamDecoder(_SR, profile=profile, recover_timing=False)
    for chunk in rxapp._open_iq_source(rx_args):
        dec.push(chunk)
    frames = dec.flush()

    expected = ax25.encode_ui(dest="ES1", src="DSN0", info=payload)
    assert expected in frames
    ui = ax25.decode_ui(frames[0])
    assert ui is not None and ui.info == payload

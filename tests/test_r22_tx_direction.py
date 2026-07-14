"""R-22 + F-03 leftover regressions.

R-22: TX endpoints get TX-explicit settings ONLY (``sdr_tx_*`` params /
``GS_SDR_TX_*`` env). RX-oriented antenna/gain-element names (``LNAW``,
``LNA/TIA/PGA``) never reach a TX direction — on LMS7/XTRX they raise in
``setAntenna``/``setGain`` and kill the pass.

Stop→abort: a pass stop cancels an IN-FLIGHT burst (outcome="cancelled")
instead of radiating it to completion — wired from the apps' stop events
down to the shared bounded TX transport's ``should_abort`` hook.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cubesat_gfsk_ax25_tx as txapp
import cubesat_gfsk_endurosat_bidir as bidir
import numpy as np
from _soapy import merge_sdr_params_tx

_SR = 99_840


def _clear_env(monkeypatch) -> None:
    for name in ("GS_SDR_ANTENNA", "GS_SDR_GAIN_DB", "GS_SDR_GAINS", "GS_SDR_AGC",
                 "GS_SDR_TX_ANTENNA", "GS_SDR_TX_GAIN_DB", "GS_SDR_TX_GAINS"):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------- R-22 merge


def test_tx_merge_is_empty_when_nothing_tx_specific_is_configured(monkeypatch) -> None:
    _clear_env(monkeypatch)
    # Generic/RX sources present everywhere — none of them may leak to TX.
    monkeypatch.setenv("GS_SDR_ANTENNA", "LNAW")
    monkeypatch.setenv("GS_SDR_GAINS", "LNA=30,TIA=9,PGA=3")
    monkeypatch.setenv("GS_SDR_GAIN_DB", "45")
    merged = merge_sdr_params_tx(
        {"sdr_antenna": "LNAL", "sdr_gains": {"LNA": 10}, "sdr_gain_db": 45.0, "sdr_agc": True}
    )
    assert merged == {}


def test_tx_merge_params_win_over_tx_env(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_TX_ANTENNA", "BAND1")
    monkeypatch.setenv("GS_SDR_TX_GAIN_DB", "20")
    monkeypatch.setenv("GS_SDR_TX_GAINS", "PAD=40,IAMP=6")
    merged = merge_sdr_params_tx({"sdr_tx_gain_db": 12.0, "sdr_tx_gains": {"PAD": 52.0}})
    assert merged["sdr_antenna"] == "BAND1"  # env fills what params omit
    assert merged["sdr_gain_db"] == 12.0  # per-pass wins
    assert merged["sdr_gains"] == {"PAD": 52.0}  # per-pass staging wins whole


def test_tx_merge_env_only(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("GS_SDR_TX_GAINS", "PAD=40,IAMP=6")
    merged = merge_sdr_params_tx(None)
    assert merged == {"sdr_gains": {"PAD": 40.0, "IAMP": 6.0}}


# ------------------------------------------------------------- stop -> abort


def test_sink_iq_file_path_honors_should_abort(tmp_path: Path) -> None:
    cap = tmp_path / "tx.cf32"
    args = argparse.Namespace(sample_rate=_SR, sdr_args=f"file:{cap}")
    result = txapp._sink_iq(
        args, np.ones(64, np.complex64), None, should_abort=lambda: True
    )
    assert result.outcome == "cancelled" and result.accepted == 0
    assert not cap.exists()  # nothing radiated after the stop


def test_bidir_file_io_honors_should_abort(tmp_path: Path) -> None:
    tx_path = tmp_path / "uplink.cf32"
    io = bidir.FileBidirIo(None, str(tx_path))
    result = io.transmit_burst(
        np.ones(32, np.complex64), should_abort=lambda: True
    )
    assert result.outcome == "cancelled" and result.accepted == 0
    assert not tx_path.exists()
    assert io.sent_samples == 0


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


def test_tx_controller_threads_should_abort_to_the_burst(tmp_path: Path) -> None:
    # End to end through the round-10 two-step handshake: the burst is STAGED while the station is
    # still cold, and only then keyed + sent. With the app's stop already requested, the burst is
    # cancelled and transmit_complete says so — truthful events (R-16), zero samples, cancelled.
    io = bidir.FileBidirIo(None, str(tmp_path / "uplink.cf32"))
    tx = bidir._TxController(io, sample_rate=96_000.0, should_abort=lambda: True)
    socks = _FakeSockets()

    async def _stage_then_burst() -> int:
        assert await tx.prepare(socks, "f1", b"payload", 96_000.0, {})  # 10 sps @9k6
        return await tx.transmit(socks, "f1")

    accepted = asyncio.run(_stage_then_burst())
    assert accepted == 0
    events = [
        json.loads(line)
        for line in socks.status_writer.buf.decode().splitlines()
        if line.strip()
    ]
    complete = next(e for e in events if e["event"] == "transmit_complete")
    assert complete["outcome"] == "cancelled" and complete["samples"] == 0
    assert not any(e["event"] == "transmit_started" for e in events)

"""X-01 (R-11 + R-21) regressions: 'ready' is proof, not process startup.

R-11: an RX engine emits ready only after its source delivered first samples;
a source that cannot open / stays silent fails the pass closed (EngineFailure /
error event + nonzero exit). An engine that dies mid-pass fails the pass too.
R-21: applied front-end settings are observable — requested vs applied vs
read-back — and correction failures are reported, never silently suppressed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cubesat_gfsk_ax25_rx as rxapp
import cubesat_gfsk_ax25_tx as txapp
import numpy as np
import pytest
from _recorder import StreamRecorder, first_sample_probe
from _soapy import apply_corrections, readback_soapy_settings, sdr_ready_fields
from _spawn_contract import (
    EngineFailure,
    await_first_samples,
    run_command_loop,
    send_event,
    watch_engine_death,
)

from gfsk_ax25 import ax25, endurosat

_SR = 99_840


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
    return [
        json.loads(line)
        for line in socks.status_writer.buf.decode().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------- R-21 helpers


class _GrEndpoint:
    """gr-soapy-shaped endpoint: snake_case getters, (channel) signature."""

    def get_antenna(self, ch: int) -> str:
        return "LNAW"

    def get_gain(self, ch: int) -> float:
        return 41.5

    def get_sample_rate(self, ch: int) -> float:
        return 2_048_000.0

    def set_frequency_correction(self, ch: int, ppm: float) -> None:
        return None

    def set_dc_offset_mode(self, ch: int, on: bool) -> None:
        raise RuntimeError("driver refuses")


class _NativeEndpoint:
    """SoapySDR.Device-shaped endpoint: camelCase getters, (direction, channel)."""

    def getAntenna(self, direction: int, ch: int) -> str:  # noqa: N802
        return f"TX/{direction}"

    def getGain(self, direction: int, ch: int) -> float:  # noqa: N802
        return 20.0


def test_readback_reads_gr_soapy_getters_and_lists_unreadable_keys() -> None:
    actual = readback_soapy_settings(_GrEndpoint())
    assert actual["antenna"] == "LNAW"
    assert actual["gain_db"] == 41.5
    assert actual["sample_rate_hz"] == 2_048_000.0
    # Keys with no working getter are NAMED, not silently absent (R-21).
    for missing in ("agc", "bandwidth_hz", "frequency_hz", "ppm", "dc_removal"):
        assert missing in actual["unreadable"]


def test_readback_native_device_needs_direction() -> None:
    # Without a direction the camelCase getters are skipped (they need the
    # direction constant) — everything lands in unreadable.
    blind = readback_soapy_settings(_NativeEndpoint())
    assert "antenna" in blind["unreadable"]
    actual = readback_soapy_settings(_NativeEndpoint(), direction=1)
    assert actual["antenna"] == "TX/1"
    assert actual["gain_db"] == 20.0


def test_apply_corrections_reports_failures_instead_of_suppressing() -> None:
    ep = _GrEndpoint()
    report = apply_corrections(ep, ppm=2.5, dc_removal=True)
    assert report["ppm"] == 2.5  # applied
    assert "dc_removal_error" in report  # refused — recorded, not swallowed
    assert "dc_removal" not in report


def test_sdr_ready_fields_shape() -> None:
    fields = sdr_ready_fields(
        device="driver=xtrx,serial=abc",
        requested={"sdr_gain_db": 40},
        applied={"gain_db": 40.0},
        actual={"gain_db": 39.5},
        stream_active=True,
        first_samples=None,
    )
    assert fields["sdr"]["device"] == "driver=xtrx,serial=abc"
    assert fields["sdr"]["requested"] == {"sdr_gain_db": 40}
    assert fields["stream_active"] is True
    assert fields["first_samples"] is None  # proof unavailable is stated


# ---------------------------------------------------------------- R-11 helpers


def test_await_first_samples_true_once_probe_goes_positive() -> None:
    counts = iter([0, 0, 3])

    async def run() -> bool:
        return await await_first_samples(
            lambda: next(counts, 3), timeout_s=2.0, poll_s=0.01
        )

    assert asyncio.run(run()) is True


def test_await_first_samples_false_on_timeout_and_survives_probe_errors() -> None:
    def bad_probe() -> int:
        raise OSError("stat failed")

    async def run() -> bool:
        return await await_first_samples(bad_probe, timeout_s=0.05, poll_s=0.01)

    assert asyncio.run(run()) is False


def test_first_sample_probe_reads_recorder_cf32_size(tmp_path: Path) -> None:
    assert first_sample_probe(None) is None
    args = argparse.Namespace(
        record_iq=True, record_formats="sdf", output_dir=str(tmp_path / "p"),
        center_freq_hz=401_000_000,
    )
    rec = StreamRecorder.maybe_start(args, sample_rate_hz=_SR)
    assert rec is not None
    probe = first_sample_probe(rec)
    assert probe is not None
    assert probe() == 0
    rec.write(np.ones(16, dtype=np.complex64))
    rec.close()
    assert probe() == 16 * 8


def test_watch_engine_death_emits_error_and_feeds_eof() -> None:
    async def run() -> tuple[list[dict], str]:
        writer = _FakeWriter()
        reader = asyncio.StreamReader()
        stop_requested = asyncio.Event()

        async def dying_engine() -> None:
            raise RuntimeError("SDR exploded")

        task = asyncio.create_task(dying_engine())
        watch_engine_death(task, writer, reader, stop_requested)  # type: ignore[arg-type]
        # The command loop must exit on the transport-loss path ("eof"), which
        # the app turns into a nonzero exit (P0-08) — the pass FAILS.
        reason = await asyncio.wait_for(
            run_command_loop(reader, {}, writer), timeout=5.0  # type: ignore[arg-type]
        )
        events = [
            json.loads(line)
            for line in writer.buf.decode().splitlines()
            if line.strip()
        ]
        return events, reason

    events, reason = asyncio.run(run())
    assert reason == "eof"
    assert any(
        e["event"] == "error" and e["code"] == "engine-died" and "SDR exploded" in e["detail"]
        for e in events
    )


def test_watch_engine_death_stays_quiet_on_requested_stop() -> None:
    async def run() -> bytes:
        writer = _FakeWriter()
        reader = asyncio.StreamReader()
        stop_requested = asyncio.Event()
        stop_requested.set()  # a requested stop may interrupt the engine mid-raise

        async def dying_engine() -> None:
            raise RuntimeError("teardown race")

        task = asyncio.create_task(dying_engine())
        watch_engine_death(task, writer, reader, stop_requested)  # type: ignore[arg-type]
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0.05)  # give a (wrong) error event a chance to appear
        return bytes(writer.buf)

    assert asyncio.run(run()) == b""


# ---------------------------------------------------------------- dsp RX engine


def _rx_args(cap: Path) -> argparse.Namespace:
    return argparse.Namespace(
        sample_rate=_SR,
        sdr_args=f"file:{cap}",
        center_freq_hz=endurosat.CENTER_FREQUENCY_HZ,
    )


def test_dsp_ready_carries_first_sample_proof_and_identity(tmp_path: Path) -> None:
    cap = tmp_path / "capture.cf32"
    body = ax25.encode_ui(dest="DSN0", src="ES1", info=b"x01-ready")
    profile = endurosat.LinkProfile()
    iq = endurosat.transmit(body, _SR, profile=profile)
    txapp._sink_iq(argparse.Namespace(sample_rate=_SR, sdr_args=f"file:{cap}"), iq)

    socks = _FakeSockets()
    started, stop = asyncio.Event(), asyncio.Event()
    started.set()
    asyncio.run(
        rxapp._run_dsp_engine(_rx_args(cap), socks, {}, started, stop, profile, {"hz": 0.0})
    )
    events = _events(socks)
    ready = next(e for e in events if e["event"] == "ready")
    # R-11: ready went out only after the source proved it delivers samples.
    assert ready["first_samples"] is True
    assert ready["stream_active"] is True
    # R-21: device identity rides in the ready event.
    assert ready["sdr"]["device"] == f"file:{cap}"
    # The proof chunk is data, not a throwaway: the frame still decodes.
    assert any(e["event"] == "frame_received" for e in events)


def test_dsp_engine_fails_closed_when_source_is_empty(tmp_path: Path) -> None:
    cap = tmp_path / "empty.cf32"
    cap.write_bytes(b"")  # opens fine, delivers nothing, ends immediately
    socks = _FakeSockets()
    started, stop = asyncio.Event(), asyncio.Event()
    started.set()
    with pytest.raises(EngineFailure):
        asyncio.run(
            rxapp._run_dsp_engine(
                _rx_args(cap), socks, {}, started, stop, endurosat.LinkProfile(), {"hz": 0.0}
            )
        )
    assert not any(e["event"] == "ready" for e in _events(socks))  # never claimed ready


# ---------------------------------------------------------------- TX readiness


def test_tx_spawn_probe_skips_non_hardware_sinks(tmp_path: Path) -> None:
    ok, fields = txapp._tx_spawn_probe(
        argparse.Namespace(sdr_args=f"file:{tmp_path/'tx.cf32'}"), {}
    )
    assert ok is True
    assert fields["tx_ready"] == "non-hardware-sink"
    assert fields["first_samples"] is None  # nothing implied verified


def test_tx_spawn_probe_fails_closed_on_unopenable_hardware() -> None:
    # No SoapySDR on this host — exactly the configured-but-unopenable case:
    # the probe must report failure (→ error event + exit 1), never ready.
    ok, fields = txapp._tx_spawn_probe(
        argparse.Namespace(sdr_args="driver=xtrx", sample_rate=_SR, center_freq_hz=401_500_000),
        {},
    )
    assert ok is False
    assert fields["code"] == "tx-device-probe-failed"


# ---------------------------------------------------------------- send_event passthrough


def test_ready_fields_are_json_serializable() -> None:
    async def run() -> None:
        w = _FakeWriter()
        await send_event(
            w,  # type: ignore[arg-type]
            {
                "event": "ready",
                **sdr_ready_fields(
                    device="d", requested={"a": 1}, applied={}, actual={"unreadable": ["x"]},
                    stream_active=True, first_samples=True,
                ),
            },
        )
        assert json.loads(bytes(w.buf).decode())["sdr"]["actual"]["unreadable"] == ["x"]

    asyncio.run(run())

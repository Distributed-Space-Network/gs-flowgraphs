"""X-02 sub-slice B regressions (R-17/R-18/R-19): every app's frame_received
carries id + crc_ok (the orchestrator's parser counted id/crc-less frames
invalid), signal telemetry is never fabricated and says what it measures, and
the FM RX rate plan refuses rates that mislabel the audio product."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cubesat_gfsk_ax25_rx as rxapp
import cubesat_gfsk_endurosat_bidir as bidir
import pytest
import satellite_rx
from _rateplan import AUDIO_RATE_HZ, fm_rx_plan
from _spawn_contract import frame_received_event

from gfsk_ax25 import ax25

_APPS = Path(__file__).resolve().parents[1] / "apps"


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


# ---------------------------------------------------------------- builder


def test_frame_builder_carries_id_and_crc_ok() -> None:
    e1 = frame_received_event(b"abc", crc_ok=True, framing="endurosat")
    e2 = frame_received_event(b"def", crc_ok=False)
    assert e1["event"] == "frame_received" and e1["framing"] == "endurosat"
    assert e1["frame"]["crc_ok"] is True and e2["frame"]["crc_ok"] is False
    assert e1["frame"]["id"] and e2["frame"]["id"]
    assert e1["frame"]["id"] != e2["frame"]["id"]  # unique per frame
    assert e1["frame"]["len"] == 3
    ext = frame_received_event(b"x", crc_ok=True, extra_frame_fields={"dest": "DSN0"})
    assert ext["frame"]["dest"] == "DSN0"


# ------------------------------------------------- contract: every emitter


def test_cubesat_emitter_contract() -> None:
    body = ax25.encode_ui(dest="DSN0", src="ES1", info=b"telemetry")
    socks = _FakeSockets()
    asyncio.run(rxapp._emit_frame(socks, body))
    frame = _events(socks)[0]["frame"]
    assert frame["crc_ok"] is True and frame["id"]
    assert frame["dest"] == "DSN0" and frame["src"] == "ES1"  # extras preserved


def test_bidir_emitter_contract() -> None:
    socks = _FakeSockets()
    asyncio.run(bidir.emit_frame(socks, b"\x01\x02payload"))
    evt = _events(socks)[0]
    assert evt["framing"] == "endurosat"
    assert evt["frame"]["crc_ok"] is True and evt["frame"]["id"]
    assert bytes(socks.data_writer.buf) == b"\x01\x02payload"


def test_satellite_rx_emitter_contract(tmp_path: Path) -> None:
    """THE R-18 repro: this app's frames had no id/crc_ok, so gs-client
    counted every decoded frame invalid."""
    socks = _FakeSockets()
    asyncio.run(
        satellite_rx._emit_frame(
            socks, b"FRAMEBYTES", "CUTE-1", decoder="gr-satellites",
            output_dir=str(tmp_path),
        )
    )
    evt = _events(socks)[0]
    assert evt["frame"]["crc_ok"] is True
    assert evt["frame"]["id"]
    assert evt["decoder"] == "gr-satellites" and evt["satellite"] == "CUTE-1"


def test_satellite_rx_preserves_native_metadata_without_fake_utc(tmp_path: Path) -> None:
    socks = _FakeSockets()
    asyncio.run(
        satellite_rx._emit_frame(
            socks,
            b"NATIVE",
            "26390",
            decoder="gfsk9600",
            output_dir=str(tmp_path),
            framing="tt64",
            source_start=123,
            source_end=651,
            source_offset_kind="demodulated_symbol",
            integrity="passed",
            polarity="inverted",
            sync_distance=1.0,
            corrected_symbols=2,
        )
    )
    evt = _events(socks)[0]
    assert evt["frame"]["crc_ok"] is True
    assert evt["framing"] == "tt64"
    assert (evt["source_start"], evt["source_end"]) == (123, 651)
    assert evt["corrected_symbols"] == 2
    record = json.loads((tmp_path / "frames.jsonl").read_text(encoding="utf-8"))
    assert record["timestamp_status"] == "unavailable"
    assert "ts" not in record and "timestamp" not in record


def test_satellite_rx_no_integrity_profile_is_not_crc_ok(tmp_path: Path) -> None:
    socks = _FakeSockets()
    asyncio.run(
        satellite_rx._emit_frame(
            socks,
            b"SYNC-ONLY",
            "44832",
            decoder="gfsk5000",
            output_dir=str(tmp_path),
            framing="smogp_signalling",
            integrity="not_present",
        )
    )
    assert _events(socks)[0]["frame"]["crc_ok"] is False


# ----------------------------------------------------------- signal truth


def test_bidir_signal_declares_source_and_calibration() -> None:
    socks = _FakeSockets()
    asyncio.run(bidir.emit_signal(socks, -73.24))
    evt = _events(socks)[0]
    assert evt["event"] == "signal"
    assert evt["lock"] is False  # never fabricated
    assert evt["source"] == "iq-power"
    assert evt["calibrated"] is False  # dBFS-style level, says so


def test_fm_rx_no_longer_fabricates_telemetry() -> None:
    # Bench-only module (imports gnuradio) — source-level lock: the sine-wave
    # RSSI / constant SNR / fabricated lock must be gone, replaced by the
    # measured audio level with explicit source/calibration flags.
    src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
    assert "math.sin(now)" not in src
    assert '"lock": True' not in src
    assert '"source": "audio-power"' in src
    assert '"calibrated": False' in src
    assert "last_power" in src  # measured, not synthesized


# ------------------------------------------------------------- rate plans


def test_fm_rx_plan_accepts_exact_48k_chains() -> None:
    assert fm_rx_plan(192_000) == (1, 192_000, 4)
    assert fm_rx_plan(96_000) == (1, 96_000, 2)
    assert fm_rx_plan(48_000) == (1, 48_000, 1)
    assert fm_rx_plan(384_000) == (2, 192_000, 4)


def test_fm_rx_plan_rejects_mislabeled_audio_rates() -> None:
    """THE R-19 repro: 1 MHz produced 50 kHz audio advertised as 48 kHz."""
    with pytest.raises(ValueError, match="50000 Hz audio"):
        fm_rx_plan(1_000_000)
    with pytest.raises(ValueError, match="audio"):
        fm_rx_plan(2_048_000)  # 51.2 kHz audio — also mislabeled
    with pytest.raises(ValueError):
        fm_rx_plan(0)
    # The app wires the plan + fails closed with an error event (source lock —
    # the module imports gnuradio and cannot run here).
    src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
    assert "fm_rx_plan(args.sample_rate)" in src
    assert "rate-plan-invalid" in src


def test_fm_advertised_audio_rate_is_the_plan_constant() -> None:
    src = (_APPS / "amateur_fm_narrowband_rx.py").read_text(encoding="utf-8")
    assert "_AUDIO_RATE_HZ = 48_000" in src
    assert AUDIO_RATE_HZ == 48_000

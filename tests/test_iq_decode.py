"""Tests for the post-pass decode tool (apps/iq_decode.py).

Covers the Doppler de-rotation (the crux — the recorded .cf32 is raw/pre-NCO), a full CCSDS
round-trip through a SIMULATED Doppler offset re-corrected by a matching gs-orbitd track, and the
safety property that a non-CCSDS (EnduroSat) capture yields ZERO false ccsds frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import FrameResult, IntegrityStatus, Polarity, SymbolInput
from native_framing.crc import CRC16_CC11XX
from native_framing.linecode import pn9_bytes
from native_framing.profiles.geoscan import SYNCWORD as GEOSCAN_SYNCWORD

from gfsk_ax25 import ccsds, gfsk
from gfsk_ax25 import endurosat_link as el

_FS = 96_000.0
_SYM = 9600.0
_H = ccsds.TMHeader(
    version=0,
    spacecraft_id=0x2AB,
    virtual_channel_id=3,
    ocf_flag=0,
    master_channel_frame_count=42,
    virtual_channel_frame_count=7,
    secondary_header_flag=0,
    sync_flag=0,
    first_header_pointer=0,
)


def _modulate_bits(bits: np.ndarray) -> np.ndarray:
    gp = gfsk.GfskParams(sample_rate_hz=_FS, symbol_rate_hz=_SYM)
    return gfsk.modulate(np.asarray(bits, np.uint8), gp)


def _apply_offset(iq: np.ndarray, f_hz: float) -> np.ndarray:
    n = np.arange(len(iq))
    return (np.asarray(iq, np.complex64) * np.exp(2j * np.pi * f_hz * n / _FS)).astype(np.complex64)


def _write_cf32(tmp_path: Path, iq: np.ndarray) -> Path:
    p = tmp_path / "cap.cf32"
    np.asarray(iq, np.complex64).tofile(p)
    return p


def _guard(iq: np.ndarray, n: int = 2000) -> np.ndarray:
    z = np.zeros(n, np.complex64)
    return np.concatenate([z, np.asarray(iq, np.complex64), z]).astype(np.complex64)


def test_derotate_doppler_returns_tone_to_dc():
    n = 8192
    f0 = 5000.0
    shifted = _apply_offset(np.ones(n, np.complex64), f0)
    out = iq_decode._derotate_doppler(shifted, _FS, [(0.0, f0), (n / _FS, f0)])
    spec = np.abs(np.fft.fft(out))
    peak_hz = np.fft.fftfreq(n, 1.0 / _FS)[int(np.argmax(spec))]
    assert abs(peak_hz) < 2.0 * _FS / n  # back at DC after de-rotation


def test_derotate_doppler_no_track_is_identity():
    iq = _apply_offset(np.ones(1024, np.complex64), 3000.0)
    assert np.array_equal(iq_decode._derotate_doppler(iq, _FS, []), iq)


def test_load_track_reads_nonempty_json_file(tmp_path: Path) -> None:
    track_path = tmp_path / "doppler.json"
    track_path.write_text("[[0,125.5],[1.25,-80]]", encoding="utf-8")
    assert iq_decode._load_track(str(track_path)) == [(0.0, 125.5), (1.25, -80.0)]


def test_decode_capture_ccsds_roundtrip_through_doppler(tmp_path):
    # A CCSDS TM frame under a DRIFTING Doppler (a +28 kHz → −28 kHz linear sweep across the
    # capture) — this is what the gs-orbitd track is FOR. A single per-window CFO estimate cannot
    # follow a carrier that sweeps 56 kHz mid-window (the discriminator sees a moving tone → no
    # frame), so only the matching track de-rotates the sweep back to DC and decodes it. Makes the
    # test non-vacuous: no track → 0 frames; matching track → the frame.
    data = bytes(range(100))
    base = _guard(_modulate_bits(ccsds.build_tm_frame(_H, data)))
    n = np.arange(len(base))
    f_of_n = 28_000.0 - 56_000.0 * (n / len(base))  # linear Doppler chirp, Hz
    rx = (base * np.exp(1j * 2 * np.pi * np.cumsum(f_of_n) / _FS)).astype(np.complex64)
    cap = _write_cf32(tmp_path, rx)
    dur = len(rx) / _FS

    # No track → a single per-window CFO can't track the 56 kHz sweep → nothing decodes.
    assert iq_decode.decode_capture(
        cap, sample_rate_hz=_FS, symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",), doppler_track=None,
    ) == []

    # Matching track sampled along the chirp → de-rotates it → decodes.
    track = [(i * dur / 40.0, 28_000.0 - 56_000.0 * (i / 40.0)) for i in range(41)]
    recs = iq_decode.decode_capture(
        cap, sample_rate_hz=_FS, symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",), doppler_track=track,
        capture_start_unix_s=1_767_225_600.0,
    )
    assert len(recs) == 1
    assert recs[0]["framing"] == "ccsds_tm"
    assert recs[0]["post_pass"] is True
    assert recs[0]["source_sample_offset"] == 0
    assert recs[0]["source_offset_kind"] == "window_start"
    assert recs[0]["ts"] == 1_767_225_600.0
    assert recs[0]["timestamp"] == "2026-01-01T00:00:00Z"
    assert bytes.fromhex(recs[0]["payload_hex"])[6 : 6 + 100] == data
    # ...and appended to the pass frames.jsonl, tagged post_pass.
    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["post_pass"] is True


def test_decode_capture_no_false_positives_on_endurosat(tmp_path):
    # An EnduroSat (light-framing) capture must yield ZERO ccsds_tm frames — the RS/FECF gate
    # rejects non-CCSDS bits, so the post-pass sweep never emits garbage (the "no-spam" property).
    cap = _write_cf32(tmp_path, _guard(el.transmit(bytes(range(24)), _FS)))
    recs = iq_decode.decode_capture(
        cap,
        sample_rate_hz=_FS,
        symbol_rate_hz=_SYM,
        framings_to_try=("ccsds_tm",),
        doppler_track=[(0.0, 0.0)],
    )
    assert recs == []
    assert not (tmp_path / "frames.jsonl").exists()


def test_decode_capture_routes_native_geoscan_and_derives_source_time(tmp_path):
    payload = bytes(range(64))
    decoded = CRC16_CC11XX.append(payload, byteorder="big")
    wire = pn9_bytes(decoded)
    sync = np.fromiter((char == "1" for char in GEOSCAN_SYNCWORD), dtype=np.uint8)
    bits = np.concatenate((sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))
    cap = _write_cf32(tmp_path, _guard(_modulate_bits(bits)))
    assert iq_decode.decode_capture(
        cap,
        sample_rate_hz=_FS,
        symbol_rate_hz=_SYM,
        framings_to_try=("GEOSCAN",),
        doppler_track=[(0.0, 0.0)],
    ) == []
    recs = iq_decode.decode_capture(
        cap,
        sample_rate_hz=_FS,
        symbol_rate_hz=_SYM,
        framings_to_try=("GEOSCAN",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in recs] == [payload]
    assert recs[0]["framing"] == "geoscan"
    assert recs[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert recs[0]["source_sample_offset"] >= 0
    assert recs[0]["timestamp"].startswith("2026-01-01T00:00:")


def test_decode_capture_fractional_sps_iq_vector(tmp_path) -> None:
    from scipy.signal import resample_poly

    capture_rate = 44_100.0
    symbol_rate = 1_200.0  # 36.75 capture samples/symbol
    payload = bytes(range(64))
    decoded = CRC16_CC11XX.append(payload, byteorder="big")
    wire = pn9_bytes(decoded)
    sync = np.fromiter((char == "1" for char in GEOSCAN_SYNCWORD), dtype=np.uint8)
    bits = np.concatenate((sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))

    # The repository modulator deliberately requires integer SPS. Generate at 48 kHz/1200
    # (40 SPS), then resample the IQ by 147/160 to a real fractional-SPS capture clock.
    source = gfsk.modulate(bits, gfsk.GfskParams(sample_rate_hz=48_000, symbol_rate_hz=symbol_rate))
    fractional_iq = resample_poly(source, 147, 160).astype(np.complex64)
    cap = _write_cf32(tmp_path, _guard(fractional_iq, n=2_000))

    records = iq_decode.decode_capture(
        cap,
        sample_rate_hz=capture_rate,
        symbol_rate_hz=symbol_rate,
        framings_to_try=("GEOSCAN",),
        doppler_track=[(0.0, 0.0)],
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [payload]
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["source_sample_offset"] >= 0


def test_postpass_adapts_hard_decisions_to_declared_symbol_convention() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    np.testing.assert_array_equal(
        iq_decode._symbols_for_profile(bits, SymbolInput.HARD_BITS), bits
    )
    np.testing.assert_array_equal(
        iq_decode._symbols_for_profile(bits, SymbolInput.SOFT_SYMBOLS),
        np.array([-1.0, 1.0, 1.0, -1.0]),
    )


def test_exact_live_replay_decodes_fast_symbol_chunks_without_symbol_queue() -> None:
    class _Decoder:
        def __init__(self) -> None:
            self.symbols = 0

        def push(self, symbols):
            self.symbols += len(symbols)
            return []

        def flush(self):
            return [
                FrameResult(
                    canonical_framing="usp",
                    payload=b"complete-frame",
                    integrity=IntegrityStatus.PASSED,
                    source_start=10,
                    source_end=20,
                    polarity=Polarity.NORMAL,
                )
            ]

    decoder = _Decoder()
    fanout = iq_decode._ReplayDecoderFanout((("USP", decoder),))

    # More than the old 256-item live handoff bound, with no control-thread drain between pushes.
    # Symbols are consumed synchronously and therefore cannot overflow a symbol-chunk queue.
    for _ in range(2_048):
        fanout.push(np.ones(32, dtype=np.float32))

    assert decoder.symbols == 65_536
    assert fanout.drain_results() == []
    flushed = fanout.flush_results()
    assert len(flushed) == 1
    assert flushed[0][0] == "USP"
    assert flushed[0][1].payload == b"complete-frame"


@pytest.mark.parametrize(
    "value",
    ([], [1_200.0], [1_200.0, 2_200.0, 3_200.0], [True, 2_200.0], "1200,2200"),
)
def test_afsk_tone_parameters_fail_closed(value: object) -> None:
    with pytest.raises(ValueError, match="exactly two numeric"):
        iq_decode._afsk_tones({"tones_hz": value})

    assert iq_decode._afsk_tones({}) == (1_200.0, 2_200.0)
    assert iq_decode._afsk_tones({"tones_hz": [1_200, 1_800]}) == (1_200.0, 1_800.0)


def test_fm_discriminator_recovers_audio_without_changing_sample_clock() -> None:
    sample_rate = 48_000.0
    sample = np.arange(480, dtype=np.float64)
    audio = 0.4 * np.sin(2.0 * np.pi * 1_200.0 * sample / sample_rate)
    iq = np.exp(1j * np.cumsum(audio)).astype(np.complex64)

    recovered = iq_decode._fm_discriminator(iq)

    assert recovered.shape == iq.shape
    np.testing.assert_allclose(recovered[1:], audio[1:], atol=1e-7)
    with pytest.raises(ValueError, match="one-dimensional complex IQ"):
        iq_decode._fm_discriminator(np.zeros(16, dtype=np.float64))


def test_decode_capture_uses_exact_fractional_symbol_clock(tmp_path, monkeypatch) -> None:
    cap = _write_cf32(tmp_path, np.zeros(256, dtype=np.complex64))

    class _Profile:
        decoder_available = True
        symbol_input = SymbolInput.SOFT_SYMBOLS
        parameters: dict[str, object] = {}

    class _Decoder:
        def push(self, symbols):
            np.testing.assert_array_equal(symbols, np.array([-1.0, 1.0]))
            return [
                FrameResult(
                    canonical_framing="clock-test",
                    payload=b"clock",
                    integrity=IntegrityStatus.PASSED,
                    source_start=4,
                    source_end=5,
                    polarity=Polarity.NORMAL,
                    sync_distance=2.5,
                    corrected_symbols=3,
                    metadata={"profile_detail": "preserved"},
                )
            ]

        def flush(self):
            return []

    monkeypatch.setattr(
        iq_decode.gfsk,
        "demodulate_capture",
        lambda *args, **kwargs: np.array([0, 1], dtype=np.uint8),
    )
    monkeypatch.setattr(iq_decode, "resolve_profile", lambda name: _Profile())
    monkeypatch.setattr(iq_decode, "build_decoder", lambda name, parameters: _Decoder())

    records = iq_decode.decode_capture(
        cap,
        sample_rate_hz=44_100.0,
        symbol_rate_hz=1_200.0,
        framings_to_try=("SNET",),
        doppler_track=[(0.0, 0.0)],
        native_evaluation=True,
    )
    assert len(records) == 1
    assert records[0]["source_sample_offset"] == 147  # exactly round(4 * 36.75)
    assert records[0]["source_sample_end_offset"] == 184
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    assert records[0]["source_start"] == 4
    assert records[0]["source_end"] == 5
    assert records[0]["integrity"] == "passed"
    assert records[0]["polarity"] == "normal"
    assert records[0]["sync_distance"] == 2.5
    assert records[0]["corrected_symbols"] == 3
    assert records[0]["metadata"] == {"profile_detail": "preserved"}


def test_main_threads_directive_parameter_json_into_decode_capture(tmp_path, monkeypatch) -> None:
    cap = _write_cf32(tmp_path, np.zeros(1, dtype=np.complex64))
    received: dict[str, object] = {}

    def _decode(*args, **kwargs):
        received.update(kwargs)
        return []

    monkeypatch.setattr(iq_decode, "decode_capture", _decode)
    rc = iq_decode.main([
        "--input", str(cap),
        "--sample-rate", str(_FS),
        "--framings", "SMOG-P RA",
        "--framing-parameters-json",
        '{"modulation":"fsk","mod_index":0.8,"frame_size":256.0}',
        "--native-evaluation",
    ])
    assert rc == 0
    assert received["framing_parameters"] == {
        "modulation": "fsk",
        "mod_index": 0.8,
        "frame_size": 256.0,
    }
    assert received["modulation"] is None
    assert received["mod_index"] is None
    assert received["native_evaluation"] is True


def test_main_routes_exact_live_gnuradio_replay_without_doppler(tmp_path, monkeypatch) -> None:
    cap = _write_cf32(tmp_path, np.zeros(1, dtype=np.complex64))
    received: dict[str, object] = {}

    def _decode(*args, **kwargs):
        received.update(kwargs)
        return []

    monkeypatch.setattr(iq_decode, "decode_capture_gnuradio", _decode)
    monkeypatch.setattr(
        iq_decode,
        "decode_capture",
        lambda *args, **kwargs: pytest.fail("portable numpy engine was selected"),
    )
    rc = iq_decode.main([
        "--input", str(cap),
        "--sample-rate", "48000",
        "--symbol-rate", "2400",
        "--framings", "USP",
        "--modulation", "gmsk",
        "--engine", "gnuradio-live",
        "--native-evaluation",
        "--replay-speed", "4",
    ])

    assert rc == 0
    assert received == {
        "sample_rate_hz": 48_000.0,
        "symbol_rate_hz": 2_400.0,
        "framings_to_try": ("usp",),
        "framing_parameters": {},
        "modulation": "gmsk",
        "capture_start_unix_s": 0.0,
        "native_evaluation": True,
        "use_grsatellites": True,
        "replay_speed": 4.0,
        "append_frames": False,
    }


def test_main_rejects_doppler_track_with_exact_live_replay(tmp_path, monkeypatch) -> None:
    cap = _write_cf32(tmp_path, np.zeros(1, dtype=np.complex64))
    track = tmp_path / "doppler.json"
    track.write_text("[[0,0]]", encoding="utf-8")
    called = False

    def _decode(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(iq_decode, "decode_capture_gnuradio", _decode)
    rc = iq_decode.main([
        "--input", str(cap),
        "--sample-rate", "48000",
        "--symbol-rate", "2400",
        "--framings", "USP",
        "--engine", "gnuradio-live",
        "--doppler-track", str(track),
    ])

    assert rc == 1
    assert called is False


def test_main_rejects_non_object_framing_parameter_json(tmp_path, monkeypatch) -> None:
    cap = _write_cf32(tmp_path, np.zeros(1, dtype=np.complex64))
    called = False

    def _decode(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(iq_decode, "decode_capture", _decode)
    rc = iq_decode.main([
        "--input", str(cap),
        "--sample-rate", str(_FS),
        "--framing-parameters-json", "[]",
    ])
    assert rc == 1
    assert not called


def test_main_returns_one_when_append_fails_with_records(tmp_path):
    # CA-FLOW-006: decoded records that cannot be persisted (frames.jsonl append
    # OSError — here the target is a directory; ENOSPC is the field shape) must
    # exit NONZERO. It used to warn and return 0: the frames were silently lost
    # while the pass looked successfully post-processed.
    cap = _write_cf32(
        tmp_path, _guard(_modulate_bits(ccsds.build_tm_frame(_H, bytes(range(50)))))
    )
    (tmp_path / "frames.jsonl").mkdir()  # open("a") raises OSError
    rc = iq_decode.main([
        "--input", str(cap), "--sample-rate", str(_FS), "--symbol-rate", str(_SYM),
        "--framings", "ccsds_tm",
    ])
    assert rc == 1


def test_main_no_records_remains_clean_zero(tmp_path):
    # No decoded records -> the append never runs -> clean zero, even with the
    # same unwritable frames.jsonl target present.
    cap = _write_cf32(tmp_path, _guard(el.transmit(bytes(range(24)), _FS)))
    (tmp_path / "frames.jsonl").mkdir()
    rc = iq_decode.main([
        "--input", str(cap), "--sample-rate", str(_FS), "--symbol-rate", str(_SYM),
        "--framings", "ccsds_tm",
    ])
    assert rc == 0


def test_decode_capture_missing_file_and_empty_framings(tmp_path):
    assert iq_decode.decode_capture(
        tmp_path / "nope.cf32", sample_rate_hz=_FS, framings_to_try=("ccsds_tm",)
    ) == []
    cap = _write_cf32(tmp_path, _guard(_modulate_bits(ccsds.build_tm_frame(_H, b"\x01\x02"))))
    assert iq_decode.decode_capture(cap, sample_rate_hz=_FS, framings_to_try=()) == []

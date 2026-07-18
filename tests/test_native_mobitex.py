"""Upstream parity and construction tests for native Mobitex profiles.

Copyright 2020 Daniel Estevez <daniel@destevez.net>
Copyright 2025 Fabian P. Schmidt <kerel@mailbox.org>
Adapted and extended for gs-flowgraphs native parity testing in 2026.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import (
    FskAudioConfig,
    build_decoder,
    decode_fsk_audio_mm_profile,
    decode_fsk_audio_profile,
    demodulate_fsk_audio,
    demodulate_fsk_audio_mm,
)
from native_framing.codes.mobitex import (
    MobitexFecStatus,
    decode_mobitex_fec,
    encode_mobitex_fec,
)
from native_framing.crc import CRC16_X25, CrcSpec
from native_framing.linecode import additive_randomize_bits
from native_framing.profiles.mobitex import (
    BLOCK_WIRE_BYTES,
    MAX_BLOCKS,
    MAX_CAPTURE_BYTES,
    NX_SYNCWORD,
    SYNCWORD,
    _decode_control,
    _decode_data_block,
    _recover_callsign,
)
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity, SymbolInput
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM = _ROOT / "related-projects/gr-satellites/python/components/deframers"
_CALLSIGN_CRC = CrcSpec("Mobitex callsign CRC", 16, 0x1021, 0, 0, False, False)
_INTERLEAVE = np.arange(240).reshape(12, 20).T.ravel()
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 4_800.0
_BEESAT9_WAV = _ROOT / "related-projects/satellite-recordings/beesat_9.wav"
_BEESAT9_WAV_SHA256 = "900be8482dc8733e82a49394ab106d4612a29857a7a6743679ce78688aeafc0f"
_BEESAT9_FRAME_SHA256 = "658db9f4e540ed73cf9e5d4f653932700ed712a698c6c4ead4354d0c6cf331a7"
_CLASSIC_WAVS = {
    "sokrat.wav": "de062de2ece5d5ca040959947001c9b7c24a1e42235bc813ea8d653b9be56879",
    "dstar_one.wav": "fcb849002e9684f103e3e7a99884a35177004c0452e04602733b4d433f8711c7",
    "amgu_1.wav": "3d6fe482e8bfbc54cc82282477337954a2368b33f20c8b73938ebd6f9d923594",
}
_CLASSIC_FRAMES = {
    "sokrat.wav": [
        (
            "b9eb7bc207a699c11588d0b35a46bf7dd0ce8ef4f2aae77e6671a2db880fefee",
            884,
            2_364,
            8_829,
            23_628,
            0,
        )
    ],
    "dstar_one.wav": [
        (
            "af0d4189b9a7e7505000d8cebe5fe4e4571a52a0dad55d7acf3857d81eb18c81",
            605,
            2_085,
            6_044,
            20_844,
            0,
        ),
        (
            "ed5e01ba465131d6148ffd8d532b06fcdb9a589068a2963a052c5d74ca72cbe8",
            8_811,
            10_291,
            88_045,
            102_849,
            0,
        ),
        (
            "5816b597a9f2a454e4e61d893b42a95fe62fb5c0e50630845385574953ac5b09",
            16_289,
            17_769,
            162_789,
            177_588,
            0,
        ),
    ],
    "amgu_1.wav": [
        (
            "41e87f54f936860dd4f0af6819753899d6508ba20481e6eaf71ea173bfc33fb4",
            1_280,
            2_760,
            12_783,
            27_583,
            0,
        )
    ],
}


def _fsk_capture(bits: np.ndarray, *, mod_index: float = 0.8) -> np.ndarray:
    samples_per_symbol = int(_IQ_SAMPLE_RATE / _IQ_SYMBOL_RATE)
    symbols = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    instantaneous_hz = (
        np.repeat(symbols, samples_per_symbol) * mod_index * _IQ_SYMBOL_RATE / 2.0
    )
    phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / _IQ_SAMPLE_RATE
    burst = np.exp(1j * phase).astype(np.complex64)
    return np.concatenate(
        (np.zeros(2_000, dtype=np.complex64), burst, np.zeros(2_000, dtype=np.complex64))
    )


def _control_fec(control0: int, control1: int) -> int:
    return ((encode_mobitex_fec(control0) & 0xF) << 4) | (
        encode_mobitex_fec(control1) & 0xF
    )


def _encode_block(body: bytes, *, valid_crc: bool = True) -> np.ndarray:
    if len(body) != 18:
        raise ValueError("Mobitex construction body must contain 18 bytes")
    crc = CRC16_X25.compute(body).to_bytes(2, "big")
    if not valid_crc:
        crc = bytes([crc[0] ^ 1, crc[1]])
    codewords = [encode_mobitex_fec(value) for value in body + crc]
    bits = np.concatenate(
        [
            np.fromiter((int(bit) for bit in f"{word:012b}"), dtype=np.uint8)
            for word in codewords
        ]
    )
    randomized = np.empty_like(bits)
    randomized[_INTERLEAVE] = bits
    return randomized


def _stream(
    *,
    nx: bool,
    bodies: tuple[bytes, ...],
    callsign: bytes = b"DP0BEM",
    invalid_blocks: tuple[int, ...] = (),
    fec_error_bit: int | None = None,
    variant: str = "default",
) -> tuple[np.ndarray, bytes]:
    control0 = len(bodies) - 1 if nx else 0x71
    control1 = 2 if nx else len(bodies)
    callsign_crc = _CALLSIGN_CRC.compute(callsign).to_bytes(2, "big")
    header = bytes([control0, control1, _control_fec(control0, control1)])
    if nx and variant != "BEESAT-1":
        header += callsign + callsign_crc
    coded = np.concatenate(
        [
            _encode_block(body, valid_crc=index not in invalid_blocks)
            for index, body in enumerate(bodies)
        ]
    )
    if fec_error_bit is not None:
        # Error injection is made after permutation, in one 12-bit FEC word.
        target = int(np.flatnonzero(fec_error_bit == _INTERLEAVE)[0])
        coded[target] ^= 1
    wire_bits = additive_randomize_bits(
        coded, mask=0x22, seed=0x1FF, register_length=9
    )
    wire = header + bytes(np.packbits(wire_bits, bitorder="big"))
    wire = wire.ljust(MAX_CAPTURE_BYTES, b"\x00")
    syncword = NX_SYNCWORD if nx else SYNCWORD
    logical = np.concatenate(
        (
            np.fromiter((int(bit) for bit in syncword), dtype=np.uint8),
            np.unpackbits(np.frombuffer(wire, dtype=np.uint8)),
        )
    )
    # gr-satellites multiplies the soft stream by -1 before binary slicing.
    soft = 1.0 - 2.0 * logical.astype(np.float64)
    invalid_mask = sum(1 << index for index in invalid_blocks)
    expected_header = bytes([control0, control1])
    if nx and variant != "BEESAT-1":
        expected_header += callsign + callsign_crc
    expected = (
        expected_header
        + b"".join(bodies)
        + b"\xAA"
        + invalid_mask.to_bytes(4, "little")
        + b"\xBB"
    )
    return soft, expected


@pytest.mark.skipif(
    not (_UPSTREAM / "qa_mobitex_deframer_symbols.f32").is_file(),
    reason="pinned gr-satellites QA checkout is unavailable",
)
def test_mobitex_nx_exact_pinned_gr_satellites_binary_qa() -> None:
    symbols = np.fromfile(_UPSTREAM / "qa_mobitex_deframer_symbols.f32", dtype="<f4")
    expected = (_UPSTREAM / "qa_mobitex_deframer_frame.bin").read_bytes()
    decoder = build_decoder(
        "Mobitex-NX",
        {"callsign": "DP0BEM", "callsign_threshold": 0, "sync_threshold": 0},
    )
    frames = decoder.push(symbols)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].source_start == 1639
    assert frames[0].source_end == 9423
    assert frames[0].polarity is Polarity.INVERTED
    assert frames[0].integrity is IntegrityStatus.PASSED


def test_mobitex_beesat9_pinned_real_wav_replays_all_blocks_byte_exactly() -> None:
    assert hashlib.sha256(_BEESAT9_WAV.read_bytes()).hexdigest() == _BEESAT9_WAV_SHA256
    sample_rate, audio = wavfile.read(_BEESAT9_WAV)
    assert sample_rate == 48_000 and audio.ndim == 1

    symbols, decoded = decode_fsk_audio_profile(
        audio,
        FskAudioConfig(sample_rate, 4_800),
        "Mobitex-NX",
        {"variant": "BEESAT-9", "callsign": "DP0BEM"},
        phase_samples=5,
    )
    assert len(decoded) == 1
    frame = decoded[0].frame
    assert hashlib.sha256(frame.payload).hexdigest() == _BEESAT9_FRAME_SHA256
    assert len(frame.payload) == 592
    assert frame.payload[:10] == bytes.fromhex("3F0244503042454D4DF7")
    assert (frame.source_start, frame.source_end) == (2162, 9946)
    assert (decoded[0].source_sample_start, decoded[0].source_sample_end) == (
        21_625,
        99_465,
    )
    assert symbols.sample_offset(frame.source_start) == decoded[0].source_sample_start
    assert frame.corrected_symbols == 42
    assert frame.metadata["num_blocks"] == 32
    assert frame.metadata["valid_blocks"] == 32
    assert frame.metadata["invalid_block_mask"] == 0
    assert frame.metadata["fec_errors_uncorrectable"] == 0


@pytest.mark.parametrize("filename", sorted(_CLASSIC_WAVS))
def test_classic_mobitex_published_wavs_replay_byte_exactly(filename: str) -> None:
    path = _ROOT / "related-projects/satellite-recordings" / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _CLASSIC_WAVS[filename]
    sample_rate, audio = wavfile.read(path)
    assert sample_rate == 48_000 and audio.ndim == 1

    symbols, decoded = decode_fsk_audio_mm_profile(
        audio,
        FskAudioConfig(sample_rate, 4_800),
        "Mobitex",
        cutoff_hz=2_400,
        gain_mu=0.15,
    )

    expected = _CLASSIC_FRAMES[filename]
    assert len(decoded) == len(expected)
    for located, (digest, start, end, sample_start, sample_end, invalid_mask) in zip(
        decoded, expected, strict=True
    ):
        frame = located.frame
        assert hashlib.sha256(frame.payload).hexdigest() == digest
        assert len(frame.payload) == 2 + 6 * 18 + 6
        assert (frame.source_start, frame.source_end) == (start, end)
        assert (located.source_sample_start, located.source_sample_end) == (
            sample_start,
            sample_end,
        )
        assert frame.integrity is (
            IntegrityStatus.PASSED if invalid_mask == 0 else IntegrityStatus.FAILED
        )
        assert frame.metadata["num_blocks"] == 6
        assert frame.metadata["valid_blocks"] == 6 - invalid_mask.bit_count()
        assert frame.metadata["invalid_block_mask"] == invalid_mask
        assert frame.metadata["callsign_bit_errors"] is None
        assert located.source_sample_start == symbols.sample_offset(frame.source_start)
        assert located.source_sample_end == symbols.sample_offset(frame.source_end)


def test_classic_mobitex_historical_timing_preserves_crc_failure() -> None:
    """The old timing chain must not silently bless its damaged D-STAR block."""

    sample_rate, audio = wavfile.read(
        _ROOT / "related-projects/satellite-recordings/dstar_one.wav"
    )
    _, decoded = decode_fsk_audio_mm_profile(
        audio,
        FskAudioConfig(sample_rate, 4_800),
        "Mobitex",
    )

    assert len(decoded) == 3
    middle = decoded[1].frame
    assert hashlib.sha256(middle.payload).hexdigest() == (
        "415fcda688063a74d99ecb4f5a1157e5342aa33133c49cc3e181c4da2aed8109"
    )
    assert middle.integrity is IntegrityStatus.FAILED
    assert middle.metadata["invalid_block_mask"] == 1
    assert middle.metadata["fec_errors_uncorrectable"] == 2


def test_fsk_audio_replay_contract_rejects_invalid_inputs() -> None:
    config = FskAudioConfig(48_000, 4_800)
    with pytest.raises(ValueError, match="one-dimensional real"):
        demodulate_fsk_audio(np.zeros((2, 2)), config)
    with pytest.raises(ValueError, match="one-dimensional real"):
        demodulate_fsk_audio(np.zeros(2, dtype=np.complex64), config)
    with pytest.raises(ValueError, match="finite"):
        demodulate_fsk_audio(np.asarray([0.0, np.nan]), config)
    with pytest.raises(ValueError, match="non-negative"):
        demodulate_fsk_audio(np.zeros(8), config, phase_samples=-1)
    with pytest.raises(ValueError, match="exceeds"):
        demodulate_fsk_audio(np.zeros(8), config, phase_samples=9)
    with pytest.raises(ValueError, match="finite positive"):
        FskAudioConfig("invalid", 4_800)
    with pytest.raises(ValueError, match="1..4096"):
        FskAudioConfig(48_000, 4_800, dc_block_symbols=0)
    with pytest.raises(ValueError, match="below Nyquist"):
        demodulate_fsk_audio_mm(np.zeros(8), config, cutoff_hz=24_000)
    with pytest.raises(ValueError, match="exceeds 65537"):
        demodulate_fsk_audio_mm(np.zeros(8), config, transition_hz=1)


@pytest.mark.parametrize(
    ("nx", "label", "canonical"),
    [
        (False, "Mobitex", "mobitex"),
        (True, "Mobitex-NX", "mobitex_nx"),
    ],
)
def test_mobitex_fsk_file_iq_replay_routes_soft_native_profiles(
    tmp_path: Path, nx: bool, label: str, canonical: str
) -> None:
    soft_stream, expected = _stream(nx=nx, bodies=(bytes(range(18)),))
    logical_bits = np.asarray(soft_stream < 0, dtype=np.uint8)
    capture = _fsk_capture(logical_bits)
    path = tmp_path / f"{canonical}-fsk.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=(label,),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="fsk",
        mod_index=0.8,
        window_s=3.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == canonical
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


@pytest.mark.parametrize("nx,label", [(False, "Mobitex"), (True, "Mobitex NX")])
@pytest.mark.parametrize("invert", [False, True])
def test_mobitex_profiles_chunking_polarity_and_generated_frame(
    nx: bool, label: str, invert: bool
) -> None:
    body = bytes(range(18))
    stream, expected = _stream(nx=nx, bodies=(body,))
    if invert:
        stream = -stream
    parameters = {"sync_threshold": 0}
    if nx:
        parameters["callsign"] = "DP0BEM"
    decoder = build_decoder(label, parameters)
    frames = []
    for start in range(0, stream.size, 43):
        frames += decoder.push(stream[start : start + 43])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].polarity is (Polarity.NORMAL if invert else Polarity.INVERTED)


@pytest.mark.parametrize("nx,label", [(False, "Mobitex"), (True, "Mobitex-NX")])
def test_mobitex_all_dynamic_block_counts_are_exact_and_bounded(
    nx: bool, label: str
) -> None:
    for num_blocks in range(1, MAX_BLOCKS + 1):
        bodies = tuple(bytes([block_id]) * 18 for block_id in range(num_blocks))
        stream, expected = _stream(nx=nx, bodies=bodies)
        parameters = {"sync_threshold": 0}
        if nx:
            parameters["callsign"] = "DP0BEM"
        decoder = build_decoder(label, parameters)
        frames = []
        for start in range(0, stream.size, 127):
            frames += decoder.push(stream[start : start + 127])
            assert decoder.retained_symbols <= decoder.max_retained_symbols

        assert [frame.payload for frame in frames] == [expected]
        assert frames[0].source_start == 0
        header_bytes = 11 if nx else 3
        assert frames[0].source_end == (
            len(SYNCWORD) + (header_bytes + BLOCK_WIRE_BYTES * num_blocks) * 8
        )
        assert frames[0].integrity is IntegrityStatus.PASSED
        assert frames[0].metadata["num_blocks"] == num_blocks
        assert frames[0].metadata["valid_blocks"] == num_blocks
        assert frames[0].metadata["invalid_block_mask"] == 0


def test_mobitex_fec_literal_vectors_and_single_bit_correction() -> None:
    assert encode_mobitex_fec(0x2C) == 0x2C8
    clean = decode_mobitex_fec(0x2C8)
    assert (clean.message, clean.status) == (0x2C, MobitexFecStatus.NO_ERROR)
    for bit in range(12):
        corrected = decode_mobitex_fec(0x2C8 ^ (1 << bit))
        assert (corrected.message, corrected.status) == (
            0x2C,
            MobitexFecStatus.ERROR_CORRECTED,
        )
    # Exact two-bit collision documented by upstream: the syndrome aliases a
    # one-bit error, so the bounded decoder reports a correction to 0x2E.
    collision = decode_mobitex_fec(0x2C8 | 0b0010_0010)
    assert (collision.message, collision.status) == (
        0x2E,
        MobitexFecStatus.ERROR_CORRECTED,
    )


def test_mobitex_exact_upstream_fec_block_and_control_qa_vectors() -> None:
    encoded = bytes.fromhex(
        "1ABCF3FCC1D10B922D18DE0818D0000005DCC130000007D72C8D115FA198"
    )
    expected = bytes.fromhex("1ACFFC1D0B2218E01800005DC000007D2CD15F19")
    decoded, corrected, uncorrectable = _decode_data_block(encoded)
    assert decoded == expected
    assert corrected == 1
    assert uncorrectable == 0
    assert CRC16_X25.strip_if_valid(decoded, byteorder="big") == expected[:-2]

    assert _decode_control(0x3F, 0x02, 0xC6) == (bytes.fromhex("3F02C6"), 0)
    assert _decode_control(0b0011_1011, 0x02, 0xC6) == (
        bytes.fromhex("3F02C6"),
        1,
    )
    assert _decode_control(0b0011_1010, 0x02, 0xC6) is None


def test_mobitex_reports_fec_correction_and_partial_crc_bitmap() -> None:
    bodies = (bytes(range(18)), bytes(range(18, 36)))
    stream, expected = _stream(
        nx=True, bodies=bodies, invalid_blocks=(1,), fec_error_bit=0
    )
    frames = build_decoder("Mobitex-NX", {"callsign": "DP0BEM"}).push(stream)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].integrity is IntegrityStatus.FAILED
    assert frames[0].corrected_symbols == 1
    assert frames[0].metadata["valid_blocks"] == 1
    assert frames[0].metadata["invalid_block_mask"] == 2


def test_mobitex_callsign_gate_unknown_recovery_and_all_invalid_rejection() -> None:
    stream, _ = _stream(nx=True, bodies=(bytes(range(18)),))
    assert len(build_decoder("Mobitex-NX").push(stream)) == 1
    assert build_decoder(
        "Mobitex-NX", {"callsign": "BADBAD", "callsign_threshold": 0}
    ).push(stream) == []
    invalid, _ = _stream(
        nx=True, bodies=(bytes(range(18)),), invalid_blocks=(0,)
    )
    assert build_decoder("Mobitex-NX", {"callsign": "DP0BEM"}).push(invalid) == []
    with pytest.raises(ValueError, match="bounded to two"):
        build_decoder("Mobitex-NX", {"callsign_threshold": 3})
    assert build_decoder(
        "Mobitex-NX", {"callsign": "DP0BEM", "callsign_threshold": 12}
    ).push(stream)

    callsign = b"DP0BEM"
    valid = callsign + _CALLSIGN_CRC.compute(callsign).to_bytes(2, "big")
    for position in range(64):
        damaged = bytearray(valid)
        damaged[position // 8] ^= 1 << (position % 8)
        assert _recover_callsign(bytes(damaged), 1) == (callsign, valid[6:], 1)

    # These two error pairs have the same CRC syndrome. Selecting whichever
    # pair happens to be enumerated first can invent a different valid callsign.
    ambiguous = bytearray(valid)
    for position in (4, 9):
        ambiguous[position // 8] ^= 1 << (position % 8)
    assert _recover_callsign(bytes(ambiguous), 2) is None


def test_mobitex_known_callsign_uses_upstream_default_threshold_of_twelve() -> None:
    stream, expected = _stream(nx=True, bodies=(bytes(range(18)),))
    damaged = stream.copy()
    callsign_start = len(NX_SYNCWORD) + 3 * 8
    for bit in (0, 9, 18):
        damaged[callsign_start + bit] *= -1

    frames = build_decoder(
        "Mobitex-NX", {"callsign": "DP0BEM", "sync_threshold": 0}
    ).push(damaged)
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["callsign_bit_errors"] == 3
    assert build_decoder(
        "Mobitex-NX",
        {
            "callsign": "DP0BEM",
            "callsign_threshold": 2,
            "sync_threshold": 0,
        },
    ).push(damaged) == []


def test_mobitex_nx_beesat_variants_follow_upstream_header_and_block_policy() -> None:
    nx = True
    label = "Mobitex-NX"
    beesat1, expected1 = _stream(
        nx=nx,
        bodies=(bytes(range(18)),),
        variant="BEESAT-1",
    )
    frames1 = build_decoder(
        label, {"variant": "BEESAT-1", "sync_threshold": 0}
    ).push(beesat1)
    assert [frame.payload for frame in frames1] == [expected1]
    assert frames1[0].metadata["callsign_bit_errors"] is None

    bodies = tuple(bytes([index]) * 18 for index in range(32))
    beesat9, expected9 = _stream(nx=nx, bodies=bodies, variant="BEESAT-9")
    frames9 = build_decoder(
        label,
        {
            "variant": "BEESAT-9",
            "callsign": "DP0BEM",
            "sync_threshold": 0,
        },
    ).push(beesat9)
    assert [frame.payload for frame in frames9] == [expected9]
    assert frames9[0].metadata["valid_blocks"] == 32


def test_mobitex_and_nx_syncwords_cross_reject() -> None:
    classic, _ = _stream(nx=False, bodies=(bytes(range(18)),))
    nx, _ = _stream(nx=True, bodies=(bytes(range(18)),))
    assert build_decoder("Mobitex", {"sync_threshold": 0}).push(nx) == []
    assert build_decoder(
        "Mobitex-NX", {"callsign": "DP0BEM", "sync_threshold": 0}
    ).push(classic) == []


def test_mobitex_truncation_validation_and_registry_contracts() -> None:
    stream, _ = _stream(nx=True, bodies=(bytes(range(18)),))
    decoder = build_decoder("Mobitex-NX", {"callsign": "DP0BEM"})
    actual_symbols = len(NX_SYNCWORD) + (11 + BLOCK_WIRE_BYTES) * 8
    assert decoder.push(stream[: actual_symbols - 1]) == []
    assert decoder.flush() == []
    for label in ("Mobitex", "Mobitex-NX"):
        profile = REGISTRY.resolve(label)
        assert profile is not None
        assert profile.disposition is DecodeDisposition.IN_PROGRESS
        assert profile.symbol_input is SymbolInput.SOFT_SYMBOLS
        assert profile.decoder_available
    with pytest.raises(ValueError, match="six ASCII"):
        build_decoder("Mobitex-NX", {"callsign": "short"})
    with pytest.raises(ValueError, match="only ASCII"):
        build_decoder("Mobitex-NX", {"callsign": "DØ0BEM"})
    with pytest.raises(ValueError, match="<= 16"):
        build_decoder("Mobitex-NX", {"sync_threshold": 17})
    with pytest.raises(ValueError, match="variant must be one of"):
        build_decoder("Mobitex-NX", {"variant": "unknown"})
    with pytest.raises(ValueError, match="unknown parameters"):
        build_decoder("Mobitex", {"variant": "BEESAT-1"})
    with pytest.raises(ValueError, match="unknown parameters"):
        build_decoder("Mobitex", {"callsign": "DP0BEM"})
    with pytest.raises(ValueError, match="unknown parameters"):
        build_decoder("Mobitex", {"callsign_threshold": 2})
    with pytest.raises(ValueError, match=">= 0"):
        build_decoder(
            "Mobitex-NX", {"callsign": "DP0BEM", "callsign_threshold": -1}
        )
    with pytest.raises(ValueError, match="<= 64"):
        build_decoder(
            "Mobitex-NX", {"callsign": "DP0BEM", "callsign_threshold": 65}
        )

"""Pinned gr-satellites parity and construction tests for native USP."""

# Copyright 2021 Daniel Estevez <daniel@destevez.net>
# Adapted and extended for gs-flowgraphs native USP parity testing in 2026.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.linecode import ccsds_randomize
from native_framing.profiles.usp import CAPTURE_SYMBOLS, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, IntegrityStatus, Polarity
from native_framing.viterbi import ConvolutionalCode

_SYNC_BYTES = bytes([80, 114, 246, 75, 45, 144, 177, 245])
_SYNC_SOFT = 2.0 * np.unpackbits(np.frombuffer(_SYNC_BYTES, dtype=np.uint8)) - 1.0
_PLS = (
    "0111000110011101100000111100100101010011010000100010110111111010",
    "0010010011001000110101101001110000000110000101110111100010101111",
)
_RS = CcsdsReedSolomon(basis="dual", interleaving=1)
_VITERBI = ConvolutionalCode("CCSDS")
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 4_800.0


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

# Exact encoded fixtures from gr-satellites qa_usp_deframer.py at the pinned commit.
_QA_LONG = bytes.fromhex(
    "24c8d69c061778af8cf58d8257a5368e58c2fa4fecbd1a64955b917797cea809"
    "f5263e74a121964c8a0811eccf088fbc309f887f77d38fb2cf770759c7094716"
    "a5ce3659e8ee9ac780692335263ead3ae453210812bc1f43464dc6c28be2277c"
    "4ffbf609505ede511c3c900e66e158aa0c5bd5da1875f9cb3dd2eca7f302db5a"
    "9738d967a3ba6b1e01a48cd498fab4eb914c84204af07d0d19371b0a2cd6bf1f"
    "c710cb9508ae9105be3bfb935049ad3a06d8625e787a9db59c480eefc2c16a6a"
    "8575d6f6f254cec0c98a95cbe44aa3745af13c686f25793849c88fb19f9277c"
    "4ffbf609505ede511c3c900e66e158aa0c5bd5da1875f9cb3dd2eca7f302db5a"
    "9738d967a3ba6b1e01a48cd498fab4eb914c84204af07d0d19371b0a2f889df1"
    "3fefd825417b794470f240399b8562a8316f576861d7e72cf74bb29fcc0b6d6a"
    "5ce3659e8ee9ac780692335263ead3ae453210812bc1f43464dc6c28be2277c4f"
    "fbf609505ede511c3c900e66e158aa0c5bd5da1875f9cb3dd2eca7f302db5a97"
    "38d967a3ba6b1e01a48cd498fab4eb914c84204af07d0d19371b0a2f889df13f"
    "efd825417b794470f240399b8562a8316f576861d7e72cf74bb29fcc0b6d6a5c"
    "e3659e8ee9ac7548a58a95f3a41287010f0858d8fc00a19e322633e300c410a0"
    "dfb53ab40709490b27d937cfe9598d8c936549050170b8c5d807d0eab5a40646"
    "eb3ac805f9f3"
)
_QA_SHORT = bytes.fromhex(
    "719d83c953422dfa8cf58d826c618afe58c2fa4fecbd1a64955b917797cea809"
    "f5263e74a121964c8a0811eccf088fbce93f5049a0a38fb2cf770759fccdfb66"
    "a5ce389d1e1ba19d6f578a30b304b8420a8ac5487f98183fa81dd88d4fda01cc"
    "4ffbf609505ede51cbba35b51553351056424d4cbcb41e524f8a5bf4f72e07c9"
    "5d1e205ed04c7f1f131c7615d946b62d7551c57ef8507267425a041e6ae01a7e"
    "9dd029f1572c20fb9e201efefffff912a0d62d46761f8deedef86367208c8cbb"
    "05a7950ad98b4c14197b072946995e3ce5e2daab037cb5255314ade72b61ff37"
    "9ec8d1b38edbf86d2cdbf78eed31b6ab5f34276ea51f7b31b5eb6e7423dc5abf"
    "45415364add59d59f22ee539d6e8deb56bda8b843db1569f514ac99dc6347aaa"
    "07e4bc43433dc59b5bf18b585453bb59c6725731e383af4dd4c5684e6e7a1536"
    "9e3a46d04bc8ae7d1bc4f4eb658749ba6f9a5190229e1295ecb4424a85765e4f"
    "4950834f26ea295da956d324b25ccd1bba9edbd23b721d165bd15772ec21e67d"
    "9157568a0add9ac2a2b7b53dbb11bc82e99a838d8f90141bdac456ca2bb3aa9a"
    "7bd2492ecd4678aa6bf0660f64cb166007e65512672d186a171292db25976945"
    "b5431693569475a1faa2674ee8d842873b2be06b5b17061754e4c4d6d64b7d22"
    "4f332fb2d2594549cab4a553c2962129ec88a937c30b5e58c5269cc2aea4c44a"
    "7accc0741d09"
)
_EXPECTED_LONG = bytes.fromhex(
    "a464829c8c4060a4a66060a6406f00f016420200010042000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000000000001b1bff671f20250eaab14060f43c01002400f01c"
)
_EXPECTED_SHORT = bytes.fromhex(
    "a464829c8c4060a4a66060a6406f00f0e1ff020001000300002606"
)


def _qa_stream(wire: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(_SYNC_BYTES + wire, dtype=np.uint8))
    return 2.0 * bits.astype(np.float64) - 1.0


def _constructed_stream(
    data_length: int,
    payload: bytes,
    *,
    rs_errors: int = 0,
    declared_length: int | None = None,
) -> np.ndarray:
    if data_length not in (48, 223):
        raise ValueError("USP data length must be 48 or 223")
    length = len(payload) if declared_length is None else declared_length
    data = (b"\x00\x00" + length.to_bytes(2, "little") + payload).ljust(
        data_length, b"\x00"
    )
    codeword = bytearray(_RS.encode(data))
    for index in range(rs_errors):
        codeword[index * 11] ^= index + 1
    randomized = ccsds_randomize(bytes(codeword))
    encoded = _VITERBI.encode(
        np.unpackbits(np.frombuffer(randomized, dtype=np.uint8)), mode="truncated"
    )
    pls_code = 0 if data_length == 48 else 1
    capture = np.concatenate(
        (
            np.fromiter((int(bit) for bit in _PLS[pls_code]), dtype=np.uint8),
            np.asarray(encoded, dtype=np.uint8),
        )
    )
    capture = np.pad(capture, (0, CAPTURE_SYMBOLS - capture.size))
    return np.concatenate((_SYNC_SOFT, 2.0 * capture - 1.0))


@pytest.mark.parametrize(
    "wire,expected,pls_code,data_length",
    [(_QA_LONG, _EXPECTED_LONG, 1, 223), (_QA_SHORT, _EXPECTED_SHORT, 0, 48)],
)
def test_usp_exact_pinned_gr_satellites_qa_vectors(
    wire: bytes, expected: bytes, pls_code: int, data_length: int
) -> None:
    frames = build_decoder("USP").push(_qa_stream(wire))
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].metadata["pls_code"] == pls_code
    assert frames[0].metadata["data_length"] == data_length
    assert frames[0].integrity is IntegrityStatus.PASSED


@pytest.mark.parametrize(
    ("wire", "expected", "variant"),
    [(_QA_LONG, _EXPECTED_LONG, "long"), (_QA_SHORT, _EXPECTED_SHORT, "short")],
)
def test_usp_pinned_upstream_fsk_file_iq_replay_routes_soft_native_profile(
    tmp_path: Path, wire: bytes, expected: bytes, variant: str
) -> None:
    soft_stream = _qa_stream(wire)
    capture = _fsk_capture(np.asarray(soft_stream >= 0, dtype=np.uint8))
    path = tmp_path / f"usp-{variant}-fsk.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=("USP",),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation="fsk",
        mod_index=0.8,
        window_s=2.0,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [expected]
    assert records[0]["framing"] == "usp"
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


@pytest.mark.parametrize("data_length", [48, 223])
@pytest.mark.parametrize("inverted", [False, True])
def test_usp_chunking_polarity_offsets_and_bounds(
    data_length: int, inverted: bool
) -> None:
    payload = bytes(range(27))
    stream = _constructed_stream(data_length, payload)
    prefix = np.asarray([-1.0, -1.0, 1.0])
    stream = np.concatenate((prefix, stream))
    if inverted:
        stream = -stream
    decoder = build_decoder("Unified SPUTNIX Protocol")
    frames = []
    for start in range(0, stream.size, 37):
        frames += decoder.push(stream[start : start + 37])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].source_start == prefix.size
    assert frames[0].source_end == prefix.size + 64 + CAPTURE_SYMBOLS
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)


def test_usp_rs_correction_count_and_uncorrectable_rejection() -> None:
    payload = bytes(range(27))
    corrected = build_decoder("USP").push(
        _constructed_stream(223, payload, rs_errors=16)
    )
    assert [frame.payload for frame in corrected] == [payload]
    assert corrected[0].corrected_symbols == 16
    assert build_decoder("USP").push(
        _constructed_stream(223, payload, rs_errors=17)
    ) == []


def test_usp_rejects_ambiguous_pls_and_invalid_encapsulated_length() -> None:
    stream = _constructed_stream(48, b"payload")
    stream[64 : 64 + 64] = 0.0
    assert build_decoder("USP").push(stream) == []
    assert build_decoder("USP").push(
        _constructed_stream(48, b"payload", declared_length=45)
    ) == []


def test_usp_truncation_flush_and_invalid_soft_input() -> None:
    stream = _constructed_stream(48, b"payload")
    decoder = build_decoder("USP")
    assert decoder.push(stream[:-1]) == []
    assert decoder.flush() == []
    assert decoder.retained_symbols == 0
    with pytest.raises(ValueError, match="finite"):
        decoder.push([np.nan])


def test_usp_registry_contract() -> None:
    profile = REGISTRY.resolve("USP")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert profile.max_retained_symbols == CAPTURE_SYMBOLS + len(SYNCWORD) - 1
    with pytest.raises(ValueError, match="<= 64"):
        build_decoder("USP", {"sync_threshold": 65})

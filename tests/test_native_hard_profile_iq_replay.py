"""Generated FSK IQ replay for remaining bounded hard-bit native profiles.

These constructions prove the repository-owned modem-to-profile boundary. They
are not independent mission vectors and do not change production enablement.
"""

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing.codes.openlst import encode_openlst_fec
from native_framing.crc import (
    CRC16_ARC,
    CRC16_CC11XX,
    CRC16_CCITT_FALSE,
    CRC16_X25,
    CrcSpec,
)
from native_framing.linecode import (
    ccsds_randomize,
    differential_encode,
    pn9_bytes,
    reflect_bytes,
)
from native_framing.profiles.cc11xx import (
    FRAME_SIZE as CC11XX_FRAME_SIZE,
)
from native_framing.profiles.cc11xx import (
    SYNCWORD as CC11XX_SYNCWORD,
)
from native_framing.profiles.ccsds import SYNCWORD as CCSDS_SYNCWORD
from native_framing.profiles.grizu import (
    FRAME_SIZE as GRIZU_FRAME_SIZE,
)
from native_framing.profiles.grizu import (
    SCRAMBLER_SEED as GRIZU_SCRAMBLER_SEED,
)
from native_framing.profiles.grizu import (
    SYNCWORD as GRIZU_SYNCWORD,
)
from native_framing.profiles.openlst import (
    CAPTURE_SIZE as OPENLST_CAPTURE_SIZE,
)
from native_framing.profiles.openlst import (
    SYNCWORD as OPENLST_SYNCWORD,
)
from native_framing.profiles.sanosat import (
    FRAME_SIZE as SANOSAT_FRAME_SIZE,
)
from native_framing.profiles.sanosat import (
    SYNCWORD as SANOSAT_SYNCWORD,
)
from native_framing.profiles.smogp import RX_SYNCWORD as SMOGP_SIGNALLING_SYNCWORD
from native_framing.profiles.tt64 import (
    PARITY_SIZE as TT64_PARITY_SIZE,
)
from native_framing.profiles.tt64 import (
    SYNCWORD as TT64_SYNCWORD,
)

from gfsk_ax25.reedsolomon import RSCodec

_SAMPLE_RATE = 48_000.0
_SYMBOL_RATE = 9_600.0
_MOD_INDEX = 0.8


def _bits(syncword: str, wire: bytes) -> np.ndarray:
    sync = np.fromiter((char == "1" for char in syncword), dtype=np.uint8)
    return np.concatenate((sync, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


def _fsk_capture(bits: np.ndarray) -> np.ndarray:
    samples_per_symbol = int(_SAMPLE_RATE / _SYMBOL_RATE)
    symbols = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    instantaneous_hz = (
        np.repeat(symbols, samples_per_symbol) * _MOD_INDEX * _SYMBOL_RATE / 2.0
    )
    phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / _SAMPLE_RATE
    burst = np.exp(1j * phase).astype(np.complex64)
    guard = np.zeros(2_000, dtype=np.complex64)
    return np.concatenate((guard, burst, guard))


def _decode(
    tmp_path: Path,
    *,
    label: str,
    canonical: str,
    bits: np.ndarray,
    parameters: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    path = tmp_path / f"{canonical}.cf32"
    _fsk_capture(bits).tofile(path)
    waveform: dict[str, object] = {
        "modulation": "fsk",
        "mod_index": _MOD_INDEX,
    }
    waveform.update(parameters or {})
    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_SAMPLE_RATE,
        symbol_rate_hz=_SYMBOL_RATE,
        framings_to_try=(label,),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        framing_parameters=waveform,
        native_evaluation=True,
    )
    assert [record["framing"] for record in records] == [canonical]
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"
    return records


@pytest.mark.parametrize("precoding", ["none", "differential"])
def test_ccsds_uncoded_fsk_file_iq_replay_routes_parameterized_profile(
    tmp_path: Path, precoding: str
) -> None:
    payload = bytes(range(64))
    channel_bits = _bits(CCSDS_SYNCWORD, ccsds_randomize(payload))
    if precoding == "differential":
        channel_bits = np.concatenate(
            (np.tile(np.asarray([0, 1], dtype=np.uint8), 8), channel_bits)
        )
        channel_bits = differential_encode(channel_bits)
    records = _decode(
        tmp_path,
        label="CCSDS Uncoded",
        canonical="ccsds_uncoded",
        bits=channel_bits,
        parameters={
            "frame_size": 64.0,
            "scrambler": "CCSDS",
            "precoding": precoding,
            "sync_threshold": 0.0,
        },
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]
    assert records[0]["crc_ok"] is False
    assert records[0]["integrity"] == "not_present"
    metadata = records[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["precoding"] == precoding
    assert "no integrity gate" in str(metadata["false_positive_policy"])


def test_openlst_fsk_file_iq_replay_routes_full_fec_chain(tmp_path: Path) -> None:
    payload = bytes((index * 29 + 0x37) & 0xFF for index in range(93))
    body = bytes([len(payload) + 2]) + payload
    frame = CRC16_CC11XX.append(body, byteorder="little")
    wire = encode_openlst_fec(pn9_bytes(frame), encoded_size=OPENLST_CAPTURE_SIZE)
    records = _decode(
        tmp_path,
        label="OpenLST",
        canonical="openlst",
        bits=_bits(OPENLST_SYNCWORD, wire),
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]


@pytest.mark.parametrize(
    ("label", "canonical", "crc", "byteorder"),
    (
        ("Reaktor Hello World", "reaktor_hello_world", CRC16_CC11XX, "big"),
        ("AALTO-1", "aalto1", CRC16_X25, "little"),
    ),
)
def test_cc11xx_profiles_fsk_file_iq_replay_route_shared_chain(
    tmp_path: Path,
    label: str,
    canonical: str,
    crc: CrcSpec,
    byteorder: str,
) -> None:
    payload = b"mission-payload"
    without_crc = bytes([len(payload) + 3, 0xAA, 0x55]) + payload + b"\x7e"
    packet = crc.append(without_crc, byteorder=byteorder)
    wire = pn9_bytes(packet + bytes(CC11XX_FRAME_SIZE - len(packet)))
    records = _decode(
        tmp_path,
        label=label,
        canonical=canonical,
        bits=_bits(CC11XX_SYNCWORD, wire),
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]


def test_tt64_fsk_file_iq_replay_routes_rs_crc_chain(tmp_path: Path) -> None:
    payload = bytes((index * 19 + 7) & 0xFF for index in range(46))
    data = CRC16_ARC.append(payload, byteorder="little")
    wire = RSCodec(TT64_PARITY_SIZE, prim=0x11D, fcr=1, generator=2).encode(data)
    records = _decode(
        tmp_path,
        label="TT-64",
        canonical="tt64",
        bits=_bits(TT64_SYNCWORD, wire),
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]


def test_sanosat_fsk_file_iq_replay_routes_mission_wire_contract(
    tmp_path: Path,
) -> None:
    payload = b"sanosat-payload"
    declared_length = len(payload) + 4
    length = bytes([declared_length])
    crc1 = CRC16_CCITT_FALSE.append(length, byteorder="little")[1:]
    without_crc1 = length + b"\xff\xff\x00\x00" + payload
    crc2 = CRC16_CCITT_FALSE.append(without_crc1, byteorder="little")
    packet = length + crc1 + crc2[1:]
    wire = packet + bytes(SANOSAT_FRAME_SIZE - len(packet))
    records = _decode(
        tmp_path,
        label="SanoSat",
        canonical="sanosat",
        bits=_bits(SANOSAT_SYNCWORD, wire),
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]
    metadata = records[0]["metadata"]
    assert isinstance(metadata, dict)
    assert "0xb42b" in str(metadata["syncword_source"])
    assert metadata["crc1"] == metadata["crc2"] == "passed"


def test_grizu_fsk_file_iq_replay_routes_reflection_whitening_crc_chain(
    tmp_path: Path,
) -> None:
    payload = b"grizu-payload"
    without_crc = bytes([len(payload) + 3, 0xAA, 0x55]) + payload + b"\x7e"
    packet = CRC16_CC11XX.append(without_crc, byteorder="big")
    decoded = packet + bytes(GRIZU_FRAME_SIZE - len(packet))
    wire = reflect_bytes(
        pn9_bytes(reflect_bytes(decoded), seed=GRIZU_SCRAMBLER_SEED)
    )
    records = _decode(
        tmp_path,
        label="Grizu-263A",
        canonical="grizu263a",
        bits=_bits(GRIZU_SYNCWORD, wire),
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]


def test_smogp_signalling_fsk_file_iq_replay_preserves_no_integrity_policy(
    tmp_path: Path,
) -> None:
    payload = bytes(range(64))
    records = _decode(
        tmp_path,
        label="SMOG-P Signalling",
        canonical="smogp_signalling",
        bits=_bits(SMOGP_SIGNALLING_SYNCWORD, payload),
        parameters={"sync_threshold": 0.0},
    )
    assert [bytes.fromhex(str(record["payload_hex"])) for record in records] == [payload]
    assert records[0]["crc_ok"] is False
    assert records[0]["integrity"] == "not_present"
    metadata = records[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["sync_variant"] == "rx"
    assert "never autodetect" in str(metadata["false_positive_policy"])

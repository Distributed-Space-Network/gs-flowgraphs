"""Generated construction tests for the pinned SanoSat receive behavior."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.crc import CRC16_CCITT_FALSE
from native_framing.profiles.sanosat import FRAME_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)


def _wire(payload: bytes = b"sanosat-payload") -> bytes:
    # The declared length covers CRC1 + message + CRC2. The length byte and
    # four-byte delimiter account for the five additional captured bytes.
    declared_length = len(payload) + 4
    length = bytes([declared_length])
    crc1 = CRC16_CCITT_FALSE.append(length, byteorder="little")[1:]
    without_crc1 = length + b"\xff\xff\x00\x00" + payload
    crc2 = CRC16_CCITT_FALSE.append(without_crc1, byteorder="little")
    packet = length + crc1 + crc2[1:]
    return packet + bytes(FRAME_SIZE - len(packet))


def _stream(wire: bytes) -> np.ndarray:
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(wire, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 15, 16, 127, 2048])
@pytest.mark.parametrize("inverted", [False, True])
def test_sanosat_chunk_polarity_crop_and_crc(step: int, inverted: bool):
    expected = b"sanosat-payload"
    stream = _stream(_wire(expected))
    if inverted:
        stream = 1 - stream
    decoder = build_decoder("SanoSat")
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
    assert [frame.payload for frame in frames] == [expected]
    assert frames[0].polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert "0xb42b" in frames[0].metadata["syncword_source"]
    assert frames[0].metadata["crc1"] == frames[0].metadata["crc2"] == "passed"


def test_sanosat_mission_hardware_wire_vector_decodes_byte_exactly() -> None:
    wire = bytes.fromhex(
        "1f2e02ffff00004e50514449474950454154455220544553542053414e4f53415400fc6d"
    )
    frames = build_decoder("SanoSat").push(_stream(wire + bytes(FRAME_SIZE - len(wire))))

    assert [frame.payload for frame in frames] == [b"NPQDIGIPEATER TEST SANOSAT\x00"]
    assert frames[0].metadata["packet_length_before_crc1_removal"] == 36


def test_sanosat_rejects_crc_length_sync_and_truncation():
    corrupted = bytearray(_wire())
    corrupted[10] ^= 1
    assert build_decoder("SanoSat").push(_stream(bytes(corrupted))) == []

    bad_crc1 = bytearray(_wire())
    bad_crc1[1] ^= 1
    assert build_decoder("SanoSat").push(_stream(bytes(bad_crc1))) == []

    bad_delimiter = bytearray(_wire())
    bad_delimiter[3] ^= 1
    assert build_decoder("SanoSat").push(_stream(bytes(bad_delimiter))) == []

    malformed = bytearray(_wire())
    malformed[0] = 255
    assert build_decoder("SanoSat").push(_stream(bytes(malformed))) == []

    sync_error = _stream(_wire())
    sync_error[0] ^= 1
    assert build_decoder("SanoSat").push(sync_error) == []

    decoder = build_decoder("SanoSat")
    assert decoder.push(_stream(_wire())[:-1]) == []
    assert decoder.flush() == []


def test_sanosat_mission_sources_resolve_sync_but_not_live_qualification():
    profile = REGISTRY.resolve("SanoSat-1")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported
    assert not profile.post_pass_supported
    assert "0x2dd4" in profile.sync_policy
    assert "0xb42b" in profile.sync_policy

"""Source-derived construction tests for the bounded native S-NET profile."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.profiles.snet import CAPTURE_SIZE, SYNCWORD
from native_framing.registry import REGISTRY
from native_framing.types import DecodeDisposition, Polarity

_EXP = (8, 4, 2, 1, 12, 6, 3, 13, 10, 5, 14, 7, 15, 11, 9)
_K = {3: 11, 5: 7, 7: 5}
_SYNC = np.fromiter((character == "1" for character in SYNCWORD), dtype=np.uint8)
_PREFIX = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
_SNET_A_CAPTURE = bytes.fromhex(
    "062521f06748044c01314243db920209a8d482db866a64810246d0e941c2b154"
    "c1728894d2c48a9a10597802a2e2c37c4a09205c3821ab27a58e399e84e3888"
    "a895e9d4284a7949a3423912b857380862dc6811625ca2143eb45a04342c24b45"
    "e39009c708c4004301110040a011ea14001208c785730d350a765ad28723058585"
    "e3d2948060552158125032000602560ae6813b35c10aea7c0623812245064f48f"
    "c1caa54a959c8493838a01a1a0d9e089cd052994281033998e84130db59cd7088"
    "f00088408814a8129916a105290e8d04ab1e231fa3120d01800587158508850000"
    "082906080c82100215027000c80df008b80cd007380f700a0001e802700b1801e"
    "809a0070000f009a98cd4ca6a453505c5cae2ce2e0048437b45bd8edecf6f47b"
    "78bdbcdedd57fcee3b373511a4fc7efdeb2a36a8cd1b857a7d6065595783d1dd7"
    "6f56ef1c416f76aedacdefe84a521c61ab15eeb9250575f0c8c05953a02172e55"
    "1dd7c891dd899f4cb235a4247095efb57e407345efd8b8b6528f10f7e01ffffd4"
    "5ffffffffffffffffffffffffffffffff555555220af22a424204cf5fc8061d20f0"
    "6188064c3d30e241db8e0219a8e4839b8f6a3c807241952a1748198614c92f53"
    "209a0443602b2ec40a0e71ca7c99bc2b8fdb5409aa31556e5d9234f16dbac629"
    "9132cac6e223179800fd898f2b6e70858fa2923f2534b30fd11"
)
_SNET_A_PAYLOAD = bytes.fromhex(
    "f3501ae0240a2c660a873a448f0dd101a40ab063a9079210f73f5c0018001400"
    "11008b001d001a5e3211000751001600c34b1600ce17000f905b7f08625e091b"
    "f41b1816000ed3110f00000708005c000d03fe02080066005e00000070058b00"
    "460000000564ea65975b6303000000000000"
)


def _syndromes(word: int, distance: int) -> tuple[int, ...]:
    output = []
    for root in range(1, distance):
        syndrome = 0
        value = word
        for power in range(14, -1, -1):
            if value & 1:
                syndrome ^= _EXP[(power * root) % 15]
            value >>= 1
        output.append(syndrome)
    return tuple(output)


@cache
def _encode_bch(data: tuple[int, ...], distance: int) -> tuple[int, ...]:
    width = _K[distance]
    value = 0
    for bit in data:
        value = (value << 1) | int(bit)
    for parity in range(1 << (15 - width)):
        candidate = (parity << width) | value
        if not any(_syndromes(candidate, distance)):
            return tuple((candidate >> shift) & 1 for shift in range(14, -1, -1))
    raise AssertionError("no systematic BCH codeword found")


def _crc5(bits: np.ndarray, *, buggy: bool) -> int:
    rows = np.concatenate((bits, [1, 0, 1, 1, 0, 1, 1])).reshape(9, 8)
    if buggy:
        rows = np.flipud(rows).copy()
        rows[4] = rows[3]
    crc = 0x1F
    for bit in rows.ravel():
        top = (crc >> 4) & 1
        crc = (crc << 1) & 0x1F
        if top != bit:
            crc ^= 0x15
    return crc


def _crc13(payload: bytes, *, buggy: bool) -> int:
    rows = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape((-1, 8))
    if buggy:
        rows = np.flipud(rows)
    crc = 0x1FFF
    for bit in rows.ravel():
        top = (crc >> 12) & 1
        crc = (crc << 1) & 0x1FFF
        if (top or int(bit)) if buggy else (top != int(bit)):
            crc ^= 0x1CF5
    return crc


def _put(bits: np.ndarray, start: int, width: int, value: int) -> None:
    for index in range(width):
        bits[start + index] = (value >> (width - 1 - index)) & 1


def _header(payload: bytes, *, ai_type: int, buggy: bool) -> np.ndarray:
    header = np.zeros(70, dtype=np.uint8)
    _put(header, 0, 7, 0x25)
    _put(header, 7, 7, 0x12)
    _put(header, 14, 4, 3)
    _put(header, 18, 4, 4)
    _put(header, 22, 4, 9)
    _put(header, 26, 4, ai_type)
    _put(header, 30, 4, 0)
    _put(header, 42, 10, len(payload))
    _put(header, 52, 13, _crc13(payload, buggy=buggy))
    _put(header, 65, 5, _crc5(header[:65], buggy=buggy))
    data = np.fliplr(header.reshape((14, 5)))
    codewords = np.asarray([_encode_bch(tuple(row), 7) for row in data], dtype=np.uint8)
    return codewords.transpose().ravel()


def _payload(payload: bytes, *, ai_type: int) -> np.ndarray:
    if ai_type == 0:
        return np.fliplr(
            np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).reshape((-1, 8))
        ).ravel()
    data_width, distance = {1: (11, 3), 2: (7, 5), 3: (5, 7)}[ai_type]
    block_bytes = 16 * data_width // 8
    blocks = (len(payload) + block_bytes - 1) // block_bytes
    padded = payload.ljust(blocks * block_bytes, b"\xDB")
    raw = np.fliplr(
        np.unpackbits(np.frombuffer(padded, dtype=np.uint8)).reshape((-1, 8))
    ).ravel()
    wire = []
    for block in range(blocks):
        start = block * block_bytes * 8
        data = raw[start : start + block_bytes * 8].reshape((16, data_width))
        codewords = np.asarray(
            [_encode_bch(tuple(row), distance) for row in data], dtype=np.uint8
        )
        wire.append(codewords.transpose().ravel())
    return np.concatenate(wire)


def _stream(
    *,
    ai_type: int,
    buggy: bool = False,
    inverted: bool = False,
    header_errors: int = 0,
    payload_errors: int = 0,
    sync_errors: int = 0,
) -> tuple[np.ndarray, bytes]:
    payload = bytes((index * 29 + ai_type * 7 + 3) & 0xFF for index in range(23))
    header = _header(payload, ai_type=ai_type, buggy=buggy)
    for bit in range(header_errors):
        header[bit * 14] ^= 1
    body = _payload(payload, ai_type=ai_type)
    for bit in range(payload_errors):
        body[bit * (8 if ai_type == 0 else 16)] ^= 1
    capture = np.zeros(CAPTURE_SIZE * 8, dtype=np.uint8)
    capture[:210] = header
    capture[210 : 210 + body.size] = body
    sync = _SYNC.copy()
    sync[:sync_errors] ^= 1
    stream = np.concatenate((_PREFIX, sync, capture))
    if inverted:
        stream = 1 - stream
    return stream, payload


def test_snet_exact_published_real_capture() -> None:
    assert len(_SNET_A_CAPTURE) == CAPTURE_SIZE
    stream = np.concatenate(
        (_SYNC, np.unpackbits(np.frombuffer(_SNET_A_CAPTURE, dtype=np.uint8)))
    )
    frames = build_decoder("SNET", {"buggy_crc": True}).push(stream)
    assert [frame.payload for frame in frames] == [_SNET_A_PAYLOAD]
    frame = frames[0]
    assert frame.source_start == 0 and frame.source_end == stream.size
    assert frame.corrected_symbols == 0
    assert frame.metadata["src_id"] == 0
    assert frame.metadata["ai_type_src"] == 2
    assert frame.metadata["pdu_length"] == 114
    assert build_decoder("SNET").push(stream) == []


@pytest.mark.parametrize("ai_type", range(4))
@pytest.mark.parametrize("inverted", [False, True])
def test_snet_all_fec_modes_chunks_polarity_offsets_and_metadata(
    ai_type: int, inverted: bool
) -> None:
    stream, expected = _stream(ai_type=ai_type, inverted=inverted)
    decoder = build_decoder("SNET")
    frames = []
    for start in range(0, stream.size, 73):
        frames += decoder.push(stream[start : start + 73])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [expected]
    frame = frames[0]
    assert frame.source_start == _PREFIX.size
    assert frame.source_end == stream.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.corrected_symbols == 0
    assert frame.metadata["src_id"] == 0x25
    assert frame.metadata["ai_type_src"] == ai_type
    assert frame.metadata["pdu_length"] == len(expected)
    assert frame.metadata["buggy_crc"] is False


def test_snet_buggy_crc_is_explicit_and_cross_rejected() -> None:
    normal, payload = _stream(ai_type=0, buggy=False)
    buggy, _ = _stream(ai_type=0, buggy=True)
    assert [frame.payload for frame in build_decoder("SNET").push(normal)] == [payload]
    assert build_decoder("SNET", {"buggy_crc": True}).push(normal) == []
    assert build_decoder("SNET").push(buggy) == []
    frames = build_decoder("SNET", {"buggy_crc": True}).push(buggy)
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].metadata["buggy_crc"] is True


@pytest.mark.parametrize("ai_type,correctable", [(1, 1), (2, 2), (3, 3)])
def test_snet_bch_correction_boundaries(ai_type: int, correctable: int) -> None:
    accepted, payload = _stream(ai_type=ai_type, payload_errors=correctable)
    frames = build_decoder("SNET").push(accepted)
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].corrected_symbols == correctable
    rejected, _ = _stream(ai_type=ai_type, payload_errors=correctable + 1)
    assert build_decoder("SNET").push(rejected) == []


def test_snet_header_crc_sync_threshold_truncation_and_flush() -> None:
    corrected, payload = _stream(ai_type=2, header_errors=3, sync_errors=4)
    frames = build_decoder("SNET").push(corrected)
    assert [frame.payload for frame in frames] == [payload]
    assert frames[0].metadata["header_corrected_bits"] == 3
    assert frames[0].sync_distance == 4

    rejected, _ = _stream(ai_type=2, sync_errors=5)
    assert build_decoder("SNET").push(rejected) == []

    decoder = build_decoder("SNET")
    assert decoder.push(corrected[:-1]) == []
    assert decoder.flush() == []
    assert [frame.payload for frame in decoder.push(corrected)] == [payload]


def test_snet_registry_contract_is_nonproduction_and_standard_by_default() -> None:
    profile = REGISTRY.resolve("S-NET")
    assert profile is not None
    assert profile.disposition is DecodeDisposition.IN_PROGRESS
    assert profile.decoder_available
    assert not profile.live_supported and not profile.post_pass_supported
    assert profile.parameters["buggy_crc"].default is False
    with pytest.raises(ValueError):
        build_decoder("SNET", {"sync_threshold": 16})

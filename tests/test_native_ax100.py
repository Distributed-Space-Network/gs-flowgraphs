"""Generated construction and pinned-behavior tests for AX100 Mode 6."""

from __future__ import annotations

from pathlib import Path

import iq_decode
import numpy as np
import pytest
from native_framing import build_decoder
from native_framing.codes.golay24 import decode_golay24, encode_golay24
from native_framing.linecode import (
    SelfSynchronizingDescrambler,
    ccsds_randomize,
    multiplicative_scramble,
)
from native_framing.profiles.ax100 import (
    ASM_CAPTURE_SIZE,
    CAPTURE_SIZE,
    DESCRAMBLER_LENGTH,
    DESCRAMBLER_MASK,
    DESCRAMBLER_SEED,
    SYNCWORD,
)
from native_framing.registry import REGISTRY
from native_framing.rs import CcsdsReedSolomon
from native_framing.types import DecodeDisposition, Polarity

_RS = CcsdsReedSolomon(basis="conventional", interleaving=1)
_SYNC = np.fromiter((char == "1" for char in SYNCWORD), dtype=np.uint8)
_PREAMBLE = np.zeros(64, dtype=np.uint8)
_PAYLOAD = bytes((index * 31 + 9) & 0xFF for index in range(80))
_IQ_SAMPLE_RATE = 48_000.0
_IQ_SYMBOL_RATE = 9_600.0


def _cpfsk_capture(bits: np.ndarray, *, mod_index: float) -> np.ndarray:
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


def _plain_stream(
    payload: bytes = _PAYLOAD,
    *,
    corrupt_positions: tuple[int, ...] = (),
    sync_errors: tuple[int, ...] = (),
    declared_length: int | None = None,
) -> np.ndarray:
    codeword = bytearray(_RS.encode(payload))
    for index, position in enumerate(corrupt_positions):
        codeword[position] ^= index + 1
    length = len(codeword) + 1 if declared_length is None else declared_length
    capture = bytes([length]) + bytes(codeword)
    capture = capture[:CAPTURE_SIZE].ljust(CAPTURE_SIZE, b"\x00")
    sync = _SYNC.copy()
    if sync_errors:
        sync[list(sync_errors)] ^= 1
    return np.concatenate((_PREAMBLE, sync, np.unpackbits(np.frombuffer(capture, dtype=np.uint8))))


def _wire_stream(**kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
    return multiplicative_scramble(
        _plain_stream(**kwargs),
        mask=DESCRAMBLER_MASK,
        seed=DESCRAMBLER_SEED,
        register_length=DESCRAMBLER_LENGTH,
    )


def _asm_stream(
    *,
    scrambler: str = "CCSDS",
    header_errors: tuple[int, ...] = (),
    packet_errors: tuple[int, ...] = (),
    header_flags: int = 0,
    declared_length: int | None = None,
) -> np.ndarray:
    packet = bytearray(_RS.encode(_PAYLOAD))
    for index, position in enumerate(packet_errors):
        packet[position] ^= index + 1
    length = len(packet) if declared_length is None else declared_length
    header = encode_golay24((length & 0xFF) | (header_flags & 0xF00))
    for position in header_errors:
        header ^= 1 << position
    transmitted = ccsds_randomize(packet) if scrambler == "CCSDS" else bytes(packet)
    capture = header.to_bytes(3, "big") + transmitted
    capture = capture[:ASM_CAPTURE_SIZE].ljust(ASM_CAPTURE_SIZE, b"\x00")
    return np.concatenate((_SYNC, np.unpackbits(np.frombuffer(capture, dtype=np.uint8))))


@pytest.mark.parametrize("step", [1, 7, 31, 257, 4096])
@pytest.mark.parametrize("inverted", [False, True])
def test_ax100_mode6_chunk_polarity_offsets_and_metadata(step: int, inverted: bool) -> None:
    wire = _wire_stream()
    if inverted:
        wire = 1 - wire
    decoder = build_decoder("AX.100 Mode 6")
    frames = []
    for start in range(0, wire.size, step):
        frames += decoder.push(wire[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.source_start == _PREAMBLE.size
    assert frame.source_end == wire.size
    assert frame.polarity is (Polarity.INVERTED if inverted else Polarity.NORMAL)
    assert frame.corrected_symbols == 0
    assert frame.metadata["ax100_mode"] == 6
    assert frame.metadata["declared_length"] == len(_PAYLOAD) + 33
    assert frame.metadata["rs_basis"] == "conventional"


def test_ax100_mode6_rs_boundary_and_length_fail_closed() -> None:
    corrected = build_decoder("AX100 Mode 6").push(
        _wire_stream(corrupt_positions=tuple(range(0, 48, 3)))
    )
    assert [frame.payload for frame in corrected] == [_PAYLOAD]
    assert corrected[0].corrected_symbols == 16

    assert (
        build_decoder("AX100 Mode 6").push(_wire_stream(corrupt_positions=tuple(range(20)))) == []
    )
    assert build_decoder("AX100 Mode 6").push(_wire_stream(declared_length=33)) == []
    assert (
        build_decoder("AX100 Mode 6").push(_wire_stream(declared_length=len(_PAYLOAD) + 32)) == []
    )


def test_ax100_mode6_sync_threshold_truncation_and_flush_reset() -> None:
    assert len(build_decoder("AX100 Mode 6").push(_wire_stream(sync_errors=(0, 1, 2, 3)))) == 1
    assert build_decoder("AX100 Mode 6").push(_wire_stream(sync_errors=(0, 1, 2, 3, 4))) == []

    truncated = build_decoder("AX100 Mode 6")
    assert truncated.push(_wire_stream()[:-1]) == []
    assert truncated.flush() == []
    assert [frame.payload for frame in truncated.push(_wire_stream())] == [_PAYLOAD]

    with pytest.raises(ValueError):
        build_decoder("AX100 Mode 6", {"sync_threshold": 33})


def test_multiplicative_descrambler_matches_pinned_gnuradio_algorithm() -> None:
    plain = np.asarray([(index * 7 + index // 3) & 1 for index in range(257)], dtype=np.uint8)
    wire = multiplicative_scramble(
        plain,
        mask=DESCRAMBLER_MASK,
        seed=DESCRAMBLER_SEED,
        register_length=DESCRAMBLER_LENGTH,
    )

    # Independent literal transcription of digital::lfsr::next_bit_descramble from the pinned
    # GNU Radio source. It guards register insertion position and tap-parity conventions.
    state = DESCRAMBLER_SEED
    reference = []
    for received in wire:
        reference.append(((state & DESCRAMBLER_MASK).bit_count() & 1) ^ int(received))
        state = (state >> 1) | (int(received) << DESCRAMBLER_LENGTH)
    assert np.array_equal(reference, plain)

    streaming = SelfSynchronizingDescrambler(DESCRAMBLER_MASK, DESCRAMBLER_SEED, DESCRAMBLER_LENGTH)
    got = np.concatenate(
        [
            streaming.push(wire[:13]),
            streaming.push(wire[13:111]),
            streaming.push(wire[111:]),
        ]
    )
    assert np.array_equal(got, plain)
    with pytest.raises(ValueError):
        SelfSynchronizingDescrambler(0, 0, 16)


def test_ax100_mode6_is_available_without_mode5_aliasing_or_completion_claim() -> None:
    mode6 = REGISTRY.resolve("AX100 RS")
    mode5 = REGISTRY.resolve("AX100 Mode 5")
    assert mode6 is not None and mode5 is not None and mode6 is not mode5
    assert mode6.disposition is DecodeDisposition.IN_PROGRESS
    assert mode6.decoder_available
    assert not mode6.live_supported and not mode6.post_pass_supported
    assert mode5.disposition is DecodeDisposition.IN_PROGRESS and mode5.decoder_available


@pytest.mark.parametrize(("modulation", "mod_index"), [("fsk", 0.8), ("msk", 0.5)])
@pytest.mark.parametrize(
    ("label", "canonical", "stream_factory"),
    [
        ("AX100 Mode 5", "ax100_mode5", _asm_stream),
        ("AX100 ASM+Golay", "ax100_asm_golay", _asm_stream),
        ("AX100 Mode 6", "ax100_mode6", _wire_stream),
    ],
)
def test_ax100_fsk_msk_file_iq_replay_routes_native_profiles(
    tmp_path: Path,
    modulation: str,
    mod_index: float,
    label: str,
    canonical: str,
    stream_factory,  # type: ignore[no-untyped-def]
) -> None:
    capture = _cpfsk_capture(stream_factory(), mod_index=mod_index)
    path = tmp_path / f"{canonical}-{modulation}.cf32"
    capture.tofile(path)

    records = iq_decode.decode_capture(
        path,
        sample_rate_hz=_IQ_SAMPLE_RATE,
        symbol_rate_hz=_IQ_SYMBOL_RATE,
        framings_to_try=(label,),
        doppler_track=[(0.0, 0.0)],
        capture_start_unix_s=1_767_225_600.0,
        modulation=modulation,
        mod_index=mod_index,
        native_evaluation=True,
    )
    assert [bytes.fromhex(record["payload_hex"]) for record in records] == [_PAYLOAD]
    assert records[0]["framing"] == canonical
    assert records[0]["source_offset_kind"] == "demodulated_symbol_estimate"


@pytest.mark.parametrize(
    "label,canonical",
    [("AX100 Mode 5", "ax100_mode5"), ("AX100 ASM", "ax100_asm_golay")],
)
@pytest.mark.parametrize("scrambler", ["CCSDS", "none"])
@pytest.mark.parametrize("step", [1, 29, 257, 4096])
def test_ax100_asm_profiles_equivalence_chunking_and_scrambler(
    label: str, canonical: str, scrambler: str, step: int
) -> None:
    stream = _asm_stream(scrambler=scrambler)
    decoder = build_decoder(label, {"scrambler": scrambler})
    frames = []
    for start in range(0, stream.size, step):
        frames += decoder.push(stream[start : start + step])
        assert decoder.retained_symbols <= decoder.max_retained_symbols
    assert [frame.payload for frame in frames] == [_PAYLOAD]
    frame = frames[0]
    assert frame.canonical_framing == canonical
    assert frame.source_start == 0 and frame.source_end == stream.size
    assert frame.metadata["ax100_mode"] == 5
    assert frame.metadata["forced_scrambler"] == scrambler
    assert frame.metadata["profile_equivalence"].startswith("AX100 Mode 5")


def test_ax100_asm_golay_and_rs_correction_boundaries() -> None:
    corrected = build_decoder("AX100 Mode 5").push(
        _asm_stream(
            header_errors=(0, 7, 23),
            packet_errors=tuple(range(0, 48, 3)),
            header_flags=0,
        )
    )
    assert [frame.payload for frame in corrected] == [_PAYLOAD]
    assert corrected[0].metadata["golay_corrected_bits"] == 3
    assert corrected[0].corrected_symbols == 16
    # Pinned ASM construction forces Viterbi off, scrambler on, and RS on. Header flags are only
    # diagnostic in this profile and must not silently override the configured chain.
    assert corrected[0].metadata["header_viterbi_flag"] is False
    assert corrected[0].metadata["header_scrambler_flag"] is False
    assert corrected[0].metadata["header_rs_flag"] is False

    assert build_decoder("AX100 Mode 5").push(
        _asm_stream(packet_errors=tuple(range(20)))
    ) == []
    assert build_decoder("AX100 Mode 5").push(_asm_stream(declared_length=32)) == []


def test_ax100_asm_rejects_wrong_scrambler_mode6_and_invalid_parameters() -> None:
    stream = _asm_stream(scrambler="CCSDS")
    assert build_decoder("AX100 Mode 5", {"scrambler": "none"}).push(stream) == []
    assert build_decoder("AX100 Mode 6").push(stream) == []
    with pytest.raises(ValueError):
        build_decoder("AX100 Mode 5", {"scrambler": "auto"})


def test_golay24_exact_codewords_and_bounded_correction() -> None:
    for data in (0, 1, 0x5A5, 0xFFF):
        codeword = encode_golay24(data)
        clean = decode_golay24(codeword)
        assert clean is not None and clean.data == data and clean.corrected_bits == 0
        for mask in (
            1 << 0,
            (1 << 0) | (1 << 13),
            (1 << 0) | (1 << 7) | (1 << 23),
        ):
            corrected = decode_golay24(codeword ^ mask)
            assert corrected is not None
            assert corrected.data == data
            assert corrected.codeword == codeword
            assert corrected.corrected_bits == mask.bit_count()

    with pytest.raises(ValueError):
        encode_golay24(0x1000)
    with pytest.raises(ValueError):
        decode_golay24(0x1000000)

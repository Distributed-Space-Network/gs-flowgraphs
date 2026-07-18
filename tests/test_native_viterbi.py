"""CCSDS/NASA K=7 convolutional-code vectors, modes, and soft decode."""

from __future__ import annotations

import numpy as np
import pytest
from native_framing.viterbi import (
    CONVENTIONS,
    ConvolutionalCode,
    StreamingViterbiDecoder,
    decode_hypotheses,
)


def test_upstream_convolutional_conventions_are_exact_profile_data():
    assert CONVENTIONS == {
        "CCSDS": (79, -109),
        "NASA-DSN": (-109, 79),
        "CCSDS uninverted": (79, 109),
        "NASA-DSN uninverted": (109, 79),
    }


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("mode", ["terminated", "truncated", "tail_biting"])
def test_every_convention_and_finite_block_mode_roundtrips(
    convention: str, mode: str
):
    bits = tuple(np.random.default_rng(1982).integers(0, 2, 97))
    codec = ConvolutionalCode(convention)
    encoded = codec.encode(bits, mode=mode)
    result = codec.decode_hard(encoded, mode=mode)
    assert result.bits == bits
    assert result.metric == 0.0
    assert result.mode == mode
    if mode == "terminated":
        assert len(encoded) == 2 * (len(bits) + 6)
        assert result.final_state == 0
    else:
        assert len(encoded) == 2 * len(bits)


def test_signed_polynomial_encoder_has_stable_literal_vector():
    # Literal protects polynomial order/sign and six-pair termination sizing.
    encoded = ConvolutionalCode("CCSDS").encode((1, 0, 1, 1, 0, 0, 1, 0))
    assert encoded == (
        1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
    )


def test_clean_hard_and_soft_decode_are_equivalent():
    bits = tuple(np.random.default_rng(7).integers(0, 2, 256))
    codec = ConvolutionalCode("CCSDS")
    encoded = codec.encode(bits)
    hard = codec.decode_hard(encoded)
    soft = codec.decode_soft(np.asarray(encoded) * 7.0 - 3.5)
    assert hard.bits == soft.bits == bits
    assert hard.final_state == soft.final_state == 0


def test_soft_decoder_corrects_deterministic_symbol_errors():
    bits = tuple(np.random.default_rng(19).integers(0, 2, 200))
    codec = ConvolutionalCode("NASA-DSN")
    encoded = np.asarray(codec.encode(bits), dtype=np.float64)
    soft = encoded * 2.0 - 1.0
    soft[[23, 88, 171, 309]] *= -0.35
    result = codec.decode_soft(soft)
    assert result.bits == bits
    assert result.metric > 0.0


def test_hypothesis_decoder_recovers_convention_and_one_symbol_phase():
    bits = tuple(np.random.default_rng(27).integers(0, 2, 144))
    encoded = ConvolutionalCode("NASA-DSN").encode(bits, mode="truncated")
    soft = (0.2,) + tuple(4.0 if bit else -4.0 for bit in encoded)
    result = decode_hypotheses(soft, mode="truncated")
    assert result.bits == bits
    assert result.convention == "NASA-DSN"
    assert result.phase == 1


def test_tail_biting_automatic_start_is_last_six_information_bits():
    bits = (1, 0, 0, 1, 1, 1, 0, 1)
    codec = ConvolutionalCode("CCSDS")
    result = codec.decode_hard(codec.encode(bits, mode="tail_biting"), mode="tail_biting")
    assert result.bits == bits
    assert result.start_state == int("011101", 2)
    assert result.final_state == result.start_state


def test_short_tail_biting_block_wraps_its_information_bits():
    bits = (1, 0, 1)
    codec = ConvolutionalCode("CCSDS")
    encoded = codec.encode(bits, mode="tail_biting")
    assert codec.decode_hard(encoded, mode="tail_biting").bits == bits


@pytest.mark.parametrize("split", range(1, 38))
def test_streaming_viterbi_is_chunk_invariant_bounded_and_flushes(split: int):
    bits = tuple(np.random.default_rng(311).integers(0, 2, 173))
    encoded = ConvolutionalCode("CCSDS").encode(bits, mode="truncated")
    soft = tuple(3.0 if bit else -3.0 for bit in encoded)
    decoder = StreamingViterbiDecoder(
        "CCSDS", traceback_depth=35, start_state=0
    )
    output: tuple[int, ...] = ()
    for offset in range(0, len(soft), split):
        output += decoder.push(soft[offset : offset + split])
        assert decoder.retained_symbol_count <= 2 * 35 + 1
    output += decoder.flush()
    assert output == bits
    assert decoder.retained_symbol_count == 0


def test_streaming_viterbi_reset_and_incomplete_pair_are_explicit():
    codec = ConvolutionalCode("NASA-DSN")
    first = codec.encode((1,) * 100, mode="truncated")
    second_bits = (0, 1) * 60
    second = codec.encode(second_bits, mode="truncated")
    decoder = StreamingViterbiDecoder("NASA-DSN", start_state=0)
    decoder.push(tuple(1.0 if bit else -1.0 for bit in first))
    decoder.reset()
    output = decoder.push(tuple(1.0 if bit else -1.0 for bit in second))
    assert output + decoder.flush() == second_bits

    decoder.push((1.0,))
    with pytest.raises(ValueError, match="incomplete"):
        decoder.flush()


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda: ConvolutionalCode("unknown"), "unknown"),
        (lambda: ConvolutionalCode((0x4F,)), "exactly two"),
        (lambda: ConvolutionalCode((0x4F, 0)), "non-zero"),
        (lambda: ConvolutionalCode().encode(()), "non-empty"),
        (lambda: ConvolutionalCode().encode((0, 2)), "zero and one"),
        (lambda: ConvolutionalCode().decode_hard((0.5, 1)), "0/1"),
        (lambda: ConvolutionalCode().encode((0,), mode="streaming"), "mode"),
        (lambda: ConvolutionalCode().decode_soft((1.0,)), "complete pairs"),
        (lambda: ConvolutionalCode().decode_soft((1.0, float("nan"))), "finite"),
        (lambda: decode_hypotheses((1.0, -1.0), phases=(2,)), "phase"),
        (lambda: StreamingViterbiDecoder(traceback_depth=5), "at least six"),
    ],
)
def test_viterbi_validation_is_fail_closed(operation, match: str):
    with pytest.raises(ValueError, match=match):
        operation()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from native_framing.provenance import load_manifest
from native_lora.framing import (
    MAX_FRAME_SYMBOLS,
    WHITENING_SEQUENCE,
    LoRaFrameError,
    LoRaIntegrityError,
    LoRaPhyConfig,
    decode_lora_symbols,
    hamming_decode_codeword,
    lora_payload_crc,
)
from native_lora.modem import (
    LoRaModemConfig,
    LoRaSyncBuffer,
    LoRaSyncError,
    build_upchirp,
    demodulate_payload_bins,
    find_lora_sync,
    normalize_framing_symbols,
)

_COMMIT = "862746dd1cf635c9c8a4bfbaa2c3a0ec3a5306c9"
_TINYGS_COMMIT = "6dcbf47c45ac35bd3c2307113d12bdad42f415bd"
_LITERAL_SF7_CR48_VECTOR = (
    4,
    0,
    7,
    16,
    9,
    27,
    0,
    7,
    31,
    112,
    85,
    85,
    90,
    120,
    86,
    31,
    83,
    56,
    85,
    108,
    72,
    13,
    117,
    96,
    18,
    42,
    40,
    43,
    62,
    93,
    55,
    17,
    1,
    127,
    0,
    31,
    0,
    0,
    3,
    0,
)


def _synthetic_lora_iq(
    symbols: tuple[int, ...],
    config: LoRaModemConfig,
    *,
    ldro: bool = False,
    leading_samples: int = 23,
    cfo_bins: float = 0.0,
    sync_word: int | None = None,
) -> np.ndarray:
    """Construct a deterministic chirp-domain RX oracle, not a production encoder."""

    waveform_config = (
        config
        if sync_word is None
        else LoRaModemConfig(
            sf=config.sf,
            bandwidth=config.bandwidth,
            sample_rate=config.sample_rate,
            sync_word=sync_word,
            preamble_length=config.preamble_length,
            max_input_samples=config.max_input_samples,
        )
    )
    upchirp = build_upchirp(0, waveform_config)
    downchirp = np.conjugate(upchirp)
    raw_bins = [
        (symbol * 4 if index < 8 or ldro else symbol) % waveform_config.bins
        for index, symbol in enumerate(symbols)
    ]
    parts = [np.zeros(leading_samples, dtype=np.complex128)]
    parts.extend([upchirp] * waveform_config.preamble_length)
    parts.extend(build_upchirp(symbol, waveform_config) for symbol in waveform_config.sync_symbols)
    parts.extend((downchirp, downchirp, downchirp[: waveform_config.samples_per_symbol // 4]))
    parts.extend(
        build_upchirp((symbol + 1) % waveform_config.bins, waveform_config)
        for symbol in raw_bins
    )
    iq = np.concatenate(parts)
    if cfo_bins:
        sample = np.arange(len(iq), dtype=np.float64)
        iq = iq * np.exp(
            2j * np.pi * cfo_bins * sample / waveform_config.samples_per_symbol
        )
    return np.conjugate(iq) if config.inverted_iq else iq


def _bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _value(bits: list[int]) -> int:
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def _encode_codeword(nibble: int, cr: int) -> int:
    """Literal test oracle from pinned hamming_enc_impl.cc."""

    data = _bits(nibble, 4)
    if cr == 1:
        parity = data[0] ^ data[1] ^ data[2] ^ data[3]
        return (
            (data[3] << 4)
            | (data[2] << 3)
            | (data[1] << 2)
            | (data[0] << 1)
            | parity
        )
    p0 = data[3] ^ data[2] ^ data[1]
    p1 = data[2] ^ data[1] ^ data[0]
    p2 = data[3] ^ data[2] ^ data[0]
    p3 = data[3] ^ data[1] ^ data[0]
    full = (
        (data[3] << 7)
        | (data[2] << 6)
        | (data[1] << 5)
        | (data[0] << 4)
        | (p0 << 3)
        | (p1 << 2)
        | (p2 << 1)
        | p3
    )
    return full >> (4 - cr)


def _inverse_gray(value: int) -> int:
    output = value
    for shift in range(1, 12):
        output ^= value >> shift
    return output


def _interleave(codewords: list[int], sf: int, cr: int, reduced: bool) -> list[int]:
    """Literal transmit-side differential oracle; not a production encoder."""

    sf_app = sf - 2 if reduced else sf
    cw_len = cr + 4
    padded = codewords + [0] * (sf_app - len(codewords))
    output: list[int] = []
    for i in range(cw_len):
        row = [0] * sf
        for j in range(sf_app):
            row[j] = _bits(padded[(i - j - 1) % sf_app], cw_len)[i]
        if reduced:
            row[sf_app] = sum(row) % 2
        interleaved = _value(row)
        transmitted_bin = (_inverse_gray(interleaved) + 1) % (1 << sf)
        # fft_demod subtracts one and divides reduced-rate symbols by four.
        output.append(((transmitted_bin - 1) % (1 << sf)) // (4 if reduced else 1))
    return output


def _header_nibbles(length: int, cr: int, has_crc: bool) -> list[int]:
    a, b, c = length >> 4, length & 0xF, (cr << 1) | has_crc
    a3, a2, a1, a0 = _bits(a, 4)
    b3, b2, b1, b0 = _bits(b, 4)
    c3, c2, c1, c0 = _bits(c, 4)
    checksum = (
        ((a3 ^ a2 ^ a1 ^ a0) << 4)
        | ((a3 ^ b3 ^ b2 ^ b1 ^ c0) << 3)
        | ((a2 ^ b3 ^ b0 ^ c3 ^ c1) << 2)
        | ((a1 ^ b2 ^ b0 ^ c2 ^ c1 ^ c0) << 1)
        | (a0 ^ b1 ^ c3 ^ c2 ^ c1 ^ c0)
    )
    return [a, b, c, checksum >> 4, checksum & 0xF]


def _source_construction(
    payload: bytes,
    *,
    sf: int,
    cr: int,
    has_crc: bool,
    explicit: bool,
    ldro: bool,
    bad_header: bool = False,
    bad_crc: bool = False,
) -> list[int]:
    """Compose pinned upstream TX stages as a differential construction."""

    nibbles: list[int] = []
    for index, byte in enumerate(payload):
        whitened = byte ^ WHITENING_SEQUENCE[index]
        nibbles.extend((whitened & 0xF, whitened >> 4))
    if explicit:
        header = _header_nibbles(len(payload), cr, has_crc)
        if bad_header:
            header[4] ^= 1
        nibbles = header + nibbles
    if has_crc:
        crc = lora_payload_crc(payload) ^ int(bad_crc)
        nibbles.extend((crc >> (4 * index)) & 0xF for index in range(4))

    first, remaining = nibbles[: sf - 2], nibbles[sf - 2 :]
    symbols = _interleave([_encode_codeword(nibble, 4) for nibble in first], sf, 4, True)
    sf_app = sf - 2 if ldro else sf
    while remaining:
        block, remaining = remaining[:sf_app], remaining[sf_app:]
        symbols.extend(
            _interleave([_encode_codeword(nibble, cr) for nibble in block], sf, cr, ldro)
        )
    return symbols


def test_pinned_manifest_classifies_source_only_evidence():
    manifest = Path(__file__).parent / "fixtures" / "native_lora" / "MANIFEST.csv"
    artifacts = load_manifest(manifest)
    assert len(artifacts) == 20
    assert {artifact.source_commit for artifact in artifacts} == {
        _COMMIT,
        _TINYGS_COMMIT,
    }
    assert {artifact.license for artifact in artifacts} == {"GPL-3.0-only"}
    assert {artifact.evidence_class for artifact in artifacts} == {"upstream_oracle"}
    simulation = next(
        item for item in artifacts if item.artifact_id == "grlora-simulation-flowgraph"
    )
    assert "not an independent or hardware vector" in simulation.expected_output


def test_whitening_and_payload_crc_literals_are_exact():
    assert len(WHITENING_SEQUENCE) == 255
    assert WHITENING_SEQUENCE[:16].hex() == "fffefcf8f0e1c2850b172f5ebc78f1e3"
    assert WHITENING_SEQUENCE[-15:].hex() == "e5ca942850a142840913274f9f3f7f"
    assert lora_payload_crc(b"123456789") == 0xBEEF


@pytest.mark.parametrize("cr", range(1, 5))
def test_hamming_clean_codeword_matrix(cr: int):
    for nibble in range(16):
        decoded = hamming_decode_codeword(_encode_codeword(nibble, cr), cr)
        assert decoded.nibble == nibble
        assert not decoded.error_detected
        assert not decoded.data_bit_corrected


@pytest.mark.parametrize("cr", (3, 4))
def test_hamming_47_and_48_correct_every_single_bit(cr: int):
    for nibble in range(16):
        encoded = _encode_codeword(nibble, cr)
        for bit in range(cr + 4):
            decoded = hamming_decode_codeword(encoded ^ (1 << bit), cr)
            assert decoded.nibble == nibble


def test_literal_explicit_sf7_cr48_vector_decodes_byte_exactly():
    result = decode_lora_symbols(
        _LITERAL_SF7_CR48_VECTOR,
        LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False),
    )
    assert result.payload == b"123456789"
    assert result.header.payload_length == 9
    assert result.header.cr == 4
    assert result.header.has_crc
    assert result.header.checksum == 4
    assert result.crc_valid is True
    assert result.consumed_symbols == 40


@pytest.mark.parametrize("sf", range(7, 13))
@pytest.mark.parametrize("cr", range(1, 5))
@pytest.mark.parametrize("ldro", (False, True))
def test_explicit_source_differential_matrix(sf: int, cr: int, ldro: bool):
    payload = bytes((0x00, 0x7F, 0x80, 0xFF, sf, cr, int(ldro), 0x55, 0xAA))
    symbols = _source_construction(
        payload,
        sf=sf,
        cr=cr,
        has_crc=True,
        explicit=True,
        ldro=ldro,
    )
    result = decode_lora_symbols(
        symbols,
        LoRaPhyConfig(sf=sf, bandwidth=125_000, explicit_header=True, ldro=ldro),
    )
    assert result.payload == payload
    assert result.header.cr == cr
    assert result.ldro is ldro


@pytest.mark.parametrize("sf", range(5, 13))
@pytest.mark.parametrize("cr", range(1, 5))
@pytest.mark.parametrize("has_crc", (False, True))
def test_implicit_source_differential_matrix(sf: int, cr: int, has_crc: bool):
    payload = bytes((sf, cr, 0x00, 0xFF, 0x35, 0xCA))
    ldro = sf >= 11
    symbols = _source_construction(
        payload,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        explicit=False,
        ldro=ldro,
    )
    config = LoRaPhyConfig(
        sf=sf,
        bandwidth=125_000,
        explicit_header=False,
        cr=cr,
        payload_length=len(payload),
        has_crc=has_crc,
        ldro=ldro,
    )
    result = decode_lora_symbols(symbols, config)
    assert result.payload == payload
    assert result.header.explicit is False
    assert result.crc_valid is (True if has_crc else None)


def test_auto_ldro_uses_strict_sixteen_millisecond_threshold():
    assert not LoRaPhyConfig(sf=10, bandwidth=64_000, explicit_header=True).resolved_ldro
    assert LoRaPhyConfig(sf=10, bandwidth=63_999, explicit_header=True).resolved_ldro
    assert LoRaPhyConfig(sf=11, bandwidth=125_000, explicit_header=True).resolved_ldro


def test_bad_header_crc_and_payload_crc_fail_closed():
    payload = b"integrity"
    bad_header = _source_construction(
        payload,
        sf=7,
        cr=4,
        has_crc=True,
        explicit=True,
        ldro=False,
        bad_header=True,
    )
    config = LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False)
    with pytest.raises(LoRaFrameError, match="header checksum"):
        decode_lora_symbols(bad_header, config)

    bad_crc = _source_construction(
        payload,
        sf=7,
        cr=4,
        has_crc=True,
        explicit=True,
        ldro=False,
        bad_crc=True,
    )
    with pytest.raises(LoRaIntegrityError, match="payload CRC"):
        decode_lora_symbols(bad_crc, config)
    diagnostic = decode_lora_symbols(bad_crc, config, require_valid_crc=False)
    assert diagnostic.payload == payload
    assert diagnostic.crc_valid is False


def test_single_symbol_errors_are_corrected_but_double_error_reaches_crc_gate():
    one_error = list(_LITERAL_SF7_CR48_VECTOR)
    one_error[8] ^= 1
    corrected = decode_lora_symbols(
        one_error,
        LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False),
    )
    assert corrected.payload == b"123456789"
    assert corrected.corrected_codewords == 1

    two_errors = list(one_error)
    two_errors[15] ^= 1
    with pytest.raises(LoRaIntegrityError):
        decode_lora_symbols(
            two_errors,
            LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False),
        )


def test_max_payload_and_frame_bounds_are_enforced():
    payload = bytes(range(255))
    symbols = _source_construction(
        payload,
        sf=12,
        cr=4,
        has_crc=True,
        explicit=True,
        ldro=True,
    )
    result = decode_lora_symbols(
        symbols,
        LoRaPhyConfig(sf=12, bandwidth=125_000, explicit_header=True, ldro=True),
    )
    assert result.payload == payload

    with pytest.raises(LoRaFrameError, match="symbol bound"):
        decode_lora_symbols(
            [0] * (MAX_FRAME_SYMBOLS + 1),
            LoRaPhyConfig(sf=7, bandwidth=125_000),
        )


def test_truncation_trailing_and_symbol_width_are_rejected():
    config = LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False)
    with pytest.raises(LoRaFrameError, match="truncated"):
        decode_lora_symbols(_LITERAL_SF7_CR48_VECTOR[:-1], config)
    with pytest.raises(LoRaFrameError, match="trailing symbols"):
        decode_lora_symbols(_LITERAL_SF7_CR48_VECTOR + (0,), config)
    invalid_symbol = list(_LITERAL_SF7_CR48_VECTOR)
    invalid_symbol[0] = 32
    with pytest.raises(LoRaFrameError, match="5-bit boundary"):
        decode_lora_symbols(invalid_symbol, config)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"sf": 4, "bandwidth": 125_000}, "sf"),
        ({"sf": 13, "bandwidth": 125_000}, "sf"),
        ({"sf": 7, "bandwidth": 0}, "bandwidth"),
        ({"sf": 6, "bandwidth": 125_000}, "explicit-header"),
        (
            {"sf": 7, "bandwidth": 125_000, "explicit_header": False},
            "implicit-header cr",
        ),
    ],
)
def test_configuration_fails_closed(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        LoRaPhyConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("inverted_iq", "cfo_bins"),
    ((False, -1.75), (False, 0.375), (True, -0.625), (True, 1.5)),
)
def test_synthetic_iq_recovers_literal_payload_cfo_inversion_and_offsets(
    inverted_iq: bool, cfo_bins: float
):
    modem_config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=125_000,
        inverted_iq=inverted_iq,
    )
    iq = _synthetic_lora_iq(
        _LITERAL_SF7_CR48_VECTOR,
        modem_config,
        cfo_bins=cfo_bins,
    )

    sync = find_lora_sync(iq, modem_config)
    demodulated = demodulate_payload_bins(
        iq, modem_config, sync, len(_LITERAL_SF7_CR48_VECTOR)
    )
    symbols = normalize_framing_symbols(
        demodulated.bins,
        sf=modem_config.sf,
        ldro=False,
    )
    decoded = decode_lora_symbols(
        symbols,
        LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False),
    )

    assert sync.preamble_start == 23
    assert sync.payload_start == 23 + (
        modem_config.preamble_length + 4
    ) * modem_config.samples_per_symbol + modem_config.samples_per_symbol // 4
    assert sync.cfo_bins == pytest.approx(cfo_bins, abs=2e-5)
    assert sync.sync_bins == modem_config.sync_symbols
    assert sync.inverted_iq is inverted_iq
    assert demodulated.source_start == sync.payload_start
    assert demodulated.source_end == len(iq)
    assert min(demodulated.peak_ratios) > 0.999
    assert symbols == _LITERAL_SF7_CR48_VECTOR
    assert decoded.payload == b"123456789"
    assert decoded.crc_valid is True


@pytest.mark.parametrize(
    ("sf", "oversampling"),
    ((5, 1), (7, 2), (9, 4), (12, 1)),
)
def test_synthetic_sync_covers_sf_and_oversampling_boundaries(
    sf: int, oversampling: int
):
    config = LoRaModemConfig(
        sf=sf,
        bandwidth=125_000,
        sample_rate=125_000 * oversampling,
    )
    iq = _synthetic_lora_iq((), config, leading_samples=31)

    sync = find_lora_sync(iq, config)

    assert sync.preamble_start == 31
    assert sync.payload_start == len(iq)
    assert sync.sync_bins == config.sync_symbols
    assert sync.preamble_correlation > 0.999
    assert sync.downchirp_correlation > 0.999


def test_sync_rejects_wrong_word_noise_inversion_and_input_bounds():
    config = LoRaModemConfig(sf=7, bandwidth=125_000, sample_rate=125_000)
    wrong_sync = _synthetic_lora_iq((), config, sync_word=0x34)
    with pytest.raises(LoRaSyncError, match="sync validation"):
        find_lora_sync(wrong_sync, config)

    noise = np.random.default_rng(0x4C4F5241).normal(
        size=config.minimum_sync_samples
    ) + 1j * np.random.default_rng(0x53534154).normal(
        size=config.minimum_sync_samples
    )
    with pytest.raises(LoRaSyncError, match="correlation"):
        find_lora_sync(noise, config)

    inverted = np.conjugate(_synthetic_lora_iq((), config))
    with pytest.raises(LoRaSyncError):
        find_lora_sync(inverted, config)

    bounded = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=125_000,
        max_input_samples=config.minimum_sync_samples,
    )
    with pytest.raises(ValueError, match="sample bound"):
        find_lora_sync(np.zeros(config.minimum_sync_samples + 1), bounded)


def test_sync_buffer_preserves_absolute_offsets_and_is_bounded():
    base = LoRaModemConfig(sf=7, bandwidth=125_000, sample_rate=125_000)
    iq = _synthetic_lora_iq((), base, leading_samples=47)
    config = LoRaModemConfig(
        sf=base.sf,
        bandwidth=base.bandwidth,
        sample_rate=base.sample_rate,
        max_input_samples=len(iq),
    )
    buffer = LoRaSyncBuffer(config)
    cuts = (17, 3 * config.samples_per_symbol + 5, len(iq) - 1)
    start = 0
    for stop in cuts:
        assert buffer.feed(iq[start:stop]) is None
        start = stop

    sync = buffer.feed(iq[start:])
    assert sync is not None
    assert sync.preamble_start == 47
    assert sync.payload_start == len(iq)
    assert buffer.sample_count == len(iq)

    with pytest.raises(ValueError, match="buffered IQ"):
        buffer.feed(np.zeros(1, dtype=np.complex128))

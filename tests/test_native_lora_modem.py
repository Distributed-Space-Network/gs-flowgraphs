from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from native_lora import (
    LoRaModemConfig,
    LoRaPhyConfig,
    LoRaSyncBuffer,
    LoRaSyncError,
    build_upchirp,
    decode_lora_symbols,
    demodulate_payload_bins,
    find_lora_sync,
    normalize_framing_symbols,
)
from numpy.typing import NDArray

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


def _synthetic_frame(
    config: LoRaModemConfig,
    raw_payload_bins: tuple[int, ...],
    *,
    prefix: int = 37,
    cfo_bins: float = 0.0,
    emit_sync_word: int | None = None,
    valid_downchirps: bool = True,
) -> NDArray[np.complex128]:
    upchirp = build_upchirp(0, config)
    downchirp = np.conjugate(upchirp)
    wire_sync = config.sync_word if emit_sync_word is None else emit_sync_word
    sync_symbols = (((wire_sync >> 4) & 0xF) << 3, (wire_sync & 0xF) << 3)
    down = downchirp if valid_downchirps else upchirp
    pieces = (
        [np.zeros(prefix, dtype=np.complex128)]
        + [upchirp] * config.preamble_length
        + [build_upchirp(symbol, config) for symbol in sync_symbols]
        + [down, down, down[: config.samples_per_symbol // 4]]
        + [
            build_upchirp((symbol + 1) % config.bins, config)
            for symbol in raw_payload_bins
        ]
    )
    samples = np.concatenate(pieces)
    if cfo_bins:
        index = np.arange(len(samples), dtype=np.float64)
        samples *= np.exp(2j * np.pi * cfo_bins * index / config.samples_per_symbol)
    return np.conjugate(samples) if config.inverted_iq else samples


def _timing_warp(
    samples: NDArray[np.complex128], *, sto_samples: float, sfo_ppm: float
) -> NDArray[np.complex128]:
    scale = 1.0 + sfo_ppm / 1_000_000.0
    output_length = int(np.ceil(sto_samples + (len(samples) - 1) * scale)) + 2
    observed = np.arange(output_length, dtype=np.float64)
    ideal = (observed - sto_samples) / scale
    source = np.arange(len(samples), dtype=np.float64)
    return np.interp(ideal, source, samples.real, left=0.0, right=0.0) + 1j * np.interp(
        ideal, source, samples.imag, left=0.0, right=0.0
    )


@pytest.mark.parametrize("oversampling", (1, 2, 4))
@pytest.mark.parametrize("cfo_bins", (-1.25, -0.25, 0.0, 0.25, 1.25))
@pytest.mark.parametrize("inverted", (False, True))
def test_sync_cfo_iq_inversion_and_fft_demod_matrix(
    oversampling: int, cfo_bins: float, inverted: bool
):
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=125_000 * oversampling,
        sync_word=0x12,
        inverted_iq=inverted,
        min_correlation=0.40,
    )
    raw_bins = (0, 1, 17, 63, 127)
    samples = _synthetic_frame(config, raw_bins, cfo_bins=cfo_bins)
    sync = find_lora_sync(samples, config)
    demodulated = demodulate_payload_bins(samples, config, sync, len(raw_bins))

    assert sync.preamble_start == 37
    assert sync.payload_start == 37 + config.minimum_sync_samples
    assert sync.cfo_bins == pytest.approx(cfo_bins, abs=2e-5)
    assert sync.sync_bins == config.sync_symbols
    assert sync.preamble_correlation >= config.min_correlation
    assert sync.downchirp_correlation > 0.99
    assert sync.inverted_iq is inverted
    assert sync.sto_samples == 0.0
    assert sync.sfo_ppm == 0.0
    assert sync.snr_db > 70.0
    assert demodulated.bins == raw_bins
    assert min(demodulated.peak_ratios) > 0.99
    assert demodulated.source_start == sync.payload_start
    assert demodulated.source_end == len(samples)


@pytest.mark.parametrize(
    ("sto_samples", "sfo_ppm"),
    ((-0.4, -150.0), (0.35, 150.0)),
)
def test_fractional_sto_sfo_are_estimated_and_compensated(
    sto_samples: float, sfo_ppm: float
):
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=250_000,
        min_correlation=0.25,
    )
    raw_bins = (0, 1, 17, 63, 127, 88, 42, 5, 99, 100, 7, 11)
    prefix = 80
    ideal = _synthetic_frame(config, raw_bins, prefix=prefix, cfo_bins=0.25)
    samples = _timing_warp(
        ideal, sto_samples=sto_samples, sfo_ppm=sfo_ppm
    )

    sync = find_lora_sync(samples, config)
    demodulated = demodulate_payload_bins(samples, config, sync, len(raw_bins))
    expected_start = sto_samples + prefix * (1.0 + sfo_ppm / 1_000_000.0)

    assert sync.preamble_start + sync.sto_samples == pytest.approx(
        expected_start, abs=0.2
    )
    assert sync.sfo_ppm == pytest.approx(sfo_ppm, abs=120.0)
    assert np.sign(sync.sfo_ppm) == np.sign(sfo_ppm)
    assert sync.cfo_bins == pytest.approx(0.25, abs=3e-4)
    assert demodulated.bins == raw_bins
    assert min(demodulated.peak_ratios) > 0.70
    assert 0 <= demodulated.source_start < demodulated.source_end <= len(samples)


def test_dechirped_snr_gate_accepts_qualified_signal_and_rejects_lower_snr():
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=250_000,
        min_correlation=0.03,
        min_snr_db=-8.0,
        max_abs_sfo_ppm=1_000.0,
    )
    raw_bins = (0, 1, 17, 63, 127)
    clean = _synthetic_frame(config, raw_bins, prefix=80, cfo_bins=0.25)

    def noisy(input_snr_db: float) -> NDArray[np.complex128]:
        generator = np.random.default_rng(12345)
        sigma = np.sqrt(1.0 / (2.0 * 10 ** (input_snr_db / 10.0)))
        return clean + sigma * (
            generator.normal(size=len(clean))
            + 1j * generator.normal(size=len(clean))
        )

    qualified = noisy(-6.0)
    sync = find_lora_sync(qualified, config)
    demodulated = demodulate_payload_bins(qualified, config, sync, len(raw_bins))
    assert sync.snr_db >= config.min_snr_db
    assert demodulated.bins == raw_bins

    with pytest.raises(LoRaSyncError, match="sync validation"):
        find_lora_sync(noisy(-9.0), config)


def test_synthetic_iq_replays_through_literal_phy_vector():
    config = LoRaModemConfig(sf=7, bandwidth=125_000, sample_rate=250_000)
    raw_bins = tuple(
        symbol * 4 if index < 8 else symbol
        for index, symbol in enumerate(_LITERAL_SF7_CR48_VECTOR)
    )
    samples = _synthetic_frame(config, raw_bins, prefix=91, cfo_bins=0.25)
    sync = find_lora_sync(samples, config)
    demodulated = demodulate_payload_bins(samples, config, sync, len(raw_bins))
    framing_symbols = normalize_framing_symbols(demodulated.bins, sf=7, ldro=False)
    result = decode_lora_symbols(
        framing_symbols,
        LoRaPhyConfig(sf=7, bandwidth=125_000, explicit_header=True, ldro=False),
    )
    assert framing_symbols == _LITERAL_SF7_CR48_VECTOR
    assert result.payload == b"123456789"
    assert result.crc_valid is True


def test_sync_word_and_downchirp_validation_reject_false_candidates():
    config = LoRaModemConfig(sf=7, bandwidth=125_000, sample_rate=125_000)
    wrong_sync = _synthetic_frame(config, (0,), emit_sync_word=0x34)
    with pytest.raises(LoRaSyncError, match="sync validation"):
        find_lora_sync(wrong_sync, config)

    wrong_downchirps = _synthetic_frame(config, (0,), valid_downchirps=False)
    with pytest.raises(LoRaSyncError, match="sync validation"):
        find_lora_sync(wrong_downchirps, config)


def test_noise_and_zero_input_do_not_create_a_frame():
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=125_000,
        min_correlation=0.65,
    )
    length = config.minimum_sync_samples + 4 * config.samples_per_symbol
    with pytest.raises(LoRaSyncError):
        find_lora_sync(np.zeros(length, dtype=np.complex128), config)
    for seed in range(8):
        generator = np.random.default_rng(seed)
        noise = generator.normal(size=length) + 1j * generator.normal(size=length)
        with pytest.raises(LoRaSyncError):
            find_lora_sync(noise, config)


def test_chunk_buffer_matches_one_shot_and_is_bounded():
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=250_000,
        max_input_samples=10_000,
    )
    samples = _synthetic_frame(config, (0, 1, 2), prefix=73, cfo_bins=-0.25)
    expected = find_lora_sync(samples, config)
    buffer = LoRaSyncBuffer(config)
    found = None
    cursor = 0
    sizes = (1, 17, 509, 3, 1024, 211, 7, 4096)
    for size in sizes:
        if cursor >= len(samples):
            break
        found = buffer.feed(samples[cursor : cursor + size]) or found
        cursor += size
    if cursor < len(samples):
        found = buffer.feed(samples[cursor:]) or found
    assert found == expected
    assert buffer.sample_count == len(samples)

    small = replace(config, max_input_samples=config.minimum_sync_samples)
    bounded = LoRaSyncBuffer(small)
    bounded.feed(np.zeros(small.minimum_sync_samples, dtype=np.complex128))
    with pytest.raises(ValueError, match="buffered IQ"):
        bounded.feed(np.zeros(1, dtype=np.complex128))


def test_reduced_rate_normalization_is_explicit_and_bounded():
    raw = (4, 8, 12, 16, 20, 24, 28, 32, 33, 34)
    assert normalize_framing_symbols(raw, sf=7, ldro=False) == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        33,
        34,
    )
    assert normalize_framing_symbols(raw, sf=7, ldro=True) == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        8,
        8,
    )
    with pytest.raises(ValueError, match="bin 0"):
        normalize_framing_symbols((128,), sf=7, ldro=False)


def test_iq_shapes_finiteness_lengths_and_output_bounds_fail_closed():
    config = LoRaModemConfig(
        sf=7,
        bandwidth=125_000,
        sample_rate=125_000,
        max_input_samples=4096,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        find_lora_sync(np.zeros((10, 10), dtype=np.complex128), config)
    nonfinite = np.zeros(config.minimum_sync_samples, dtype=np.complex128)
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        find_lora_sync(nonfinite, config)
    with pytest.raises(ValueError, match="sample bound"):
        find_lora_sync(np.zeros(4097, dtype=np.complex128), config)
    with pytest.raises(LoRaSyncError, match="shorter"):
        find_lora_sync(np.zeros(config.minimum_sync_samples - 1), config)

    complete = _synthetic_frame(config, (0,))
    sync = find_lora_sync(complete, config)
    with pytest.raises(ValueError, match="requested aligned"):
        demodulate_payload_bins(complete, config, sync, 2)
    with pytest.raises(ValueError, match="1..4096"):
        demodulate_payload_bins(complete, config, sync, 0)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"sf": 4, "bandwidth": 125_000, "sample_rate": 125_000}, "sf"),
        ({"sf": 7, "bandwidth": 0, "sample_rate": 125_000}, "positive"),
        (
            {"sf": 7, "bandwidth": 125_000, "sample_rate": 200_000},
            "integer multiple",
        ),
        ({"sf": 7, "bandwidth": 125_000, "sample_rate": 125_000, "sync_word": 256}, "octet"),
        (
            {"sf": 7, "bandwidth": 125_000, "sample_rate": 125_000, "preamble_length": 4},
            "at least five",
        ),
        (
            {"sf": 7, "bandwidth": 125_000, "sample_rate": 125_000, "max_abs_sto_samples": 2.0},
            "max_abs_sto_samples",
        ),
        (
            {"sf": 7, "bandwidth": 125_000, "sample_rate": 125_000, "max_abs_sfo_ppm": 5_001.0},
            "max_abs_sfo_ppm",
        ),
        (
            {"sf": 7, "bandwidth": 125_000, "sample_rate": 125_000, "min_snr_db": -41.0},
            "min_snr_db",
        ),
    ],
)
def test_modem_configuration_fails_closed(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        LoRaModemConfig(**kwargs)  # type: ignore[arg-type]

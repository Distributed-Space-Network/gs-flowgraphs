"""CCSDS/conventional RS vectors, boundaries, shortening, and interleaving."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from native_framing.provenance import load_manifest
from native_framing.rs import CcsdsReedSolomon, ccsds_generator_log_coefficients

_TEST_FILE = Path(__file__).resolve()
_KPLABS = _TEST_FILE.parents[2] / "related-projects" / "reed-solomon-ccsds"
_KPLABS_HASHES = {
    "LICENSE": "885a03f54b157961236f46843e79972abfcd6890b6cbb368bc7eca328ff95a12",
    "reed_solomon_ccsds/reed_solomon.py": (
        "0dc3dcfab52c516c8d6a5359bc55dd9544edf8aa11daf35eebd3fd5a1742c10c"
    ),
    "tests/test_reed_solomon.py": (
        "b82a3e8d53cb91d187215394fbc41530e4d38e4a8d300195a9e1f25446d2a235"
    ),
}

# Literal LGPL-2.1 KP Labs vectors at the commit above.  Keeping the parity
# bytes literal prevents the repository-owned encoder from acting as its own
# oracle.
_KPLABS_CONVENTIONAL_PARITY = bytes.fromhex(
    "2f bd 4f b4 74 84 94 b9 ac d5 54 62 72 12 ee b3"
    " eb ed 41 19 1d e1 d3 63 20 ea 49 29 0b 25 ab cf"
)


def _kplabs_literal(name: str) -> bytes:
    """Read one bytes([...]) vector from the hash-pinned oracle test source."""

    source = (_KPLABS / "tests/test_reed_solomon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Call) or len(statement.value.args) != 1:
            break
        values = ast.literal_eval(statement.value.args[0])
        return bytes(values)
    raise AssertionError(f"pinned KP Labs vector {name!r} is absent")

_UPSTREAM_CCSDS_POLY = (
    0, 249, 59, 66, 4, 43, 126, 251, 97, 30, 3, 213, 50, 66, 170, 5, 24,
    5, 170, 66, 50, 213, 3, 30, 97, 251, 126, 43, 4, 66, 59, 249, 0,
)


def test_ccsds_generator_polynomial_matches_pinned_libfec_table_exactly():
    assert ccsds_generator_log_coefficients() == _UPSTREAM_CCSDS_POLY


def test_rs_matches_pinned_independent_kplabs_codewords() -> None:
    artifacts = {
        artifact.source_path: artifact
        for artifact in load_manifest(
            _TEST_FILE.parent / "fixtures" / "native_framing" / "MANIFEST.csv"
        )
        if artifact.source_commit == "b75e8c9d497fbbca5f5f518700f05ec6c897a2bd"
    }
    assert set(artifacts) == set(_KPLABS_HASHES)
    for relative, expected_hash in _KPLABS_HASHES.items():
        source = _KPLABS / relative
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
        artifact = artifacts[relative]
        assert artifact.sha256 == expected_hash
        assert artifact.license == "LGPL-2.1-only"
        assert artifact.evidence_class == "independent_oracle"

    conventional = CcsdsReedSolomon(basis="conventional")
    conventional_vector = _kplabs_literal("GOOD_BLOCK")
    assert conventional_vector == bytes(range(223)) + _KPLABS_CONVENTIONAL_PARITY
    assert conventional.encode(bytes(range(223))) == conventional_vector

    dual = CcsdsReedSolomon(basis="dual")
    dual_vector = _kplabs_literal("GOOD_BLOCK_DUAL_BASIS")
    assert dual.encode(dual_vector[:223]) == dual_vector

    interleaved = CcsdsReedSolomon(basis="conventional", interleaving=4)
    payload = bytes([0x00, 0xFF, 0xDE, 0xAD]) * 223
    assert interleaved.encode(payload) == bytes([0x00, 0xFF, 0xDE, 0xAD]) * 255


@pytest.mark.parametrize("basis", ["conventional", "dual"])
@pytest.mark.parametrize("interleaving", [1, 2, 4, 5])
@pytest.mark.parametrize("path_size", [1, 17, 111, 223])
def test_rs_clean_roundtrip_shortening_and_interleaving(
    basis: str, interleaving: int, path_size: int
):
    size = path_size * interleaving
    payload = bytes(np.random.default_rng(size).integers(0, 256, size, dtype=np.uint8))
    codec = CcsdsReedSolomon(basis=basis, interleaving=interleaving)
    codeword = codec.encode(payload)
    assert len(codeword) == len(payload) + 32 * interleaving
    result = codec.decode(codeword)
    assert result is not None
    assert result.payload == payload
    assert result.corrected_symbols == 0


@pytest.mark.parametrize("basis", ["conventional", "dual"])
def test_rs_corrects_t_and_rejects_t_plus_one_per_path(basis: str):
    payload = bytes(range(223))
    codec = CcsdsReedSolomon(basis=basis)
    codeword = bytearray(codec.encode(payload))
    for position in range(16):
        codeword[position * 11] ^= position + 1
    result = codec.decode(codeword)
    assert result is not None
    assert result.payload == payload
    assert result.corrected_symbols == 16

    beyond = bytearray(codec.encode(payload))
    for position in range(17):
        beyond[position * 11] ^= position + 1
    assert codec.decode(beyond) is None


def test_rs_interleaving_corrects_each_path_independently():
    codec = CcsdsReedSolomon(basis="dual", interleaving=2)
    payload = bytes(range(223)) * 2
    codeword = bytearray(codec.encode(payload))
    for path in range(2):
        for error in range(16):
            codeword[path + error * 2] ^= error + 1
    result = codec.decode(codeword)
    assert result is not None
    assert result.payload == payload
    assert result.corrected_symbols == 32


def test_rs_conventional_and_dual_basis_codewords_cross_reject() -> None:
    payload = bytes(range(223))
    conventional = CcsdsReedSolomon(basis="conventional")
    dual = CcsdsReedSolomon(basis="dual")
    assert conventional.decode(dual.encode(payload)) is None
    assert dual.decode(conventional.encode(payload)) is None


@pytest.mark.parametrize("basis", ["conventional", "dual"])
@pytest.mark.parametrize("interleaving", [1, 2, 5])
@pytest.mark.parametrize("path_size", [33, 47, 223])
def test_rs_erasures_extend_shortened_interleaved_correction_per_path(
    basis: str,
    interleaving: int,
    path_size: int,
) -> None:
    payload_size = (path_size - 32) * interleaving
    payload = bytes(
        np.random.default_rng(payload_size + interleaving).integers(
            0, 256, payload_size, dtype=np.uint8
        )
    )
    codec = CcsdsReedSolomon(basis=basis, interleaving=interleaving)
    original = codec.encode(payload)
    damaged = bytearray(original)
    erasures: list[int] = []
    expected_corrections = 0
    for path in range(interleaving):
        # Ten unknown errors plus twelve known erasures exactly consume the
        # path's 32 parity-symbol budget: 2*10 + 12 == 32.
        local_positions = list(range(22))
        for sequence, local in enumerate(local_positions, start=1):
            wire_position = path + local * interleaving
            damaged[wire_position] ^= (17 * sequence + path) & 0xFF or 1
        erasures.extend(path + local * interleaving for local in local_positions[10:])
        expected_corrections += len(local_positions)

    assert codec.decode(bytes(damaged)) is None
    result = codec.decode(bytes(damaged), erase_pos=erasures)
    assert result is not None
    assert result.payload == payload
    assert result.corrected_symbols == expected_corrections


def test_rs_erasure_limit_and_wire_position_mapping_are_fail_closed() -> None:
    codec = CcsdsReedSolomon(basis="dual", interleaving=2)
    payload = bytes(range(223)) * 2
    original = codec.encode(payload)

    damaged = bytearray(original)
    one_path = [2 * index for index in range(33)]
    for position in one_path:
        damaged[position] ^= 0xA5
    assert codec.decode(bytes(damaged), erase_pos=one_path) is None

    with pytest.raises(ValueError, match="out of range"):
        codec.decode(original, erase_pos=[len(original)])
    with pytest.raises(ValueError, match="out of range"):
        codec.decode(original, erase_pos=[-1])
    for invalid in ([True], [1.5], ["1"]):
        with pytest.raises(TypeError, match="integers"):
            codec.decode(original, erase_pos=invalid)  # type: ignore[arg-type]

    # Duplicate erasure coordinates describe one wire symbol, not two parity
    # costs, and generators are consumed exactly once.
    duplicate = bytearray(original)
    duplicate[7] ^= 0x44
    result = codec.decode(bytes(duplicate), erase_pos=(value for value in [7, 7]))
    assert result is not None
    assert result.payload == payload
    assert result.corrected_symbols == 1


def test_rs_validation_is_fail_closed():
    with pytest.raises(ValueError, match="basis"):
        CcsdsReedSolomon(basis="guess")
    with pytest.raises(ValueError, match="positive"):
        CcsdsReedSolomon(interleaving=0)
    codec = CcsdsReedSolomon(interleaving=2)
    with pytest.raises(ValueError, match="divisible"):
        codec.encode(b"odd")
    with pytest.raises(ValueError, match="223"):
        codec.encode(bytes(448))
    assert codec.decode(bytes(65)) is None

"""CA-FLOW-003 — an EXPLICIT bad params file must fail the spawn.

The orchestrator VALIDATED the engine/framing/rates/gains/uplink settings before
writing params.json. load_params used to return {} for a missing/torn/malformed/
non-object EXPLICIT file, silently substituting the app's built-in defaults for the
validated configuration. Only an OMITTED --params-file may mean defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from _spawn_contract import ParamsFileError, load_params  # noqa: E402

pytestmark = []


def _args(params_file: str | None) -> argparse.Namespace:
    return argparse.Namespace(params_file=params_file)


def test_omitted_flag_means_defaults() -> None:
    assert load_params(_args(None)) == {}
    assert load_params(_args("")) == {}


def test_valid_object_round_trips(tmp_path: Path) -> None:
    f = tmp_path / "params.json"
    f.write_text('{"engine":"dsp","framing":"endurosat","baud":9600}', encoding="utf-8")
    assert load_params(_args(str(f))) == {"engine": "dsp", "framing": "endurosat", "baud": 9600}


def test_explicit_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ParamsFileError, match="not loadable"):
        load_params(_args(str(tmp_path / "nope.json")))


def test_explicit_malformed_json_fails(tmp_path: Path) -> None:
    f = tmp_path / "torn.json"
    f.write_text('{"engine": "dsp", "fra', encoding="utf-8")  # torn write
    with pytest.raises(ParamsFileError, match="not loadable"):
        load_params(_args(str(f)))


@pytest.mark.parametrize("body", ['["a","b"]', '"scalar"', "42", "null"])
def test_explicit_non_object_fails(tmp_path: Path, body: str) -> None:
    f = tmp_path / "shape.json"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ParamsFileError, match="not a JSON object"):
        load_params(_args(str(f)))


def test_explicit_unreadable_file_fails(tmp_path: Path) -> None:
    # A directory at the path raises OSError on open — the unreadable case.
    d = tmp_path / "dir.json"
    d.mkdir()
    with pytest.raises(ParamsFileError, match="not loadable"):
        load_params(_args(str(d)))

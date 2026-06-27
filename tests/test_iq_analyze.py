"""iq_analyze capture loading — the cf32 (whole-pass) path via the metadata sidecar.

The VSA CSV path is the EnduroSat lab workflow; the cf32 path lets the same tool decode
the WHOLE pass the GR engines record (not just a VSA window)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from iq_analyze import load_capture, load_cf32


def test_load_cf32_reads_sidecar(tmp_path: Path) -> None:
    cf = tmp_path / "p.cf32"
    np.zeros(1000, dtype=np.complex64).tofile(cf)
    (tmp_path / "p.cf32.json").write_text(
        '{"sample_rate_hz": 96000.0, "center_hz": 401000000.0, "format": "cf32le"}'
    )
    cap = load_cf32(cf)
    assert cap.fs == 96000.0
    assert cap.center_hz == 401000000.0
    assert len(cap.iq) == 1000


def test_load_cf32_sample_rate_fallback_when_no_sidecar(tmp_path: Path) -> None:
    cf = tmp_path / "n.cf32"
    np.zeros(10, dtype=np.complex64).tofile(cf)
    cap = load_cf32(cf, sample_rate_hz=48000.0)
    assert cap.fs == 48000.0


def test_load_capture_dispatches_and_requires_rate(tmp_path: Path) -> None:
    cf = tmp_path / "x.cf32"
    np.zeros(10, dtype=np.complex64).tofile(cf)
    with pytest.raises(ValueError):  # no sidecar + no --sample-rate → can't analyze
        load_capture(cf)
    cap = load_capture(cf, sample_rate_hz=48000.0)
    assert cap.fs == 48000.0

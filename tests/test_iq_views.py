"""Post-pass view derivation: PNG + CSV from a recorded .cf32 (pure numpy, no GNU Radio).

This is what gs-client runs AFTER the flowgraph exits, decoupled from the stop path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from iq_views import derive_views, main


def _capture(path: Path, n: int = 144_000, fs: float = 48_000.0) -> None:
    t = np.arange(n)
    iq = (0.2 * np.exp(2j * np.pi * 5_000 * t / fs)).astype(np.complex64)
    iq.tofile(path)


def test_derive_views_writes_png_and_csv(tmp_path: Path) -> None:
    cf32 = tmp_path / "cmd_47.cf32"
    _capture(cf32)
    written = derive_views(
        cf32, center_hz=401_510_000.0, sample_rate_hz=48_000.0, formats=("png", "csv")
    )
    assert cf32.with_suffix(".png").exists()
    assert cf32.with_suffix(".csv").exists()
    assert set(written) == {cf32.with_suffix(".png"), cf32.with_suffix(".csv")}


def test_derive_views_respects_requested_formats(tmp_path: Path) -> None:
    cf32 = tmp_path / "p.cf32"
    _capture(cf32)
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png",))
    assert cf32.with_suffix(".png").exists()
    assert not cf32.with_suffix(".csv").exists()


def test_torn_write_is_truncated_not_fatal(tmp_path: Path) -> None:
    # A SIGTERM mid-write can leave a non-multiple-of-8 file; memmap must still work.
    cf32 = tmp_path / "torn.cf32"
    _capture(cf32, n=60_000)
    with cf32.open("ab") as fh:
        fh.write(b"\x01\x02\x03")  # 3 stray bytes
    derive_views(cf32, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png", "csv"))
    assert cf32.with_suffix(".png").exists()
    assert cf32.with_suffix(".csv").exists()


def test_missing_or_empty_input_is_noop(tmp_path: Path) -> None:
    assert derive_views(tmp_path / "nope.cf32", center_hz=0.0, sample_rate_hz=48_000.0,
                        formats=("png", "csv")) == []
    empty = tmp_path / "empty.cf32"
    empty.write_bytes(b"")
    assert derive_views(empty, center_hz=0.0, sample_rate_hz=48_000.0, formats=("png", "csv")) == []


def test_cli_main(tmp_path: Path) -> None:
    cf32 = tmp_path / "cli.cf32"
    _capture(cf32)
    rc = main(["--input", str(cf32), "--sample-rate", "48000", "--center-hz", "401e6",
               "--formats", "png,csv", "--csv-seconds", "1"])
    assert rc == 0
    assert cf32.with_suffix(".png").exists() and cf32.with_suffix(".csv").exists()

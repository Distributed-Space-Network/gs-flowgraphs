"""Import-free checks for the bench-only GNU Radio native-framing wiring."""

from __future__ import annotations

from pathlib import Path

_APPS = Path(__file__).parents[1] / "apps"


def test_gnuradio_live_path_is_explicitly_gated_and_streaming() -> None:
    source = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")
    assert 'os.environ.get("GS_NATIVE_FRAMING_LIVE"' in source
    assert "self._native_decoder = build_decoder(" in source
    assert "self._native_decoder.push(fresh)" in source
    assert "self._native_decoder.flush()" in source
    assert "def flush_frames(self) -> list[_DecodedFrame]" in source
    assert "native_live and native_profile is not None" in source
    assert "plan_native_rx_pairing(" in source
    assert "native_pairing_available" in source
    assert "SoftSymbolSink" in source
    assert "tb.connect(soft, native_sink)" in source
    assert "ShadowReconciler[_DecodedFrame]" in source
    assert "self._shadow.reconcile(our_frames, gr_frames)" in source
    assert "def finalize_shadow(self) -> ShadowStats | None" in source


def test_live_writer_preserves_metadata_without_wall_clock_fallback() -> None:
    source = (_APPS / "satellite_rx.py").read_text(encoding="utf-8")
    assert "decoded.source_start" in source
    assert '"source_offset_kind": source_offset_kind' in source
    assert '"timestamp_status": "unavailable"' in source
    assert "time.time()" not in source
    assert 'crc_ok = not integrity or integrity == "passed"' in source
    assert "for decoded in ctx.flush_frames():" in source
    assert '"event": "native_shadow_summary"' in source


def test_live_scheduler_handoffs_are_bounded_counted_and_fail_closed() -> None:
    gfsk = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
    satellites = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")

    assert gfsk.count("BoundedQueue[np.ndarray]") >= 2
    assert "_SYMBOL_QUEUE_CAPACITY_SYMBOLS" in gfsk
    assert "require_lossless(stats, label=label" in gfsk
    assert gfsk.count("def queue_stats(self) -> QueueStats") >= 2

    assert "BoundedQueue[bytes]" in satellites
    assert "_FRAME_QUEUE_CAPACITY_BYTES" in satellites
    assert 'require_lossless(stats, label="gr-satellites frame"' in satellites
    assert "def queue_stats(self) -> QueueStats" in satellites

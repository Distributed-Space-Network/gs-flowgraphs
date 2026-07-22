"""Import-free checks for the bench-only GNU Radio native-framing wiring."""

from __future__ import annotations

from pathlib import Path

_APPS = Path(__file__).parents[1] / "apps"


def test_gnuradio_live_path_is_explicitly_gated_and_streaming() -> None:
    source = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")
    assert 'os.environ.get("GS_NATIVE_FRAMING_LIVE"' in source
    assert "self._native_decoders.append(" in source
    assert "decoder.push(fresh.copy())" in source
    assert "decoder.flush()" in source
    assert "def flush_frames(self) -> list[_DecodedFrame]" in source
    assert "native_live and native_profile_available" in source
    assert "local_deframer_enabled = native_pairing_available" in source
    assert "should_build_demod(" in source
    assert "plan_native_rx_pairing(" in source
    assert "native_pairing_available" in source
    assert "SoftSymbolSink" in source
    assert "collect_hard=collect_hard" in source
    assert "if sink is not None and (legacy_hard_enabled or native_hard):" in source
    gfsk = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
    assert "blocks.null_sink(gr.sizeof_char)" in gfsk
    assert "tb.connect(soft, soft_sink)" in source
    assert "framings_list=framing_labels" in source
    assert "native_framings=tuple(native_hard)" in source
    assert "native_framings=tuple(native_soft)" in source
    assert "make_grsat_deframers(framing_labels)" in source
    assert "for label, decoder in grsat_deframers:" in source
    assert "tagged_sink = _FrameSink(label, sink._q)" in source
    assert "modes=[mode]" in source
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

    assert "BoundedQueue[_DecodedFrame]" in satellites
    assert "_FRAME_QUEUE_CAPACITY_BYTES" in satellites
    assert 'require_lossless(stats, label="gr-satellites frame"' in satellites
    assert "def queue_stats(self) -> QueueStats" in satellites


def test_live_fsk_has_pinned_grsat_and_satnogs_adaptive_paths() -> None:
    source = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
    function = source[
        source.index("def connect_gfsk_demod(") : source.index("def connect_psk_demod(")
    ]

    assert "deviation = float(profile.mod_index) * baud / 2.0" in function
    assert "from satellites.components.demodulators import" in function
    assert "fsk_demodulator," in function
    assert "soft = fsk_demodulator(" in function
    assert "iq=True" in function
    assert "deviation=deviation" in function
    assert "dc_block=dc_block" in function
    assert "if adaptive_centering:" in function
    assert "blocks.moving_average_ff(1024, 1.0 / 1024.0, 4096, 1)" in function
    assert "blocks.vco_c(float(sample_rate), -float(sample_rate), 1.0)" in function
    assert "0.625 * baud" in function
    assert "digital.clock_recovery_mm_ff(" in function
    assert "2.0 * math.pi / 100.0" in function
    assert "fll_band_edge_cc" not in function


def test_runtime_log_distinguishes_active_graph_from_capability_plan() -> None:
    source = (_APPS / "gnuradio_satellites.py").read_text(encoding="utf-8")

    assert '"active decode graph: local=%s grsat_components=%d grsat_monolithic=%s"' in source
    assert '"capability-only decode plan (not the active graph): %s"' in source
    assert '_log.info("decode plan: %s"' not in source


def test_installed_iq_replay_reuses_exact_live_demod_and_both_deframers() -> None:
    source = (_APPS / "iq_decode.py").read_text(encoding="utf-8")
    function = source[
        source.index("def decode_capture_gnuradio(") : source.index("def _append_frames(")
    ]

    assert "from gnuradio_satellites import" in function
    assert "make_grsat_deframers," in function
    assert "_unused_hard, soft = modem.build_demod(" in function
    assert 'mod_index=parameters.get("mod_index")' in function
    assert "upstream_deframers = make_grsat_deframers(labels)" in function
    assert "blocks.throttle(" in function
    assert "class _NativeReplaySink(gr.sync_block):" in function
    assert "self._fanout.push(chunk)" in function
    assert "class _ReplayDecoderFanout:" in source
    assert "for result in decoder.push(symbols.copy()):" in source
    assert "BoundedQueue[tuple[str, FrameResult]]" in source
    assert "SoftSymbolSink" not in function
    assert "_build_fallbacks(" not in function
    assert "while not completed.wait(_GNU_RADIO_DRAIN_PERIOD_S):" in function
    assert "native_sink.flush_results()" in function

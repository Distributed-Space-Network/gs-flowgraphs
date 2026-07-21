"""One hard-symbol handoff feeds every RF-equivalent local deframer."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from gfsk_ax25 import ax25, endurosat_link, framing


def _load_runtime_module(monkeypatch):
    class _BasicBlock:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def message_port_register_in(self, *args) -> None:
            del args

        def set_msg_handler(self, *args) -> None:
            del args

    pmt = ModuleType("pmt")
    pmt.intern = lambda value: value  # type: ignore[attr-defined]
    pmt.cdr = lambda value: value  # type: ignore[attr-defined]
    pmt.u8vector_elements = lambda value: value  # type: ignore[attr-defined]
    gnuradio = ModuleType("gnuradio")
    gnuradio.gr = SimpleNamespace(  # type: ignore[attr-defined]
        basic_block=_BasicBlock,
        top_block=object,
    )
    satellites = ModuleType("satellites")
    satellites_core = ModuleType("satellites.core")
    satellites_core.gr_satellites_flowgraph = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pmt", pmt)
    monkeypatch.setitem(sys.modules, "gnuradio", gnuradio)
    monkeypatch.setitem(sys.modules, "satellites", satellites)
    monkeypatch.setitem(sys.modules, "satellites.core", satellites_core)
    sys.modules.pop("gnuradio_satellites", None)
    return importlib.import_module("gnuradio_satellites")


class _OneShotSink:
    def __init__(self, symbols: np.ndarray) -> None:
        self._symbols = symbols

    def drain(self) -> np.ndarray:
        symbols, self._symbols = self._symbols, np.empty(0, dtype=np.uint8)
        return symbols


def test_ax25_and_endurosat_decode_from_one_shared_symbol_chunk(monkeypatch) -> None:
    runtime = _load_runtime_module(monkeypatch)
    ax25_payload = ax25.encode_ui(dest="DSN", src="TEST", info=b"ax25")
    bits = np.concatenate(
        [
            framing.encode(ax25_payload, preamble_flags=16),
            np.zeros(64, dtype=np.uint8),
            endurosat_link.frame_bits(b"endurosat"),
        ]
    )
    fallback = runtime._FallbackDemod(
        "gfsk9600",
        _OneShotSink(bits),
        framing="AX.25",
        framings_list=("AX.25", "EnduroSat"),
    )

    results = fallback.drain_frames()

    assert {(result.canonical_framing, result.payload) for result in results} == {
        ("ax25", ax25_payload),
        ("endurosat", b"endurosat"),
    }
    assert fallback.race_framings == ("ax25", "endurosat")


def test_additive_legacy_decoder_state_is_independent_across_drains(monkeypatch) -> None:
    runtime = _load_runtime_module(monkeypatch)
    enduro_bits = endurosat_link.frame_bits(b"split")
    split = len(enduro_bits) // 2
    chunks = [enduro_bits[:split], enduro_bits[split:]]

    class _ChunkSink:
        def drain(self) -> np.ndarray:
            return chunks.pop(0) if chunks else np.empty(0, dtype=np.uint8)

    fallback = runtime._FallbackDemod(
        "gfsk9600",
        _ChunkSink(),
        framings_list=("AX.25", "EnduroSat"),
    )
    assert fallback.drain_frames() == []
    results = fallback.drain_frames()
    assert [(result.canonical_framing, result.payload) for result in results] == [
        ("endurosat", b"split")
    ]


def test_usp_and_endurosat_share_one_demod_with_soft_and_hard_consumers(monkeypatch) -> None:
    runtime = _load_runtime_module(monkeypatch)
    hard_sink = _OneShotSink(np.empty(0, dtype=np.uint8))
    soft_tap = object()
    build_calls: list[tuple] = []

    modem = ModuleType("modem")

    def build_demod(*args, **kwargs):
        build_calls.append((*args, kwargs))
        return hard_sink, soft_tap

    modem.build_demod = build_demod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modem", modem)

    class _SoftSymbolSink:
        pass

    gnuradio_gfsk = ModuleType("gnuradio_gfsk")
    gnuradio_gfsk.SoftSymbolSink = _SoftSymbolSink  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gnuradio_gfsk", gnuradio_gfsk)

    connections: list[tuple[object, object]] = []
    top_block = SimpleNamespace(connect=lambda source, sink: connections.append((source, sink)))
    fallbacks, returned_soft_tap = runtime._build_fallbacks(
        top_block,
        object(),
        48_000,
        modes=[("gmsk", 9_600)],
        framing="USP",
        framings_list=("USP", "EnduroSat"),
        mod_index=0.75,
        native_enabled=True,
    )

    assert len(build_calls) == 1
    assert build_calls[0][-1]["mod_index"] == 0.75
    assert returned_soft_tap is soft_tap
    assert len(fallbacks) == 2
    assert fallbacks[0]._legacy_framings == ("EnduroSat",)
    assert [label for label, _profile, _decoder in fallbacks[1]._native_decoders] == ["USP"]
    assert connections == [(soft_tap, fallbacks[1]._sink)]

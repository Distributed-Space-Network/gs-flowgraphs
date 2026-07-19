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

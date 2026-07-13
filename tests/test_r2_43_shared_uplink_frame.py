"""R2-43 — one payload/framing selector, and both TX engines obey it.

The defect: the two engines each built their own frame, and disagreed.

    dsp        payload from uplink_b64 | uplink_file | on-disk uplink.bin; framing ax25 OR endurosat
    gnuradio   payload from uplink_b64 ONLY; framing ALWAYS ax25 — `framing=endurosat` was ignored

The waveform schema ADVERTISES ``gnuradio`` + ``endurosat``. Flying that pair keyed the PA and
radiated a well-formed AX.25 UI frame at a satellite that speaks EnduroSat chip packets: the wrong
protocol, correctly modulated, reported as a success. Asking that engine for a file-sourced uplink
transmitted an EMPTY frame instead.

GNU Radio is not importable off-bench, so these drive the pure selector — which is the whole
point of extracting it: the thing that decides WHAT to transmit no longer needs a radio to test.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_APPS = Path(__file__).resolve().parents[1] / "apps"
sys.path.insert(0, str(_APPS))

from _uplink_frame import (  # noqa: E402
    ENGINES,
    FRAMINGS,
    build_uplink_frame,
    resolve_payload,
    select_framing,
    supported_pairs,
)

from gfsk_ax25 import endurosat, endurosat_link  # noqa: E402

PROFILE = endurosat.LinkProfile()
PAYLOAD = b"\xde\xad\xbe\xef airmac-ciphertext"


def _args(tmp_path: Path, rate: float = 96_000.0) -> SimpleNamespace:
    return SimpleNamespace(sample_rate=rate, output_dir=str(tmp_path))


class TestEveryPayloadSourceWorksOnEveryEngine:
    """The gnuradio engine knew about ONE of the three sources. The other two silently produced an
    empty frame — and an empty frame still keys the PA."""

    def test_uplink_b64(self, tmp_path: Path) -> None:
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode()}
        payload, source = resolve_payload(_args(tmp_path), params)
        assert payload == PAYLOAD
        assert source == "uplink_b64"

    def test_uplink_file(self, tmp_path: Path) -> None:
        f = tmp_path / "cmd.bin"
        f.write_bytes(PAYLOAD)
        payload, source = resolve_payload(_args(tmp_path), {"uplink_file": str(f)})
        assert payload == PAYLOAD
        assert "cmd.bin" in source

    def test_on_disk_uplink_bin(self, tmp_path: Path) -> None:
        (tmp_path / "uplink.bin").write_bytes(PAYLOAD)
        payload, source = resolve_payload(_args(tmp_path), {})
        assert payload == PAYLOAD
        assert "uplink.bin" in source

    def test_no_payload_is_reported_as_such(self, tmp_path: Path) -> None:
        """An empty uplink is not an error here, but it must be VISIBLE — the app logs a warning,
        because keying a PA to transmit nothing is a thing an operator should learn about."""
        payload, source = resolve_payload(_args(tmp_path), {})
        assert payload == b""
        assert source == "NONE"


class TestFramingIsHonouredRatherThanAssumed:
    def test_endurosat_framing_produces_the_chip_packet_not_ax25(self, tmp_path: Path) -> None:
        """THE R2-43 SCENARIO. The gnuradio engine built an AX.25 UI frame here, every time.

        The EnduroSat chip packet is [0xAA x5][0x7E][len][payload][CRC-16]: no HDLC, no NRZI, no
        G3RUH scrambling. Its bits must round-trip through the EnduroSat deframer, and the payload
        must come back VERBATIM — it is already-encrypted AirMAC and nothing may touch it."""
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "endurosat"}
        frame = build_uplink_frame(
            _args(tmp_path), params, PROFILE, framing_name=select_framing(params)
        )
        assert frame.framing == "endurosat"
        assert frame.payload_len == len(PAYLOAD)
        # The bits ARE an EnduroSat packet — proven by deframing them, not by inspecting a flag.
        assert endurosat_link.deframe(frame.bits) == [PAYLOAD]
        # ...and the modulation travels with the framing, rather than each engine guessing.
        assert frame.symbol_rate_hz == endurosat_link.DEFAULT_SYMBOL_RATE_HZ
        assert frame.mod_index == endurosat_link.DEFAULT_MOD_INDEX

    def test_ax25_framing_still_produces_ax25(self, tmp_path: Path) -> None:
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "ax25"}
        frame = build_uplink_frame(
            _args(tmp_path), params, PROFILE, framing_name=select_framing(params)
        )
        assert frame.framing == "ax25"
        assert frame.symbol_rate_hz == PROFILE.symbol_rate_hz
        # An AX.25 frame is NOT an EnduroSat packet — the deframer must find nothing in it.
        assert endurosat_link.deframe(frame.bits) == []

    def test_an_unknown_framing_falls_back_to_ax25_rather_than_inventing_one(self) -> None:
        assert select_framing({"framing": "mobitex"}) == "ax25"

    def test_the_endurosat_payload_is_not_scrambled_or_nrzi_encoded(self, tmp_path: Path) -> None:
        """A distinct assertion because it is the exact corruption AX.25 framing would apply: the
        AirMAC ciphertext must arrive bit-identical or the satellite cannot decrypt it."""
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "endurosat"}
        frame = build_uplink_frame(
            _args(tmp_path), params, PROFILE, framing_name=select_framing(params)
        )
        assert endurosat_link.deframe(frame.bits)[0] == PAYLOAD


class TestBothEnginesModulateTheSameBits:
    """The engines are now pure modulators over a frame they did not build. The GNU Radio one cannot
    be imported off-bench, so what is pinned here is the CONTRACT it consumes: an UplinkFrame that
    already carries the bits and the modulation, so there is nothing left for it to decide."""

    @pytest.mark.parametrize("framing_name", FRAMINGS)
    def test_the_frame_carries_everything_an_engine_needs(
        self, tmp_path: Path, framing_name: str
    ) -> None:
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": framing_name}
        frame = build_uplink_frame(_args(tmp_path), params, PROFILE, framing_name=framing_name)
        assert frame.bits.size > 0
        assert set(np.unique(frame.bits)) <= {0, 1}
        assert frame.sample_rate_hz > 0
        assert frame.symbol_rate_hz > 0
        assert frame.sps == frame.sample_rate_hz / frame.symbol_rate_hz

    def test_the_gnuradio_engine_no_longer_knows_what_a_frame_is(self) -> None:
        """Pinned so the split cannot silently grow back: the engine must not resolve payloads or
        choose framings. If these names return to its CODE, so does the bug.

        The docstring is stripped first — it deliberately names the old behaviour to explain why the
        function no longer has it, and an earlier version of this assertion failed on exactly that.
        Grep the code, not the prose about the code."""
        import ast

        src = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
        fn = next(
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "modulate_gnuradio"
        )
        body = fn.body[1:] if ast.get_docstring(fn) else fn.body
        code = "\n".join(ast.unparse(stmt) for stmt in body)
        assert "uplink_b64" not in code, "the engine is resolving payloads again"
        assert "encode_ui" not in code, "the engine is building AX.25 frames again"
        assert "b64decode" not in code
        # ...and it takes the shared frame as its ONLY input.
        assert [a.arg for a in fn.args.args] == ["frame"]

    def test_the_dsp_engine_routes_through_the_shared_selector_too(self) -> None:
        src = (_APPS / "cubesat_gfsk_ax25_tx.py").read_text(encoding="utf-8")
        assert "build_uplink_frame" in src
        assert "_build_frame(args, params, profile)" in src


class TestTheCapabilityTableIsHonest:
    def test_every_advertised_pair_is_implemented(self) -> None:
        """The schema advertised gnuradio+endurosat while the code silently sent AX.25. The table
        exists so config validation can reject a pair nobody built — and so this test can catch a
        pair that gets advertised without one."""
        pairs = supported_pairs()
        assert ("gnuradio", "endurosat") in pairs
        assert ("dsp", "endurosat") in pairs
        assert len(pairs) == len(ENGINES) * len(FRAMINGS)

    @pytest.mark.parametrize(("engine", "framing_name"), supported_pairs())
    def test_each_advertised_pair_actually_builds_a_frame(
        self, tmp_path: Path, engine: str, framing_name: str
    ) -> None:
        """Every tuple in the table is exercised end-to-end at the framing layer. Before R2-43 this
        test could not have been written: half the table did not go through this code at all."""
        params = {
            "uplink_b64": base64.b64encode(PAYLOAD).decode(),
            "framing": framing_name,
            "engine": engine,
        }
        frame = build_uplink_frame(_args(tmp_path), params, PROFILE, framing_name=framing_name)
        assert frame.framing == framing_name
        assert frame.payload_len == len(PAYLOAD)
        assert frame.bits.size > 0

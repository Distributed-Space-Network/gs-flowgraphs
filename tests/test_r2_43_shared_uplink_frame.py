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
    UnknownFraming,
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

    def test_an_unknown_EXPLICIT_framing_is_REFUSED_not_downgraded(self) -> None:
        """Round 5. This test used to assert the OPPOSITE — that an unknown framing quietly became
        AX.25 — and it passed, which is how the defect survived: a fallback for a framing the caller
        explicitly REQUESTED is a wrong-protocol transmission wearing a default's clothes.

        An ABSENT framing still defaults to AX.25 (the app's documented behaviour). A framing that
        was asked for and is not understood stops the pass."""
        with pytest.raises(UnknownFraming, match="refusing to fall back"):
            select_framing({"framing": "mobitex"})
        assert select_framing({}) == "ax25"  # absent -> documented default
        assert select_framing({"framing": "  "}) == "ax25"  # blank -> absent

    @pytest.mark.parametrize(
        ("label", "want"),
        [
            ("AirMAC", "endurosat"),          # the customer's own label for the session layer
            ("EnduroSat AirMAC", "endurosat"),
            ("endurosat_airmac", "endurosat"),
            ("EnduroSat", "endurosat"),
            ("AX.25", "ax25"),
            ("ax_25", "ax25"),
            ("  AX25  ", "ax25"),
        ],
    )
    def test_the_labels_the_backend_actually_emits_are_understood(
        self, label: str, want: str
    ) -> None:
        """Selection matched the exact strings "ax25"/"endurosat" and silently fell back for
        everything else. A pass whose framing said "AirMAC" — which is what the customer calls the
        EnduroSat session layer, and what the catalogues carry — therefore transmitted AX.25 at an
        EnduroSat bird and reported success. AirMAC rides INSIDE the chip packet; it is not a
        different physical framing."""
        assert select_framing({"framing": label}) == want

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


class TestTheGnuRadioEngineDoesNotPadTheFrame:
    """Round 7: the engine no longer PACKS at all, so there is nothing to pad.

    np.packbits() pads the last byte with zero bits, so the modulator radiated up to seven
    invented symbols the FCS did not cover. Round 6 padded the bitstream with flag bits at the
    framing layer — but AFTER scrambling and NRZI, so it was not encoded idle fill at all, just
    raw bits wearing a flag's clothes. The answer is not to pack: one byte per bit,
    do_unpack=False, exact bit count, no padding anywhere."""

    def test_the_engine_feeds_UNPACKED_bits(self) -> None:
        import ast

        src = (_APPS / "gnuradio_gfsk.py").read_text(encoding="utf-8")
        fn = next(
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "modulate_gnuradio"
        )
        code = "\n".join(ast.unparse(st) for st in fn.body)
        assert "packbits" not in code, "the engine still PACKS bits (and pads the last byte)"
        assert "do_unpack=False" in code, "gfsk_mod would unpack bytes we never packed"

    def test_the_framing_layer_no_longer_pads(self) -> None:
        src = (_APPS / "_uplink_frame.py").read_text(encoding="utf-8")
        assert "_byte_align" not in src


class TestNothingIsSilentlyTruncatedOrDowngraded:
    def test_an_oversized_endurosat_payload_is_REFUSED(self, tmp_path: Path) -> None:
        """It was TRUNCATED and radiated as a successful command. A truncated command is a
        DIFFERENT command: the spacecraft executes whatever the first 128 bytes happen to mean."""
        from _uplink_frame import PayloadRejected

        big = base64.b64encode(b"A" * 300).decode()
        with pytest.raises(PayloadRejected, match="REFUSING"):
            build_uplink_frame(
                _args(tmp_path),
                {"uplink_b64": big, "framing": "endurosat"},
                PROFILE,
                framing_name="endurosat",
            )

    def test_an_oversized_ax25_payload_is_REFUSED(self, tmp_path: Path) -> None:
        from _uplink_frame import PayloadRejected

        big = base64.b64encode(b"A" * 300).decode()
        with pytest.raises(PayloadRejected, match="REFUSING"):
            build_uplink_frame(
                _args(tmp_path),
                {"uplink_b64": big, "framing": "ax25"},
                PROFILE,
                framing_name="ax25",
            )

    def test_an_EMPTY_endurosat_payload_is_REFUSED(self, tmp_path: Path) -> None:
        """An empty chip packet does not deframe at the far end. It is a PA key with no effect."""
        from _uplink_frame import PayloadRejected

        with pytest.raises(PayloadRejected, match="EMPTY"):
            build_uplink_frame(
                _args(tmp_path), {"framing": "endurosat"}, PROFILE, framing_name="endurosat"
            )

    def test_params_win_over_the_environment(self) -> None:
        """GS_FLOWGRAPH_FRAMING silently overrode the framing gs-client had VALIDATED and written
        into params.json — turning a configured EnduroSat uplink back into AX.25."""
        assert select_framing({"framing": "endurosat"}, env="ax25") == "endurosat"
        assert select_framing({}, env="endurosat") == "endurosat"


class TestPreflightRunsBeforeThePaIsKeyed:
    def test_a_rate_that_cannot_modulate_is_caught_BEFORE_ready(self, tmp_path: Path) -> None:
        """THE COUNTEREXAMPLE. The canonical AX.25 TX waveform shipped at 96 kHz against
        12480 sym/s = 7.692 samples/symbol. Both engines require an integer sps, so it could NEVER
        have transmitted — and it found that out inside the modulator, after `ready`, with the T/R
        relay thrown and the PA keyed."""
        from types import SimpleNamespace

        from _uplink_frame import RateUnusable, preflight

        args = SimpleNamespace(sample_rate=96_000.0, output_dir=str(tmp_path))
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "ax25"}
        with pytest.raises(RateUnusable, match="integer"):
            preflight(args, params, PROFILE, engine="dsp", framing_name="ax25")

    def test_124800_works_for_BOTH_advertised_framings(self, tmp_path: Path) -> None:
        """124800/12480 = 10 (ax25); 124800/9600 = 13 (endurosat). One rate, both framings."""
        from types import SimpleNamespace

        from _uplink_frame import preflight

        args = SimpleNamespace(sample_rate=124_800.0, output_dir=str(tmp_path))
        for framing_name in ("ax25", "endurosat"):
            params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": framing_name}
            frame = preflight(args, params, PROFILE, engine="dsp", framing_name=framing_name)
            assert float(frame.sps).is_integer()

    def test_an_unknown_engine_is_REFUSED_not_downgraded_to_dsp(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from _uplink_frame import UnknownEngine, preflight

        args = SimpleNamespace(sample_rate=124_800.0, output_dir=str(tmp_path))
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "ax25"}
        with pytest.raises(UnknownEngine, match="refusing to fall back"):
            preflight(args, params, PROFILE, engine="gnuradio3", framing_name="ax25")


class TestRound8ModulationParametersAreFiniteAndBounded:
    """bt=0 raised ZeroDivisionError inside the Gaussian filter; bt/mod_index of NaN or Inf sailed
    through and produced non-finite IQ the SDR would happily accept. All of it AFTER `ready`, with
    the T/R relay thrown and the PA keyed."""

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("bt", 0.0),
            ("bt", float("nan")),
            ("bt", float("inf")),
            ("mod_index", float("nan")),
            ("mod_index", float("inf")),
            ("mod_index", 0.0),
        ],
    )
    def test_a_number_that_cannot_be_modulated_stops_the_pass_on_the_ground(
        self, tmp_path: Path, key: str, bad: float
    ) -> None:
        from types import SimpleNamespace

        from _uplink_frame import ModulationUnusable, preflight

        args = SimpleNamespace(sample_rate=124_800.0, output_dir=str(tmp_path))
        params = {
            "uplink_b64": base64.b64encode(PAYLOAD).decode(),
            "framing": "endurosat",
            key: bad,
        }
        with pytest.raises(ModulationUnusable):
            preflight(args, params, PROFILE, engine="dsp", framing_name="endurosat")

    def test_the_healthy_defaults_still_pass(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from _uplink_frame import preflight

        args = SimpleNamespace(sample_rate=124_800.0, output_dir=str(tmp_path))
        params = {"uplink_b64": base64.b64encode(PAYLOAD).decode(), "framing": "endurosat"}
        assert preflight(args, params, PROFILE, engine="dsp", framing_name="endurosat")


class TestRound8PreflightBuildsTheRealIq:
    """The engine was imported only INSIDE the burst, after `ready`. On a host without GNU Radio:

        ready(engine=gnuradio) -> started -> tx-failed -> ModuleNotFoundError

    with the relay thrown and the PA keyed for a modulator that does not exist. And the preflighted
    frame was DISCARDED and rebuilt after keying, so a file payload could change in between."""

    def test_preflight_imports_the_engine_and_modulates(self) -> None:
        import ast

        src = (_APPS / "cubesat_gfsk_ax25_tx.py").read_text(encoding="utf-8")
        fn = next(
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "_preflight_and_build_iq"
        )
        code = "\n".join(ast.unparse(st) for st in fn.body)
        assert "modulate_gnuradio" in code, "preflight does not load the gnuradio engine"
        assert "gfsk.modulate" in code, "preflight does not modulate on the dsp path"
        assert "isfinite" in code, "preflight does not check the IQ is finite"

    def test_the_burst_uses_the_prevalidated_iq(self) -> None:
        src = (_APPS / "cubesat_gfsk_ax25_tx.py").read_text(encoding="utf-8")
        assert "iq=prevalidated_iq" in src, "the burst rebuilds the frame it was supposed to reuse"
        assert "tx-engine-unavailable" in src, "a missing engine still fails AFTER ready"

    def test_a_missing_engine_fails_the_SPAWN_not_the_burst(self) -> None:
        """The error code exists and is emitted before `ready` — the spawn returns non-zero."""
        src = (_APPS / "cubesat_gfsk_ax25_tx.py").read_text(encoding="utf-8")
        before_ready = src[: src.index('"event": "ready"')]
        assert "tx-preflight-failed" in before_ready
        assert "tx-engine-unavailable" in before_ready

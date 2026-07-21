from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import flowgraph_runtime_check as runtime_check
import pytest


def _module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"/opt/gs-flowgraphs/bin/{name.replace('.', '/')}.py"
    if name == "satellites.components.demodulators":
        module.fsk_demodulator = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    if name == "_fallback_select":
        module.LIVE_DECODE_DRAIN_PERIOD_S = 0.05  # type: ignore[attr-defined]
        module.should_build_demod = (  # type: ignore[attr-defined]
            lambda *, mode, local_deframer_enabled, grsat_live: mode is not None
            and (local_deframer_enabled or grsat_live)
        )
        module.should_collect_hard_symbols = (  # type: ignore[attr-defined]
            lambda *, legacy_hard_enabled, native_hard_enabled: legacy_hard_enabled
            or native_hard_enabled
        )
    return module


def test_runtime_check_discovers_dependencies_from_installed_tree(tmp_path: Path) -> None:
    (tmp_path / "receiver.py").write_text(
        "import pmt\nfrom scipy import signal\nfrom native_framing import registry\n",
        encoding="utf-8",
    )
    (tmp_path / "native_framing").mkdir()
    (tmp_path / "native_framing" / "registry.py").write_text(
        "import construct\n",
        encoding="utf-8",
    )

    assert runtime_check.discover_external_imports(tmp_path) == (
        "construct",
        "pmt",
        "scipy",
    )


def test_runtime_check_reports_missing_dependency_without_hardware() -> None:
    def importer(name: str) -> ModuleType:
        if name == "pmt":
            raise ModuleNotFoundError("No module named 'pmt'")
        module = _module(name)
        if name == "gnuradio_satellites":
            module.make_grsat_deframers = lambda _label: [object()]  # type: ignore[attr-defined]
        return module

    result = runtime_check.check_runtime(
        importer=importer,
        required_modules=("scipy", "pmt"),
        deframer_labels=("USP",),
    )

    assert result["ok"] is False
    assert result["checks"][1] == {
        "check": "import:pmt",
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'pmt'",
    }


def test_runtime_check_constructs_priority_deframers() -> None:
    built: list[str] = []

    def importer(name: str) -> ModuleType:
        module = _module(name)
        if name == "gnuradio_satellites":
            def build(label: str) -> list[object]:
                built.append(label)
                return [object()]

            module.make_grsat_deframers = build  # type: ignore[attr-defined]
        return module

    labels = ("AX100 Mode 5", "AX100 Mode 6", "AX100 ASM+Golay", "USP")
    result = runtime_check.check_runtime(
        importer=importer,
        required_modules=("scipy", "pmt"),
        deframer_labels=labels,
    )

    assert result["ok"] is True
    assert built == list(labels)
    assert [check["count"] for check in result["checks"] if "count" in check] == [1] * 4
    assert {
        check["check"]: check["ok"]
        for check in result["checks"]
        if str(check["check"]).startswith("safety:")
    } == {
        "safety:recorder-only-no-demod": True,
        "safety:decode-drain-period": True,
        "safety:soft-only-no-hard-queue": True,
    }
    assert next(
        check for check in result["checks"] if check["check"] == "demodulator:GMSK@2400"
    )["ok"] is True


def test_runtime_check_fails_when_pinned_fsk_component_cannot_construct() -> None:
    def importer(name: str) -> ModuleType:
        module = _module(name)
        if name == "satellites.components.demodulators":
            def fail(*args, **kwargs):
                raise TypeError("unexpected constructor drift")

            module.fsk_demodulator = fail  # type: ignore[attr-defined]
        if name == "gnuradio_satellites":
            module.make_grsat_deframers = lambda _label: [object()]  # type: ignore[attr-defined]
        return module

    result = runtime_check.check_runtime(
        importer=importer,
        required_modules=("scipy",),
        deframer_labels=("USP",),
    )

    assert result["ok"] is False
    failed = next(
        check for check in result["checks"] if check["check"] == "demodulator:GMSK@2400"
    )
    assert failed["error"] == "TypeError: unexpected constructor drift"


def test_supported_framings_add_grsatellites_only_for_exact_live_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def importer(name: str):
        imported.append(name)
        if name == "framings":
            return SimpleNamespace(advertised_local_framings=lambda: (
                "AX.25", "EnduroSat", "ccsds_tm", "KISS"
            ))
        if name == "native_framing.registry":
            return SimpleNamespace(advertised_profiles=lambda: {
                "AX.25": SimpleNamespace(decoder_factory=object()),
                "USP": SimpleNamespace(decoder_factory=object()),
                "Planned": SimpleNamespace(decoder_factory=None),
            })
        if name == "satellites.core.gr_satellites_flowgraph":
            return SimpleNamespace(gr_satellites_flowgraph=SimpleNamespace(
                _deframer_hooks={"Light-1": object(), "USP": object()}
            ))
        raise AssertionError(name)

    monkeypatch.setattr(runtime_check.importlib, "import_module", importer)
    expected_local = ["AX.25", "ccsds_tm", "EnduroSat", "KISS", "USP"]
    assert runtime_check.supported_framings(environment={}) == expected_local
    assert runtime_check.supported_framings(environment={"GS_GRSAT_LIVE": "true"}) == [
        "AX.25",
        "ccsds_tm",
        "EnduroSat",
        "KISS",
        "USP",
    ]
    assert imported.count("satellites.core.gr_satellites_flowgraph") == 0
    assert runtime_check.supported_framings(environment={"GS_GRSAT_LIVE": "1"}) == [
        "AX.25",
        "ccsds_tm",
        "EnduroSat",
        "KISS",
        "Light-1",
        "USP",
    ]
    assert imported.count("satellites.core.gr_satellites_flowgraph") == 1


def test_runtime_check_classifies_missing_pip_dependency_without_installing(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["PyYAML>=6.0"]

[tool.gs-client.flowgraph-runtime]
os-owned-modules = ["pmt"]
optional-modules = ["dvbs2rx"]
module-distributions = { yaml = "PyYAML" }
""".strip(),
        encoding="utf-8",
    )

    def importer(name: str) -> ModuleType:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        module = _module(name)
        if name == "gnuradio_satellites":
            module.make_grsat_deframers = lambda _label: [object()]  # type: ignore[attr-defined]
        return module

    result = runtime_check.check_runtime(
        importer=importer,
        required_modules=("yaml",),
        deframer_labels=("USP",),
        client_pyproject=pyproject,
    )

    missing = result["checks"][0]
    assert missing["owner"] == "gs-client-pip"
    assert missing["requirement"] == "PyYAML>=6.0"
    assert result["suggested_actions"] == [
        f"Run: sudo {runtime_check.shlex.quote(runtime_check.sys.executable)} "
        f"-m pip install -c {runtime_check.shlex.quote(str(tmp_path / 'constraints.txt'))} "
        "'PyYAML>=6.0'"
    ]

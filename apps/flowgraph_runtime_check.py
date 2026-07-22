#!/usr/bin/env python3
"""No-hardware dependency and decoder-construction check for station flowgraphs."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import tomllib
from _fallback_select import grsat_live_enabled

OPTIONAL_MODULES = frozenset({"dvbs2rx"})

BENCH_DEFRAMERS = (
    "AX.25",
    "AX100 Mode 5",
    "AX100 Mode 6",
    "AX100 ASM+Golay",
    "USP",
)


def supported_framings(*, environment: dict[str, str] | None = None) -> list[str]:
    """Return local deframers, native profiles, and live-enabled gr-sat constructors."""
    local_registry = importlib.import_module("framings")
    registry = importlib.import_module("native_framing.registry")
    advertised = registry.advertised_profiles()
    labels = list(local_registry.advertised_local_framings())
    labels.extend(
        str(label)
        for label, profile in advertised.items()
        if getattr(profile, "decoder_factory", None) is not None
    )
    if grsat_live_enabled(environment):
        flowgraph = importlib.import_module(
            "satellites.core.gr_satellites_flowgraph"
        ).gr_satellites_flowgraph
        hooks = getattr(flowgraph, "_deframer_hooks", None)
        if not isinstance(hooks, dict) or not hooks:
            raise RuntimeError("gr-satellites exposes no deframer constructor map")
        labels.extend(hooks)
    unique: dict[str, str] = {}
    for raw in labels:
        label = raw.strip()
        if label:
            unique.setdefault(label.casefold(), label)
    return sorted(unique.values(), key=str.casefold)


def discover_external_imports(
    apps_root: Path,
    *,
    optional_modules: frozenset[str] | set[str] = OPTIONAL_MODULES,
) -> tuple[str, ...]:
    """Return imports used by the installed app tree, excluding local/stdlib code."""
    local_modules = {
        path.stem for path in apps_root.glob("*.py")
    } | {
        path.name for path in apps_root.iterdir() if path.is_dir()
    }
    imported: set[str] = set()
    for path in apps_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)

    return tuple(
        sorted(
            name
            for name in imported
            if name.partition(".")[0]
            not in sys.stdlib_module_names | local_modules | optional_modules
        )
    )


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _load_contract(pyproject: Path | None) -> dict[str, object]:
    if pyproject is None:
        return {}
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    runtime = metadata["tool"]["gs-client"]["flowgraph-runtime"]
    requirements = {
        _canonical(re.split(r"[><=!~\[ ;]", item.strip())[0]): item
        for item in metadata["project"]["dependencies"]
    }
    return {
        "pyproject": str(pyproject.resolve()),
        "os_owned": set(runtime["os-owned-modules"]),
        "optional": set(runtime["optional-modules"]),
        "module_distributions": dict(runtime["module-distributions"]),
        "requirements": requirements,
    }


def _system_import(name: str, system_python: Path) -> tuple[bool, str]:
    try:
        completed = subprocess.run(  # noqa: S603
            [
                str(system_python),
                "-c",
                f"import importlib; importlib.import_module({name!r})",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (completed.stderr or completed.stdout).strip()
    return completed.returncode == 0, detail


def _annotate_import_checks(
    checks: list[dict[str, object]],
    contract: dict[str, object],
    *,
    system_python: Path,
) -> list[str]:
    actions: set[str] = set()
    if not contract:
        return []
    os_owned = set(contract.get("os_owned", set()))
    optional = set(contract.get("optional", set()))
    mappings = dict(contract.get("module_distributions", {}))
    requirements = dict(contract.get("requirements", {}))
    pyproject = Path(str(contract["pyproject"]))
    constraints = pyproject.with_name("constraints.txt")

    for check in checks:
        label = str(check["check"])
        if not label.startswith("import:"):
            continue
        module = label.removeprefix("import:")
        root = module.partition(".")[0]
        if root in os_owned:
            check["owner"] = "station-os"
            if not check["ok"]:
                system_ok, detail = _system_import(module, system_python)
                check["system_python"] = str(system_python)
                check["system_import_ok"] = system_ok
                if detail:
                    check["system_import_detail"] = detail
                if system_ok:
                    actions.add(
                        "Run: sudo "
                        f"{shlex.quote(str(system_python))} -m venv --upgrade "
                        "--system-site-packages /opt/gs-client/venv"
                    )
                else:
                    actions.add(
                        f"Restore the pre-existing station package that supplies {root} for "
                        f"{system_python}; do not install an unrelated PyPI package with that name."
                    )
        elif root in optional:
            check["owner"] = "optional"
        else:
            distribution = str(mappings.get(root, root))
            requirement = requirements.get(_canonical(distribution))
            check["owner"] = "gs-client-pip" if requirement else "untracked"
            check["requirement"] = requirement or distribution
            if not check["ok"]:
                if requirement:
                    actions.add(
                        "Run: sudo "
                        f"{shlex.quote(sys.executable)} -m pip install -c "
                        f"{shlex.quote(str(constraints))} {shlex.quote(str(requirement))}"
                    )
                else:
                    actions.add(
                        f"Declare the distribution that supplies {root} in "
                        "gs-client/pyproject.toml "
                        "before installing anything."
                    )
    return sorted(actions)


def _check_import(name: str, importer: Callable[[str], ModuleType]) -> dict[str, object]:
    try:
        module = importer(name)
    except Exception as exc:
        return {
            "check": f"import:{name}",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "check": f"import:{name}",
        "ok": True,
        "file": str(getattr(module, "__file__", "")),
        "version": str(getattr(module, "__version__", "")),
    }


def _check_recorder_safety(importer: Callable[[str], ModuleType]) -> list[dict[str, object]]:
    """Verify the installed checkout contains the recorder-overflow fix."""

    try:
        safety = importer("_fallback_select")
        selector = safety.should_build_demod
        period_s = float(safety.LIVE_DECODE_DRAIN_PERIOD_S)
        symbol_queue_items = int(safety.LIVE_SYMBOL_QUEUE_CAPACITY_ITEMS)
        symbol_queue_symbols = int(safety.LIVE_SYMBOL_QUEUE_CAPACITY_SYMBOLS)
        recorder_only_safe = not selector(
            mode=("gmsk", 2400.0),
            local_deframer_enabled=False,
            grsat_live=False,
        )
        active_decode_built = selector(
            mode=("gmsk", 2400.0),
            local_deframer_enabled=True,
            grsat_live=False,
        )
        collect_hard = safety.should_collect_hard_symbols
        soft_only_safe = not collect_hard(
            legacy_hard_enabled=False,
            native_hard_enabled=False,
        )
        # cmd_176_176 produced about ten symbols per GNU Radio work item.
        # Scheduler fragmentation must not trip the item bound while the actual
        # symbol/resource occupancy remains tiny.
        scheduler_fragmentation_safe = (
            symbol_queue_items >= 4096
            and symbol_queue_symbols >= 1 << 20
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return [
            {"check": "safety:recorder-only-no-demod", "ok": False, "error": error},
            {"check": "safety:decode-drain-period", "ok": False, "error": error},
            {"check": "safety:soft-only-no-hard-queue", "ok": False, "error": error},
            {"check": "safety:symbol-queue-fragmentation", "ok": False, "error": error},
        ]
    return [
        {
            "check": "safety:recorder-only-no-demod",
            "ok": bool(recorder_only_safe and active_decode_built),
        },
        {
            "check": "safety:decode-drain-period",
            "ok": bool(0.0 < period_s <= 0.05),
            "period_s": period_s,
        },
        {
            "check": "safety:soft-only-no-hard-queue",
            "ok": bool(soft_only_safe),
        },
        {
            "check": "safety:symbol-queue-fragmentation",
            "ok": bool(scheduler_fragmentation_safe),
            "capacity_items": symbol_queue_items,
            "capacity_symbols": symbol_queue_symbols,
        },
    ]


def check_runtime(
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    required_modules: Sequence[str] | None = None,
    deframer_labels: Sequence[str] = BENCH_DEFRAMERS,
    apps_root: Path | None = None,
    client_pyproject: Path | None = None,
    system_python: Path = Path("/usr/bin/python3"),
) -> dict[str, object]:
    """Check imports and construct priority deframers without opening an SDR."""
    root = apps_root or Path(__file__).resolve().parent
    contract = _load_contract(client_pyproject)
    if required_modules is None:
        required_modules = discover_external_imports(
            root,
            optional_modules=set(contract.get("optional", OPTIONAL_MODULES)),
        )
    checks = [_check_import(name, importer) for name in required_modules]
    actions = _annotate_import_checks(checks, contract, system_python=system_python)
    checks.extend(_check_recorder_safety(importer))
    try:
        demodulators = importer("satellites.components.demodulators")
        builder = demodulators.fsk_demodulator
        builder(
            2_400.0,
            48_000.0,
            True,
            deviation=600.0,
            dc_block=True,
            options=None,
        )
    except Exception as exc:
        checks.append(
            {
                "check": "demodulator:GMSK@2400",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        checks.append({"check": "demodulator:GMSK@2400", "ok": True})
    try:
        modem = importer("modem")
        gr_blocks = importer("gnuradio.blocks")
        gr_runtime = importer("gnuradio.gr")
        top_block = gr_runtime.top_block("runtime_check_fsk_1200")
        source = gr_blocks.vector_source_c([0j], False)
        _hard, soft = modem.build_demod(
            "fsk",
            top_block,
            source,
            48_000.0,
            1_200.0,
            collect_hard=False,
        )
        if soft is None:
            raise RuntimeError("adaptive FSK frontend returned no soft-symbol tap")
    except Exception as exc:
        checks.append(
            {
                "check": "demodulator:FSK@1200-adaptive",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        checks.append({"check": "demodulator:FSK@1200-adaptive", "ok": True})
    engine: ModuleType | None = None
    try:
        engine = importer("gnuradio_satellites")
    except Exception as exc:
        checks.append(
            {
                "check": "engine:gnuradio_satellites",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        checks.append(
            {
                "check": "engine:gnuradio_satellites",
                "ok": True,
                "file": str(getattr(engine, "__file__", "")),
            }
        )

    if engine is not None:
        builder = getattr(engine, "make_grsat_deframers", None)
        if not callable(builder):
            checks.append(
                {
                    "check": "engine:deframer-builder",
                    "ok": False,
                    "error": "make_grsat_deframers is unavailable",
                }
            )
        else:
            for label in deframer_labels:
                try:
                    deframers = builder(label)
                    count = len(deframers)
                    if count < 1:
                        raise RuntimeError("no deframer constructed")
                except Exception as exc:
                    checks.append(
                        {
                            "check": f"deframer:{label}",
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    checks.append(
                        {"check": f"deframer:{label}", "ok": True, "count": count}
                    )

    return {
        "ok": all(bool(check["ok"]) for check in checks),
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "script": str(Path(__file__).resolve()),
        "apps_root": str(root.resolve()),
        "dependency_contract": str(contract.get("pyproject", "unavailable")),
        "suggested_actions": actions,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="emit one-line JSON")
    parser.add_argument(
        "--supported-framings",
        action="store_true",
        help="emit the runtime-supported framing labels and exit",
    )
    parser.add_argument(
        "--apps-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="installed or checkout gs-flowgraphs app tree to scan",
    )
    parser.add_argument(
        "--client-pyproject",
        type=Path,
        help="gs-client pyproject used to classify dependencies and suggest repairs",
    )
    parser.add_argument(
        "--system-python",
        type=Path,
        default=Path("/usr/bin/python3"),
        help="existing station interpreter used only for read-only OS import probes",
    )
    args = parser.parse_args(argv)
    if args.supported_framings:
        print(json.dumps(supported_framings(), separators=(",", ":"), sort_keys=True))
        return 0
    result = check_runtime(
        apps_root=args.apps_root,
        client_pyproject=args.client_pyproject,
        system_python=args.system_python,
    )
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCH_DEFRAMERS",
    "OPTIONAL_MODULES",
    "check_runtime",
    "discover_external_imports",
    "main",
    "supported_framings",
]

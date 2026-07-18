"""D-02 (audit round 2): the CMake install() list is EXPLICIT — a new file in
apps/ that is not added to CMakeLists.txt silently doesn't deploy, and the
installed tree then crashes on import at the first pass (this exact class of
bug shipped three times: _rateplan.py, _stream.py, _soapy_tx.py were imported
by installed apps but absent from the install list).

These tests pin the manifest to the tree so the drift is caught at CI time,
not on the bench:
  1. every *.py in apps/ must be named in CMakeLists.txt (or explicitly
     listed here as intentionally-not-installed),
  2. every file named in CMakeLists.txt must exist in apps/,
  3. every local module imported by an INSTALLED app must itself be installed.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
CMAKELISTS = ROOT / "CMakeLists.txt"

# Files in apps/ that are deliberately NOT installed. Keep this list empty
# unless a file is truly dev-only; document why when adding one.
INTENTIONALLY_NOT_INSTALLED: frozenset[str] = frozenset()


def _installed_entries() -> set[str]:
    """apps/<name>.py entries named in any install() stanza (comments stripped)."""
    text = CMAKELISTS.read_text(encoding="utf-8")
    entries: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        entries.update(re.findall(r"apps/([\w.]+\.py)", line))
    return entries


def _installed_packages() -> set[str]:
    """apps/<package> directories named in install(DIRECTORY ...) stanzas."""
    text = CMAKELISTS.read_text(encoding="utf-8")
    entries: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        entries.update(re.findall(r"DIRECTORY\s+apps/([\w.]+)", line))
    return entries


def _apps_on_disk() -> set[str]:
    return {p.name for p in APPS.glob("*.py")}


def _packages_on_disk() -> set[str]:
    return {
        path.name
        for path in APPS.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }


def _local_imports(path: Path, local_names: set[str]) -> set[str]:
    """Module names imported by `path` that are local to apps/ (top-level only;
    the spawn contract puts the app's own directory on sys.path)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in local_names:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            if root in local_names:
                found.add(root)
    return found


def test_every_app_file_is_in_the_install_list() -> None:
    on_disk = _apps_on_disk() - INTENTIONALLY_NOT_INSTALLED
    installed = _installed_entries()
    missing = sorted(on_disk - installed)
    assert missing == [], (
        f"apps/ files absent from CMakeLists.txt install(): {missing} — "
        f"they will silently NOT deploy to /opt/gs-flowgraphs/bin and the "
        f"installed tree will crash on import (add them to the install list "
        f"or to INTENTIONALLY_NOT_INSTALLED with a reason)"
    )


def test_every_install_entry_exists_on_disk() -> None:
    on_disk = _apps_on_disk()
    installed = _installed_entries()
    stale = sorted(installed - on_disk)
    assert stale == [], f"CMakeLists.txt installs files that do not exist: {stale}"


def test_every_app_package_is_in_the_install_list() -> None:
    missing = sorted(_packages_on_disk() - _installed_packages())
    assert missing == [], (
        f"apps/ packages absent from CMakeLists.txt install(DIRECTORY ...): {missing} — "
        "their tests can pass from the source tree while the station install omits them"
    )


def test_installed_apps_only_import_installed_modules() -> None:
    """The closure check: an installed app importing a non-installed local
    module is exactly the deployed-import-crash the audit found."""
    installed = _installed_entries()
    # Local importable module names: every apps/*.py plus package directories.
    local_names = {p.stem for p in APPS.glob("*.py")}
    local_names.update(
        p.name for p in APPS.iterdir() if p.is_dir() and (p / "__init__.py").exists()
    )
    installed_names = {e.removesuffix(".py") for e in installed}
    installed_names.update(_installed_packages())
    problems: list[str] = []
    for entry in sorted(installed):
        path = APPS / entry
        if not path.exists():
            continue  # covered by test_every_install_entry_exists_on_disk
        for mod in sorted(_local_imports(path, local_names)):
            if mod not in installed_names:
                problems.append(f"{entry} imports {mod} which is not installed")
    assert problems == [], "\n".join(problems)


def test_native_packages_import_from_a_staged_install_only(tmp_path: Path) -> None:
    """Copy exactly the declared CMake payload and import it outside the source tree."""

    staged = tmp_path / "bin"
    staged.mkdir()
    for entry in _installed_entries():
        shutil.copy2(APPS / entry, staged / entry)
    for package in _installed_packages():
        shutil.copytree(
            APPS / package,
            staged / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    package_names = ("native_framing", "native_lora", "native_telemetry")
    code = (
        "import importlib, pathlib, sys; "
        f"root = pathlib.Path({str(staged)!r}).resolve(); "
        "sys.path.insert(0, str(root)); "
        f"names = {package_names!r}; "
        "modules = [importlib.import_module(name) for name in names]; "
        "assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root) "
        "for module in modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

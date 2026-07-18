"""NF-BACKEND-001 fail-closed GNU Radio capability contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from native_framing.backends.gnuradio_core import (
    FEATURE_PATHS,
    MIN_GNURADIO_VERSION,
    GnuradioCoreUnavailable,
    probe_gnuradio_core,
    require_gnuradio_core,
)


def _callable() -> None:
    return None


def _modules(version: str = "3.11.0.0") -> dict[str, object]:
    digital = SimpleNamespace(
        correlate_access_code_tag_bb=_callable,
        correlate_access_code_ff_ts=_callable,
        diff_decoder_bb=_callable,
        additive_scrambler_bb=_callable,
        symbol_sync_ff=_callable,
    )
    fec = SimpleNamespace(code=SimpleNamespace(cc_decoder=SimpleNamespace(make=_callable)))
    return {
        "gnuradio": SimpleNamespace(__version__=version),
        "gnuradio.gr": SimpleNamespace(version=lambda: version),
        "gnuradio.digital": digital,
        "gnuradio.fec": fec,
    }


def _importer(modules: dict[str, object]):
    def load(name: str):
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    return load


def test_complete_pinned_api_contract_is_available_without_building_blocks():
    capabilities = probe_gnuradio_core(_importer(_modules()))
    assert capabilities.available is True
    assert capabilities.version_tuple == (3, 11, 0)
    assert capabilities.missing_features == ()
    assert capabilities.feature_map == {name: True for name, _, _ in FEATURE_PATHS}
    require_gnuradio_core(capabilities)


def test_missing_install_version_and_partial_api_fail_deterministically():
    missing = probe_gnuradio_core(_importer({}))
    assert missing.installed is False
    assert missing.available is False
    assert missing.reason == "GNU Radio import unavailable: ModuleNotFoundError"
    with pytest.raises(GnuradioCoreUnavailable, match="import unavailable"):
        require_gnuradio_core(missing)

    old = probe_gnuradio_core(_importer(_modules("3.9.7.0")))
    assert old.version_supported is False
    assert old.reason == "GNU Radio 3.9.7.0 is older than required 3.10.0"
    with pytest.raises(GnuradioCoreUnavailable, match="older than required"):
        require_gnuradio_core(old)

    modules = _modules()
    delattr(modules["gnuradio.digital"], "symbol_sync_ff")
    partial = probe_gnuradio_core(_importer(modules))
    assert partial.missing_features == ("fractional_symbol_timing",)
    with pytest.raises(GnuradioCoreUnavailable, match="fractional_symbol_timing"):
        require_gnuradio_core(partial)
    require_gnuradio_core(partial, required=("hard_correlation",))


def test_version_and_requested_capability_validation_are_fail_closed():
    modules = _modules("not-a-version")
    capabilities = probe_gnuradio_core(_importer(modules))
    assert capabilities.version_tuple is None
    assert capabilities.reason == "GNU Radio version is missing or unparsable"
    with pytest.raises(GnuradioCoreUnavailable, match="unparsable"):
        require_gnuradio_core(capabilities)

    valid = probe_gnuradio_core(_importer(_modules()))
    with pytest.raises(ValueError, match="unknown GNU Radio capabilities"):
        require_gnuradio_core(valid, required=("scheduler_magic",))
    assert MIN_GNURADIO_VERSION == (3, 10, 0)


def test_real_environment_probe_never_imports_at_module_import_time():
    capabilities = probe_gnuradio_core()
    assert isinstance(capabilities.installed, bool)
    assert capabilities.reason
    if not capabilities.available:
        with pytest.raises(GnuradioCoreUnavailable):
            require_gnuradio_core(capabilities)

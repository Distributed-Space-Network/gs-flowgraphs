#!/usr/bin/env python3
"""Print the Python SoapySDR call forms for the TX methods used by tx_gfsk.py.

Run this on the target host with the same interpreter that imports SoapySDR, e.g.:

  /usr/bin/python3 tools/inspect_soapy_calls.py
  /usr/bin/python3 tools/inspect_soapy_calls.py --device "driver=xtrx"

By default this only introspects the binding; ``--device`` also opens the SDR and reports a few
read-only capabilities. It does not setup or activate TX streams.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from collections.abc import Callable

METHODS = (
    "setupStream",
    "activateStream",
    "deactivateStream",
    "closeStream",
    "writeStream",
    "readStreamStatus",
    "getStreamMTU",
    "setSampleRate",
    "setFrequency",
    "setBandwidth",
    "setGain",
    "getGain",
    "getGainRange",
    "listGains",
    "setGainMode",
    "setFrequencyCorrection",
    "getHardwareTime",
    "setHardwareTime",
)

MODULE_CALLABLES = (
    "extractBuffPointer",
)

CONSTANTS = (
    "SOAPY_SDR_TX",
    "SOAPY_SDR_RX",
    "SOAPY_SDR_CF32",
    "SOAPY_SDR_CS16",
    "SOAPY_SDR_END_BURST",
    "SOAPY_SDR_HAS_TIME",
    "SOAPY_SDR_TIMEOUT",
    "SOAPY_SDR_OVERFLOW",
)


def _print_block(title: str, text: str) -> None:
    print(f"\n### {title}")
    print(text.rstrip() or "(empty)")


def _signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # noqa: BLE001 - SWIG objects often do not expose signatures
        return f"(unavailable: {type(exc).__name__}: {exc})"


def _doc(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    return textwrap.indent(doc, "  ") if doc else "  (no docstring)"


def _source(obj: object, *, max_lines: int) -> str:
    try:
        source = inspect.getsource(obj)
    except Exception as exc:  # noqa: BLE001 - compiled/SWIG methods may not have source
        return f"  (unavailable: {type(exc).__name__}: {exc})"
    lines = source.rstrip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(source.splitlines()) - max_lines} more lines)"]
    return textwrap.indent("\n".join(lines), "  ")


def _print_callable(name: str, obj: Callable[..., object], *, max_source_lines: int) -> None:
    print(f"\n## {name}")
    print(f"repr: {obj!r}")
    print(f"signature: {_signature(obj)}")
    print("doc:")
    print(_doc(obj))
    print("source:")
    print(_source(obj, max_lines=max_source_lines))


def _print_device_probe(soapy, device_args: str) -> None:
    print("\n# Device Probe")
    dev = soapy.Device(device_args)
    tx = soapy.SOAPY_SDR_TX
    try:
        for method in ("getDriverKey", "getHardwareKey", "getHardwareInfo"):
            fn = getattr(dev, method, None)
            if callable(fn):
                with _suppress_to_text(method):
                    print(f"{method}: {fn()}")
        tx_channels = 1
        get_num_channels = getattr(dev, "getNumChannels", None)
        if callable(get_num_channels):
            with _suppress_to_text("getNumChannels(TX)"):
                tx_channels = int(get_num_channels(tx))
                print(f"getNumChannels(TX): {tx_channels}")
        for channel in range(max(1, tx_channels)):
            with _suppress_to_text(f"getStreamFormats(TX,{channel})"):
                print(f"getStreamFormats(TX,{channel}): {list(dev.getStreamFormats(tx, channel))}")
            with _suppress_to_text(f"getNativeStreamFormat(TX,{channel})"):
                native = dev.getNativeStreamFormat(tx, channel)
                print(f"getNativeStreamFormat(TX,{channel}): {native}")
            with _suppress_to_text(f"listGains(TX,{channel})"):
                print(f"listGains(TX,{channel}): {list(dev.listGains(tx, channel))}")
            with _suppress_to_text(f"getGainRange(TX,{channel})"):
                print(f"getGainRange(TX,{channel}): {dev.getGainRange(tx, channel)}")
    finally:
        _release_device(soapy, dev)


class _suppress_to_text:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is not None:
            print(f"{self.label}: ERROR {type(exc).__name__}: {exc}")
            return True
        return False


def _release_device(soapy, dev) -> None:
    close = getattr(dev, "close", None)
    if callable(close):
        with _suppress_to_text("device.close"):
            close()
        return
    unmake = getattr(getattr(soapy, "Device", None), "unmake", None)
    if callable(unmake):
        with _suppress_to_text("Device.unmake"):
            unmake(dev)
        return
    unmake = getattr(soapy, "Device_unmake", None)
    if callable(unmake):
        with _suppress_to_text("Device_unmake"):
            unmake(dev)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="", help="optional SoapySDR args for read-only probe")
    parser.add_argument("--max-source-lines", type=int, default=80)
    args = parser.parse_args(argv)

    import SoapySDR  # noqa: PLC0415 - this script exists to inspect the installed binding

    print("# SoapySDR Python Binding")
    print(f"python: {sys.version}")
    print(f"module: {SoapySDR!r}")
    print(f"module_file: {getattr(SoapySDR, '__file__', '(unknown)')}")
    for name in ("SOAPY_SDR_API_VERSION", "SOAPY_SDR_ABI_VERSION"):
        if hasattr(SoapySDR, name):
            print(f"{name}: {getattr(SoapySDR, name)}")

    _print_block(
        "Constants",
        "\n".join(f"{name}={getattr(SoapySDR, name, '(missing)')!r}" for name in CONSTANTS),
    )

    for name in MODULE_CALLABLES:
        obj = getattr(SoapySDR, name, None)
        if callable(obj):
            _print_callable(f"SoapySDR.{name}", obj, max_source_lines=args.max_source_lines)
        else:
            print(f"\n## SoapySDR.{name}\n(missing)")

    device_cls = SoapySDR.Device
    stream_related = sorted(
        name for name in dir(device_cls) if "writeStream" in name or "readStreamStatus" in name
    )
    _print_block("Device stream-related dir names", "\n".join(stream_related))
    for name in METHODS:
        obj = getattr(device_cls, name, None)
        if callable(obj):
            _print_callable(f"Device.{name}", obj, max_source_lines=args.max_source_lines)
        else:
            print(f"\n## Device.{name}\n(missing)")
    for name in stream_related:
        if name in METHODS:
            continue
        obj = getattr(device_cls, name, None)
        if callable(obj):
            _print_callable(f"Device.{name}", obj, max_source_lines=args.max_source_lines)

    if args.device:
        _print_device_probe(SoapySDR, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

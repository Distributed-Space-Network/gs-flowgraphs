#!/usr/bin/env python3
"""Crash-isolated SoapySDR TX write probe for the XTRX bench path.

The parent process runs one tiny TX write per child process. If SoapyXTRX aborts, hangs, or wedges
inside writeStream(), only that child dies and the parent can still report which combination failed.

The child writes a short zero-amplitude buffer, so this probes the host/driver write path without
modulating a command frame.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import itertools
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FAILURE_MARKERS = (
    ("xtrx-delayed-buffers", "TX DMA Current delayed buffers"),
    ("xtrx-dma-timeout", "TX DMA TO"),
    ("xtrx-dma-error", "TX DMA ERROR"),
    ("xtrx-stream-error", "SoapyXTRX::writeStream"),
    ("python-abort", "Fatal Python error: Aborted"),
    ("writeStream-binding", "__writeStream"),
)

STATUS_ORDER = ("OK", "OK_DRIVER_ERR", "EXIT_10", "HANG", "SIGNAL_6")
XTRX_MIN_TX_BW_HZ = 800_000.0
XTRX_MIN_TX_RATE_HZ = 2_100_000.0
XTRX_LOW_RATE_EXPLICIT_CHANNELS_HZ = 1_000_000.0
XTRX_VERY_LOW_RATE_CF32_HZ = 500_000.0
GFSK_PROBE_PAYLOAD = b"SOAPY-TX-PROBE"


@dataclass(frozen=True)
class Case:
    fmt: str
    channels: str
    stream_args: str
    activate: str
    write_call: str
    layout: str
    chunk: int

    def name(self) -> str:
        return (
            f"fmt={self.fmt} channels={self.channels} stream_args={self.stream_args} "
            f"activate={self.activate} write={self.write_call} layout={self.layout} "
            f"chunk={self.chunk}"
        )


@dataclass(frozen=True)
class TrialResult:
    status: str
    reasons: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextlib.contextmanager
def _probe_lock(path: str, *, disabled: bool = False):
    if disabled or not path:
        yield
        return

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            text = ""
            with contextlib.suppress(OSError):
                text = lock_path.read_text(encoding="utf-8", errors="replace")
            pid = 0
            for line in text.splitlines():
                if line.startswith("pid="):
                    with contextlib.suppress(ValueError):
                        pid = int(line.partition("=")[2])
            if pid and not _pid_alive(pid):
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                continue
            msg = (
                f"probe lock exists at {lock_path} (pid={pid or 'unknown'}); "
                "use --no-lock to force"
            )
            raise RuntimeError(msg) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        break

    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()


def _parse_settings(text: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for item in (text or "").split(","):
        item = item.strip()
        if "=" not in item:
            continue
        name, _, value = item.partition("=")
        with contextlib.suppress(ValueError):
            pairs.append((name.strip(), float(value)))
    return [(name, value) for name, value in pairs if name]


def _child_phase(name: str) -> None:
    print(f"child: phase={name}", flush=True)


def _child_setup_stream(dev, soapy, args):
    tx = soapy.SOAPY_SDR_TX
    fmt = soapy.SOAPY_SDR_CS16 if args.format == "cs16" else soapy.SOAPY_SDR_CF32
    channels = [args.channel] if args.channels == "explicit" else []
    stream_args = (
        {"WIRE": "CS16", "linkFormat": "CS16"}
        if args.stream_args == "wire-cs16"
        else {}
    )

    if channels or stream_args:
        return dev.setupStream(tx, fmt, channels, stream_args)
    return dev.setupStream(tx, fmt)


def _ret_name(ret) -> str:
    with contextlib.suppress(TypeError, ValueError):
        ret = int(ret)
        return {
            0: "OK",
            -1: "TIMEOUT",
            -2: "STREAM_ERROR",
            -3: "CORRUPTION",
            -4: "OVERFLOW",
            -5: "NOT_SUPPORTED",
            -6: "TIME_ERROR",
            -7: "UNDERFLOW",
        }.get(ret, str(ret))
    return str(ret)


def _gfsk_probe_iq(sample_rate: float, min_samples: int, *, start_sample: int = 0) -> np.ndarray:
    import tx_gfsk  # noqa: PLC0415 - probe-only reuse of known-good TX modulator

    want = max(0, int(start_sample)) + max(1, int(min_samples))
    iq = tx_gfsk.build_frame_iq(
        GFSK_PROBE_PAYLOAD,
        framing="endurosat",
        sample_rate=float(sample_rate),
        symbol_rate=tx_gfsk.DEFAULT_SYMBOL_RATE,
        mod_index=0.5,
        bt=0.5,
    )
    samples = np.asarray(iq, dtype=np.complex64)
    if samples.size >= want:
        return samples[start_sample:want]
    reps = int(np.ceil(want / max(1, samples.size)))
    return np.tile(samples, reps)[start_sample:want].astype(np.complex64, copy=False)


def _sample_pattern(
    fmt: str,
    layout: str,
    chunk: int,
    pattern: str,
    amplitude: float,
    sample_rate: float,
    sample_offset: int = 0,
):
    n = max(1, chunk)
    amp = max(0.0, min(1.0, float(amplitude)))
    if pattern == "gfsk":
        iq = _gfsk_probe_iq(sample_rate, n, start_sample=sample_offset)
        iq = np.asarray(iq * amp, dtype=np.complex64)
        if fmt == "cf32":
            return np.ascontiguousarray(iq), n
        if layout == "flat":
            block = np.empty(n * 2, dtype=np.int16)
            block[0::2] = np.rint(np.clip(iq.real, -1.0, 1.0) * 32767.0).astype(np.int16)
            block[1::2] = np.rint(np.clip(iq.imag, -1.0, 1.0) * 32767.0).astype(np.int16)
            return np.ascontiguousarray(block), n
        block = np.empty((n, 2), dtype=np.int16)
        block[:, 0] = np.rint(np.clip(iq.real, -1.0, 1.0) * 32767.0).astype(np.int16)
        block[:, 1] = np.rint(np.clip(iq.imag, -1.0, 1.0) * 32767.0).astype(np.int16)
        return np.ascontiguousarray(block), n
    if fmt == "cf32":
        block = np.empty(n, dtype=np.complex64)
        if pattern == "dc":
            block.fill(np.complex64(complex(amp, 0.0)))
        elif pattern == "tone":
            phase = np.arange(n, dtype=np.float32) * (2.0 * np.pi / 16.0)
            block[:] = amp * (np.cos(phase) + 1j * np.sin(phase))
        elif pattern == "ramp":
            vals = np.linspace(-amp, amp, n, dtype=np.float32)
            block[:] = vals + 1j * vals[::-1]
        else:
            block.fill(np.complex64(0.0))
        return np.ascontiguousarray(block), n

    peak = 32767
    i_val = int(round(amp * peak))
    if layout == "flat":
        block = np.empty(n * 2, dtype=np.int16)
        block.fill(0)
        if pattern == "dc":
            block[0::2] = i_val
        elif pattern == "tone":
            phase = np.arange(n, dtype=np.float32) * (2.0 * np.pi / 16.0)
            block[0::2] = np.rint(np.cos(phase) * i_val).astype(np.int16)
            block[1::2] = np.rint(np.sin(phase) * i_val).astype(np.int16)
        elif pattern == "ramp":
            vals = np.rint(np.linspace(-i_val, i_val, n)).astype(np.int16)
            block[0::2] = vals
            block[1::2] = vals[::-1]
        return np.ascontiguousarray(block), n

    block = np.empty((n, 2), dtype=np.int16)
    block.fill(0)
    if pattern == "dc":
        block[:, 0] = i_val
    elif pattern == "tone":
        phase = np.arange(n, dtype=np.float32) * (2.0 * np.pi / 16.0)
        block[:, 0] = np.rint(np.cos(phase) * i_val).astype(np.int16)
        block[:, 1] = np.rint(np.sin(phase) * i_val).astype(np.int16)
    elif pattern == "ramp":
        vals = np.rint(np.linspace(-i_val, i_val, n)).astype(np.int16)
        block[:, 0] = vals
        block[:, 1] = vals[::-1]
    return np.ascontiguousarray(block), n


def _make_block(fmt: str, layout: str, chunk: int, args, *, sample_offset: int = 0):
    return _sample_pattern(
        fmt,
        layout,
        max(1, chunk),
        args.pattern,
        args.amplitude,
        float(args.sample_rate),
        sample_offset,
    )


def _use_overall_tx_gain(args) -> bool:
    if args.gain is None:
        return False
    if "xtrx" in str(args.device).lower() and not args.allow_xtrx_overall_gain:
        print(
            "child: skip overall --gain for XTRX; use --other-settings=PAD=<dB> "
            "or --allow-xtrx-overall-gain to force",
            flush=True,
        )
        return False
    return True


def _write(dev, stream, block, num_elems: int, args, *, flags: int):
    if args.write_call == "simple":
        return dev.writeStream(stream, [block], num_elems)
    if args.write_call == "flags":
        return dev.writeStream(stream, [block], num_elems, flags)
    return dev.writeStream(stream, [block], num_elems, flags, 0, int(args.write_timeout_us))


def _read_stream_status(dev, stream, args, label: str) -> None:
    polls = max(1, int(args.status_polls))
    timeout_us = max(0, int(args.status_timeout_us))
    for poll in range(1, polls + 1):
        _child_phase(f"status-{label}-{poll}-enter")
        t0 = time.monotonic()
        try:
            try:
                sr = dev.readStreamStatus(stream, timeoutUs=timeout_us)
            except TypeError:
                sr = dev.readStreamStatus(stream, timeout_us)
        except Exception as exc:  # noqa: BLE001 - status probing must report driver behavior
            _child_phase(f"status-{label}-{poll}-error")
            print(
                f"child: status {label} {poll}/{polls} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.status_errors_fatal:
                raise
        else:
            _child_phase(f"status-{label}-{poll}-ok")
            ret = getattr(sr, "ret", None)
            print(
                f"child: status {label} {poll}/{polls} ret={ret} ret_name={_ret_name(ret)} "
                f"flags={getattr(sr, 'flags', None)} timeNs={getattr(sr, 'timeNs', None)} "
                f"dt_ms={1000.0 * (time.monotonic() - t0):.1f}",
                flush=True,
            )
        if args.status_gap_ms > 0 and poll < polls:
            time.sleep(args.status_gap_ms / 1000.0)


def _release_device(soapy, dev) -> None:
    close = getattr(dev, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()
        return
    unmake = getattr(getattr(soapy, "Device", None), "unmake", None)
    if callable(unmake):
        with contextlib.suppress(Exception):
            unmake(dev)
        return
    unmake = getattr(soapy, "Device_unmake", None)
    if callable(unmake):
        with contextlib.suppress(Exception):
            unmake(dev)


def _tx_gfsk_direct_args(args):
    return SimpleNamespace(
        soapy_tx_device=args.device,
        samp_rate=float(args.sample_rate),
        tx_freq=float(args.freq),
        bw=float(args.bandwidth),
        allow_narrow_bw=not args.xtrx_safe_bw,
        channel=int(args.channel),
        antenna="",
        gain=args.gain,
        allow_xtrx_overall_gain=args.allow_xtrx_overall_gain,
        other_settings=args.other_settings,
        ppm=0.0,
        tx_write_call="auto",
        tx_stream_channels="auto",
        tx_activate_elems="auto",
        tx_format="auto",
        tx_scale=float(args.amplitude),
        tx_write_timeout_us=int(args.write_timeout_us),
        tx_copy_chunks=False,
        tx_pace=bool(args.pace),
        tx_write_sleep_us=int(max(0.0, args.write_gap_ms) * 1000.0),
        tx_time_mode="none",
        tx_time_lead_ms=50.0,
        allow_xtrx_timed_tx=False,
    )


def child_tx_gfsk_direct_main(args) -> int:
    faulthandler.enable(all_threads=True)

    import SoapySDR  # noqa: PLC0415 - target-only hardware probe
    import tx_gfsk  # noqa: PLC0415 - compare against known-good TX path

    dev = None
    tx_args = _tx_gfsk_direct_args(args)
    try:
        print(f"child: case {args.case_name}", flush=True)
        _child_phase("tx-gfsk-direct-build-enter")
        used_rate = tx_gfsk.resolve_rate(float(args.sample_rate), tx_gfsk.DEFAULT_SYMBOL_RATE)
        iq = tx_gfsk.build_frame_iq(
            GFSK_PROBE_PAYLOAD,
            framing="endurosat",
            sample_rate=used_rate,
            symbol_rate=tx_gfsk.DEFAULT_SYMBOL_RATE,
            mod_index=0.5,
            bt=0.5,
        )
        _child_phase("tx-gfsk-direct-build-ok")
        print(
            f"child: tx_gfsk_direct samples={len(iq)} sample_rate={used_rate:g} "
            f"tx_format={tx_gfsk._resolve_tx_format(tx_args, used_rate)} "
            f"channels={tx_gfsk._resolve_tx_stream_channels(tx_args, used_rate)} "
            f"write_call={tx_gfsk._resolve_tx_write_call(tx_args)} "
            f"activate={tx_gfsk._resolve_tx_activate_mode(tx_args)}",
            flush=True,
        )

        _child_phase("tx-gfsk-direct-device-open-enter")
        dev = SoapySDR.Device(args.device)
        _child_phase("tx-gfsk-direct-device-open-ok")
        _child_phase("tx-gfsk-direct-configure-enter")
        tx_gfsk._configure_tx(dev, tx_args, int(args.channel), used_rate)
        _child_phase("tx-gfsk-direct-configure-ok")
        _child_phase("tx-gfsk-direct-transmit-enter")
        tx_gfsk._transmit(
            dev,
            int(args.channel),
            iq,
            repeat=1,
            gap_s=0.0,
            sample_rate=used_rate,
            tx_chunk=int(args.chunk),
            write_call=tx_gfsk._resolve_tx_write_call(tx_args),
            write_timeout_us=int(args.write_timeout_us),
            copy_chunks=False,
            pace=bool(args.pace),
            write_sleep_us=int(max(0.0, args.write_gap_ms) * 1000.0),
            stream_channels=tx_gfsk._resolve_tx_stream_channels(tx_args, used_rate),
            activate_elems_mode=tx_gfsk._resolve_tx_activate_mode(tx_args),
            tx_format=tx_gfsk._resolve_tx_format(tx_args, used_rate),
            tx_scale=float(args.amplitude),
            tx_time_mode=tx_gfsk._resolve_tx_time_mode(tx_args),
            tx_time_lead_ms=50.0,
        )
        _child_phase("tx-gfsk-direct-transmit-ok")
        return 0
    finally:
        if dev is not None:
            _child_phase("tx-gfsk-direct-release-enter")
            with contextlib.suppress(Exception):
                tx_gfsk._release_soapy_device(SoapySDR, dev)
            _child_phase("tx-gfsk-direct-release-ok")


def child_main(args) -> int:
    faulthandler.enable(all_threads=True)

    if args.direct_tx_gfsk:
        return child_tx_gfsk_direct_main(args)

    import SoapySDR  # noqa: PLC0415 - target-only hardware probe

    tx = SoapySDR.SOAPY_SDR_TX
    dev = None
    stream = None
    try:
        print(f"child: case {args.case_name}", flush=True)
        _child_phase("device-open-enter")
        dev = SoapySDR.Device(args.device)
        _child_phase("device-open-ok")
        sample_rate = float(args.sample_rate)
        if (
            args.xtrx_safe_rate
            and "xtrx" in str(args.device).lower()
            and sample_rate < XTRX_MIN_TX_RATE_HZ
        ):
            print(
                f"child: sample-rate lifted {sample_rate:.0f}->{XTRX_MIN_TX_RATE_HZ:.0f}",
                flush=True,
            )
            sample_rate = XTRX_MIN_TX_RATE_HZ
            args.sample_rate = sample_rate
        _child_phase("set-sample-rate-enter")
        dev.setSampleRate(tx, args.channel, sample_rate)
        _child_phase("set-sample-rate-ok")
        _child_phase("set-frequency-enter")
        dev.setFrequency(tx, args.channel, float(args.freq))
        _child_phase("set-frequency-ok")
        if args.bandwidth > 0:
            bandwidth = float(args.bandwidth)
            if (
                args.xtrx_safe_bw
                and "xtrx" in str(args.device).lower()
                and bandwidth < XTRX_MIN_TX_BW_HZ
            ):
                print(
                    f"child: bandwidth lifted {bandwidth:.0f}->{XTRX_MIN_TX_BW_HZ:.0f}",
                    flush=True,
                )
                bandwidth = XTRX_MIN_TX_BW_HZ
            with contextlib.suppress(Exception):
                _child_phase("set-bandwidth-enter")
                dev.setBandwidth(tx, args.channel, bandwidth)
                _child_phase("set-bandwidth-ok")
        with contextlib.suppress(Exception):
            _child_phase("set-gain-mode-enter")
            dev.setGainMode(tx, args.channel, False)
            _child_phase("set-gain-mode-ok")
        if _use_overall_tx_gain(args):
            with contextlib.suppress(Exception):
                _child_phase("set-gain-overall-enter")
                dev.setGain(tx, args.channel, float(args.gain))
                _child_phase("set-gain-overall-ok")
        for name, value in _parse_settings(args.other_settings):
            with contextlib.suppress(Exception):
                _child_phase(f"get-gain-{name}-enter")
                current = float(dev.getGain(tx, args.channel, name))
                _child_phase(f"get-gain-{name}-ok")
                if abs(current - value) < 1e-9:
                    print(f"child: gain {name} already {value:g}; skip setGain", flush=True)
                    continue
            with contextlib.suppress(Exception):
                _child_phase(f"set-gain-{name}-enter")
                dev.setGain(tx, args.channel, name, value)
                _child_phase(f"set-gain-{name}-ok")

        _child_phase("setup-stream-enter")
        stream = _child_setup_stream(dev, SoapySDR, args)
        _child_phase("setup-stream-ok")
        _child_phase("get-mtu-enter")
        mtu = int(dev.getStreamMTU(stream))
        _child_phase("get-mtu-ok")
        print(f"child: stream={stream!r} mtu={mtu}", flush=True)

        block, num_elems = _make_block(args.format, args.layout, min(args.chunk, mtu), args)
        print(
            f"child: block dtype={block.dtype} shape={block.shape} strides={block.strides} "
            f"nbytes={block.nbytes} c_contig={block.flags.c_contiguous} "
            f"pattern={args.pattern} amplitude={args.amplitude:g} num_elems={num_elems}",
            flush=True,
        )

        _child_phase("activate-enter")
        if args.activate == "default":
            ar = dev.activateStream(stream)
        elif args.activate == "zero":
            ar = dev.activateStream(stream, 0, 0, 0)
        else:
            ar = dev.activateStream(stream, 0, 0, num_elems)
        _child_phase("activate-ok")
        print(f"child: activate ret={ar}", flush=True)
        if args.post_activate_sleep_ms > 0:
            _child_phase("post-activate-sleep-enter")
            time.sleep(args.post_activate_sleep_ms / 1000.0)
            _child_phase("post-activate-sleep-ok")
        if args.status_after_activate:
            _read_stream_status(dev, stream, args, "after-activate")

        total_writes = max(1, int(args.child_writes))
        accepted = 0
        end_burst = getattr(SoapySDR, "SOAPY_SDR_END_BURST", 2)
        for write_index in range(1, total_writes + 1):
            if write_index > 1 and args.pattern == "gfsk":
                block, num_elems = _make_block(
                    args.format,
                    args.layout,
                    min(args.chunk, mtu),
                    args,
                    sample_offset=accepted,
                )
            flags = end_burst if args.end_burst == "last" and write_index == total_writes else 0
            if args.status_before_write:
                _read_stream_status(dev, stream, args, f"before-write-{write_index}")
            _child_phase(f"write-{write_index}-enter")
            t0 = time.monotonic()
            sr = _write(dev, stream, block, num_elems, args, flags=flags)
            _child_phase(f"write-{write_index}-ok")
            ret = int(getattr(sr, "ret", 0) or 0)
            print(
                f"child: write {write_index}/{total_writes} ret={ret} "
                f"flags={getattr(sr, 'flags', None)} sent_flags={flags} "
                f"dt_ms={1000.0 * (time.monotonic() - t0):.1f}",
                flush=True,
            )
            if ret <= 0:
                return 10
            accepted += ret
            if args.status_after_write:
                _read_stream_status(dev, stream, args, f"after-write-{write_index}")
            if args.pace:
                time.sleep(ret / max(1.0, float(args.sample_rate)))
            if args.write_gap_ms > 0 and write_index < total_writes:
                time.sleep(args.write_gap_ms / 1000.0)
        print(f"child: accepted_total={accepted}", flush=True)
        return 0
    finally:
        if stream is not None and dev is not None:
            with contextlib.suppress(Exception):
                _child_phase("deactivate-enter")
                dev.deactivateStream(stream)
                _child_phase("deactivate-ok")
            with contextlib.suppress(Exception):
                _child_phase("close-stream-enter")
                dev.closeStream(stream)
                _child_phase("close-stream-ok")
        if dev is not None:
            _child_phase("release-device-enter")
            _release_device(SoapySDR, dev)
            _child_phase("release-device-ok")


def _default_cases(chunk: int) -> list[Case]:
    return [
        Case("cs16", "default", "none", "default", "simple", "2col", chunk),
        Case("cs16", "default", "none", "default", "simple", "flat", chunk),
        Case("cs16", "explicit", "none", "default", "simple", "2col", chunk),
        Case("cs16", "explicit", "none", "default", "simple", "flat", chunk),
        Case("cf32", "default", "none", "default", "simple", "cf32", chunk),
        Case("cf32", "explicit", "none", "default", "simple", "cf32", chunk),
        Case("cs16", "default", "wire-cs16", "default", "simple", "2col", chunk),
        Case("cf32", "default", "wire-cs16", "default", "simple", "cf32", chunk),
    ]


def _write_call_cases(chunk: int, *, include_wire_cs16: bool = False) -> list[Case]:
    """Focused matrix for first-write hangs: keep buffer layout sane and vary call/activation."""
    base = (
        ("cs16", "default", "flat"),
        ("cs16", "explicit", "flat"),
        ("cf32", "default", "cf32"),
        ("cf32", "explicit", "cf32"),
    )
    stream_arg_modes = ("none", "wire-cs16") if include_wire_cs16 else ("none",)
    cases: list[Case] = []
    for fmt, channels, layout in base:
        for stream_args, activate, write_call in itertools.product(
            stream_arg_modes, ("default", "zero", "num-elems"), ("simple", "flags", "full")
        ):
            cases.append(
                Case(fmt, channels, stream_args, activate, write_call, layout, chunk)
            )
    return cases


def _single_case(args) -> list[Case]:
    return [
        Case(
            args.format,
            args.channels,
            args.stream_args,
            args.activate,
            args.write_call,
            args.layout,
            args.chunk,
        )
    ]


def _tx_gfsk_case(args) -> list[Case]:
    is_xtrx = "xtrx" in str(args.device).lower()
    sample_rate = float(args.sample_rate)
    fmt = "cf32" if is_xtrx and sample_rate <= XTRX_VERY_LOW_RATE_CF32_HZ else "cs16"
    channels = (
        "explicit"
        if is_xtrx and sample_rate <= XTRX_LOW_RATE_EXPLICIT_CHANNELS_HZ
        else "default"
    )
    layout = "cf32" if fmt == "cf32" else "flat"
    return [Case(fmt, channels, "none", "default", "simple", layout, args.chunk)]


def _expanded_cases(chunk: int) -> list[Case]:
    cases: list[Case] = []
    for fmt, channels, stream_args, activate, write_call in itertools.product(
        ("cs16", "cf32"),
        ("default", "explicit"),
        ("none", "wire-cs16"),
        ("default", "zero", "num-elems"),
        ("simple", "flags", "full"),
    ):
        layouts = ("cf32",) if fmt == "cf32" else ("2col", "flat")
        for layout in layouts:
            cases.append(Case(fmt, channels, stream_args, activate, write_call, layout, chunk))
    return cases


def _filtered_cases(cases: list[Case], args) -> list[Case]:
    filters = (
        ("fmt", args.only_format),
        ("channels", args.only_channels),
        ("stream_args", args.only_stream_args),
        ("activate", args.only_activate),
        ("write_call", args.only_write_call),
        ("layout", args.only_layout),
    )
    out = []
    for case in cases:
        if all(not want or str(getattr(case, attr)) == want for attr, want in filters):
            out.append(case)
    return out


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _reasons(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for tag, marker in FAILURE_MARKERS:
        if marker in text:
            found.append(tag)
    if " ERROR:" in text and not any(tag.startswith("xtrx-") for tag in found):
        found.append("driver-error")
    return tuple(dict.fromkeys(found))


def _last_phase(text: str) -> str:
    phase = ""
    for line in text.splitlines():
        if "child: phase=" in line:
            phase = line.rsplit("child: phase=", 1)[-1].strip()
    return phase


def _last_status(text: str) -> str:
    status = ""
    for line in text.splitlines():
        if line.startswith("child: status "):
            parts = line.split()
            label = parts[2] if len(parts) >= 3 else "unknown"
            ret = next(
                (part.partition("=")[2] for part in parts if part.startswith("ret=")),
                "?",
            )
            ret_name = next(
                (part.partition("=")[2] for part in parts if part.startswith("ret_name=")),
                _ret_name(ret),
            )
            flags = next(
                (part.partition("=")[2] for part in parts if part.startswith("flags=")),
                "?",
            )
            status = f"{label}:ret={ret}:{ret_name}:flags={flags}"
    return status


def _with_phase_reasons(text: str, reasons: tuple[str, ...]) -> tuple[str, ...]:
    out = list(reasons)
    phase = _last_phase(text)
    if phase:
        out.append(f"last-phase:{phase}")
    status = _last_status(text)
    if status:
        out.append(f"last-status:{status}")
    return tuple(dict.fromkeys(out))


def _classify_completed(cp: subprocess.CompletedProcess[str]) -> TrialResult:
    stdout = _text(cp.stdout)
    stderr = _text(cp.stderr)
    text = f"{stderr}\n{stdout}"
    base_reasons = _reasons(text)
    if cp.returncode == 0 and not base_reasons:
        reasons = ()
    else:
        reasons = _with_phase_reasons(text, base_reasons)
    if cp.returncode == 0:
        status = "OK_DRIVER_ERR" if reasons else "OK"
    elif cp.returncode < 0:
        status = f"SIGNAL_{-cp.returncode}"
    else:
        status = f"EXIT_{cp.returncode}"
    return TrialResult(status, reasons, cp.returncode, stdout, stderr)


def _classify_timeout(exc: subprocess.TimeoutExpired) -> TrialResult:
    stdout = _text(exc.stdout)
    stderr = _text(exc.stderr)
    text = f"{stderr}\n{stdout}"
    reasons = _with_phase_reasons(text, _reasons(text))
    reasons = tuple(dict.fromkeys(("timeout", *reasons)))
    return TrialResult("HANG", reasons, None, stdout, stderr)


def _first_write_wedge(result: TrialResult) -> bool:
    return (
        (result.status == "HANG" or result.status.startswith("SIGNAL_"))
        and "last-phase:write-1-enter" in result.reasons
    )


def _ordered_counts(counts: Counter[str]) -> str:
    def key(item: tuple[str, int]) -> tuple[int, str]:
        status, _ = item
        try:
            return (STATUS_ORDER.index(status), status)
        except ValueError:
            return (len(STATUS_ORDER), status)

    return " ".join(f"{status}={count}" for status, count in sorted(counts.items(), key=key))


def _verdict(counts: Counter[str]) -> str:
    total = sum(counts.values())
    ok = counts.get("OK", 0)
    ok_warn = counts.get("OK_DRIVER_ERR", 0)
    if total <= 0:
        return "NO_TRIALS"
    if ok == total:
        return "STABLE_CLEAN"
    if ok + ok_warn == total:
        return "ACCEPTS_WITH_DRIVER_ERRORS"
    if ok + ok_warn > 0:
        return "FLAKY"
    return "FAILS"


def _print_summary(
    cases: list[Case],
    status_by_case: dict[Case, Counter[str]],
    reasons_by_case: dict[Case, Counter[str]],
) -> None:
    print("\n# Summary", flush=True)
    for case in cases:
        statuses = status_by_case.get(case, Counter())
        reasons = reasons_by_case.get(case, Counter())
        reason_text = ", ".join(
            f"{reason}({count})" for reason, count in reasons.most_common()
        ) or "-"
        print(
            f"{_verdict(statuses):26} {_ordered_counts(statuses):34} "
            f"reasons={reason_text} :: {case.name()}",
            flush=True,
        )


def _accepted_cases(
    cases: list[Case],
    status_by_case: dict[Case, Counter[str]],
) -> list[tuple[Case, int, int]]:
    accepted = []
    for case in cases:
        statuses = status_by_case.get(case, Counter())
        clean = statuses.get("OK", 0)
        dirty = statuses.get("OK_DRIVER_ERR", 0)
        if clean or dirty:
            accepted.append((case, clean, dirty))
    return accepted


def _sum_statuses(cases: list[Case], status_by_case: dict[Case, Counter[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(status_by_case.get(case, Counter()))
    return counts


def _sum_reasons(cases: list[Case], reasons_by_case: dict[Case, Counter[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(reasons_by_case.get(case, Counter()))
    return counts


def _print_diagnosis(
    cases: list[Case],
    status_by_case: dict[Case, Counter[str]],
    reasons_by_case: dict[Case, Counter[str]],
) -> None:
    statuses = _sum_statuses(cases, status_by_case)
    reasons = _sum_reasons(cases, reasons_by_case)
    total = sum(statuses.values())
    clean_ok = statuses.get("OK", 0)
    dirty_ok = statuses.get("OK_DRIVER_ERR", 0)
    accepted = clean_ok + dirty_ok
    if total <= 0:
        return

    print("\n# Overall", flush=True)
    reason_text = ", ".join(f"{reason}({count})" for reason, count in reasons.most_common()) or "-"
    print(f"trials={total} {_ordered_counts(statuses)} reasons={reason_text}", flush=True)
    for case, clean, dirty in _accepted_cases(cases, status_by_case):
        print(
            f"accepted clean={clean} dirty={dirty} :: {case.name()}",
            flush=True,
        )

    write_enter_failures = reasons.get("last-phase:write-1-enter", 0)
    if accepted == 0 and write_enter_failures == total:
        print("\n# Diagnosis", flush=True)
        print(
            "GLOBAL_FIRST_WRITE_FAILURE: every selected trial reached stream activation, then "
            "hung or aborted inside the first writeStream call.",
            flush=True,
        )
        print(
            "This run cannot choose a best write format; format/channel/activation/write-call "
            "variants all failed before any write returned.",
            flush=True,
        )
        print(
            "The common failure is below the Python call shape: XTRX TX DMA/start state, "
            "device contention, or an unsupported rate/clock/bandwidth configuration.",
            flush=True,
        )
    elif clean_ok == 0 and dirty_ok > 0 and write_enter_failures == total - dirty_ok:
        print("\n# Diagnosis", flush=True)
        if dirty_ok == total:
            print(
                "ALL_WRITES_RETURNED_WITH_DRIVER_ERRORS: every selected trial returned from "
                "writeStream, but SoapyXTRX logged TX driver errors.",
                flush=True,
            )
            print(
                "This matches the known-good call shape at the Python layer; the remaining "
                "question is whether the driver warning is benign for this RF path or still "
                "correlates with bad samples on air.",
                flush=True,
            )
        else:
            print(
                "NO_CLEAN_WRITE_FORMAT: at least one write returned, but every returned case also "
                "reported driver errors.",
                flush=True,
            )
            print(
                "Do not treat OK_DRIVER_ERR as a working format; in this probe it only means "
                "writeStream returned before SoapyXTRX logged a TX problem.",
                flush=True,
            )
            print(
                "The dominant failure remains first-write startup: all non-returning trials "
                "stopped inside writeStream before a result was available.",
                flush=True,
            )
    elif accepted and write_enter_failures:
        print("\n# Diagnosis", flush=True)
        print(
            "FLAKY_FIRST_WRITE: at least one write returned, but some trials still hung or "
            "aborted inside the first writeStream call.",
            flush=True,
        )
        print(
            "Treat OK_DRIVER_ERR as suspect; compare only cases with repeated clean OK results.",
            flush=True,
        )
    elif accepted == total:
        print("\n# Diagnosis", flush=True)
        print(
            "ALL_WRITES_RETURNED: compare OK vs OK_DRIVER_ERR before trusting a case.",
            flush=True,
        )


def parent_main(args) -> int:
    try:
        with _probe_lock(args.lock_file, disabled=args.no_lock):
            return _parent_main_locked(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _parent_main_locked(args) -> int:
    if args.stability:
        args.repeat = max(args.repeat, 10)
        args.shuffle = True
        if args.cooldown_s <= 0:
            args.cooldown_s = 0.25
        if args.print_child_output == "all":
            args.print_child_output = "none"

    case_set = "expanded" if args.expanded else args.case_set
    if case_set == "expanded":
        cases = _expanded_cases(args.chunk)
    elif case_set == "write-calls":
        cases = _write_call_cases(args.chunk, include_wire_cs16=args.include_wire_cs16)
    elif case_set == "single":
        cases = _single_case(args)
    elif case_set in {"tx-gfsk", "tx-gfsk-direct"}:
        cases = _tx_gfsk_case(args)
    else:
        cases = _default_cases(args.chunk)
    cases = _filtered_cases(cases, args)
    if not cases:
        print("No cases matched the selected filters.", file=sys.stderr)
        return 2

    trials = [(repeat_index, case) for repeat_index in range(1, args.repeat + 1) for case in cases]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(trials)
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    script = Path(__file__).resolve()
    status_by_case: dict[Case, Counter[str]] = defaultdict(Counter)
    reasons_by_case: dict[Case, Counter[str]] = defaultdict(Counter)
    worst = 0
    try:
        for index, (repeat_index, case) in enumerate(trials, start=1):
            print(
                f"\n[{index}/{len(trials)} repeat={repeat_index}/{args.repeat}] {case.name()}",
                flush=True,
            )
            cmd = [
                sys.executable,
                str(script),
                "--child",
                "--case-name",
                case.name(),
                "--device",
                args.device,
                "--freq",
                str(args.freq),
                "--sample-rate",
                str(args.sample_rate),
                "--bandwidth",
                str(args.bandwidth),
                "--channel",
                str(args.channel),
                "--other-settings",
                args.other_settings,
                "--format",
                case.fmt,
                "--channels",
                case.channels,
                "--stream-args",
                case.stream_args,
                "--activate",
                case.activate,
                "--write-call",
                case.write_call,
                "--layout",
                case.layout,
                "--chunk",
                str(case.chunk),
                "--write-timeout-us",
                str(args.write_timeout_us),
                "--child-writes",
                str(args.child_writes),
                "--write-gap-ms",
                str(args.write_gap_ms),
                "--end-burst",
                args.end_burst,
                "--pattern",
                args.pattern,
                "--amplitude",
                str(args.amplitude),
                "--status-polls",
                str(args.status_polls),
                "--status-timeout-us",
                str(args.status_timeout_us),
                "--status-gap-ms",
                str(args.status_gap_ms),
                "--post-activate-sleep-ms",
                str(args.post_activate_sleep_ms),
            ]
            if args.pace:
                cmd.append("--pace")
            if args.gain is not None:
                cmd.extend(["--gain", str(args.gain)])
            if args.allow_xtrx_overall_gain:
                cmd.append("--allow-xtrx-overall-gain")
            if case_set == "tx-gfsk-direct":
                cmd.append("--direct-tx-gfsk")
            if args.xtrx_safe_rate:
                cmd.append("--xtrx-safe-rate")
            if args.xtrx_safe_bw:
                cmd.append("--xtrx-safe-bw")
            if args.status_after_activate:
                cmd.append("--status-after-activate")
            if args.status_before_write:
                cmd.append("--status-before-write")
            if args.status_after_write:
                cmd.append("--status-after-write")
            if args.status_errors_fatal:
                cmd.append("--status-errors-fatal")
            try:
                cp = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.case_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                result = _classify_timeout(exc)
                worst = max(worst, 2)
            else:
                result = _classify_completed(cp)
                if cp.returncode < 0:
                    worst = max(worst, 3)
                elif cp.returncode > 0:
                    worst = max(worst, 1)

            show_output = (
                args.print_child_output == "all"
                or (args.print_child_output == "failures" and result.status != "OK")
            )
            if show_output and result.stdout:
                print(result.stdout.rstrip())
            if show_output and result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)

            status_by_case[case][result.status] += 1
            reasons_by_case[case].update(result.reasons)
            if result.status == "OK_DRIVER_ERR":
                worst = max(worst, 1)
            reason_text = ",".join(result.reasons) if result.reasons else "-"
            if result.status == "HANG":
                print(
                    f"RESULT: HANG after {args.case_timeout_s:.1f}s reasons={reason_text}",
                    flush=True,
                )
            elif result.status.startswith("SIGNAL_"):
                worst = max(worst, 3)
                print(
                    f"RESULT: SIGNAL {result.status.removeprefix('SIGNAL_')} "
                    f"reasons={reason_text}",
                    flush=True,
                )
            elif result.status.startswith("EXIT_"):
                print(
                    f"RESULT: EXIT {result.status.removeprefix('EXIT_')} reasons={reason_text}",
                    flush=True,
                )
            else:
                print(f"RESULT: {result.status} reasons={reason_text}", flush=True)

            if args.cooldown_s > 0 and index < len(trials):
                time.sleep(args.cooldown_s)
            if args.stop_on_ok and result.status == "OK":
                break
            if args.stop_on_wedge and _first_write_wedge(result):
                print(
                    "Stopping after first writeStream hang/abort; later trials may inherit "
                    "a wedged device state.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        worst = max(worst, 130)
        print("\nInterrupted; printing partial summary.", file=sys.stderr, flush=True)
    _print_summary(cases, status_by_case, reasons_by_case)
    _print_diagnosis(cases, status_by_case, reasons_by_case)
    return worst


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-name", default="")
    parser.add_argument("--device", default="driver=xtrx")
    parser.add_argument("--freq", type=float, default=402_500_000.0)
    parser.add_argument("--sample-rate", type=float, default=2_044_800.0)
    parser.add_argument("--bandwidth", type=float, default=800_000.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--gain", type=float, default=None, help="overall TX gain, dB")
    parser.add_argument("--allow-xtrx-overall-gain", action="store_true",
                        help="force XTRX overall setGain despite observed SoapyXTRX aborts")
    parser.add_argument("--other-settings", default="PAD=0")
    parser.add_argument("--format", default="cs16", choices=["cf32", "cs16"])
    parser.add_argument("--channels", default="default", choices=["default", "explicit"])
    parser.add_argument("--stream-args", default="none", choices=["none", "wire-cs16"])
    parser.add_argument("--activate", default="default", choices=["default", "zero", "num-elems"])
    parser.add_argument("--write-call", default="simple", choices=["simple", "flags", "full"])
    parser.add_argument("--layout", default="2col", choices=["cf32", "2col", "flat"])
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--write-timeout-us", type=int, default=100_000)
    parser.add_argument("--child-writes", type=int, default=1,
                        help="number of writeStream calls to perform inside each child trial")
    parser.add_argument("--write-gap-ms", type=float, default=0.0,
                        help="sleep between child writeStream calls")
    parser.add_argument("--end-burst", default="never", choices=["never", "last"],
                        help="send SOAPY_SDR_END_BURST on the last write for flags/full modes")
    parser.add_argument("--pace", action="store_true",
                        help="pace child writes by accepted samples / sample-rate")
    parser.add_argument("--pattern", default="zero", choices=["zero", "dc", "tone", "ramp", "gfsk"],
                        help="sample pattern for the probe buffer")
    parser.add_argument("--amplitude", type=float, default=0.25,
                        help="non-zero pattern amplitude, 0..1")
    parser.add_argument("--xtrx-safe-bw", action="store_true",
                        help="lift XTRX bandwidth requests below 800 kHz, matching tx_gfsk")
    parser.add_argument("--xtrx-safe-rate", action="store_true",
                        help="lift XTRX TX sample rates below SoapyXTRX's 2.1 MHz lower bound")
    parser.add_argument("--post-activate-sleep-ms", type=float, default=0.0,
                        help="sleep after activateStream returns and before status/write probes")
    parser.add_argument("--status-after-activate", action="store_true",
                        help="call readStreamStatus after activateStream")
    parser.add_argument("--status-before-write", action="store_true",
                        help="call readStreamStatus immediately before each writeStream")
    parser.add_argument("--status-after-write", action="store_true",
                        help="call readStreamStatus after each successful writeStream")
    parser.add_argument("--status-polls", type=int, default=1,
                        help="number of readStreamStatus polls at each enabled status point")
    parser.add_argument("--status-timeout-us", type=int, default=0,
                        help="readStreamStatus timeout per poll")
    parser.add_argument("--status-gap-ms", type=float, default=0.0,
                        help="sleep between repeated readStreamStatus polls")
    parser.add_argument("--status-errors-fatal", action="store_true",
                        help="make readStreamStatus exceptions fail the child")
    parser.add_argument("--case-timeout-s", type=float, default=5.0)
    parser.add_argument("--expanded", action="store_true", help="try the larger combination matrix")
    parser.add_argument(
        "--case-set",
        default="default",
        choices=["default", "write-calls", "single", "tx-gfsk", "tx-gfsk-direct"],
        help=(
            "case matrix: default smoke set, focused write-call matrix, one explicit case, "
            "tx_gfsk-shaped control, or direct tx_gfsk code path"
        ),
    )
    parser.add_argument("--include-wire-cs16", action="store_true",
                        help="include WIRE=CS16 stream args in the write-calls case set")
    parser.add_argument("--stability", action="store_true",
                        help="preset for flaky hardware: repeat, shuffle, cooldown, compact logs")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each selected case this many times")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle trial order to expose order-dependent driver state")
    parser.add_argument("--seed", type=int, default=1, help="shuffle seed")
    parser.add_argument("--max-trials", type=int, default=0,
                        help="cap total parent trials after repeat/shuffle (0=no cap)")
    parser.add_argument(
        "--lock-file",
        default=str(Path(tempfile.gettempdir()) / "soapy-tx-probe.lock"),
        help="parent lock file to prevent overlapping probe runs",
    )
    parser.add_argument("--no-lock", action="store_true", help="disable the parent probe lock")
    parser.add_argument("--cooldown-s", type=float, default=0.0,
                        help="sleep between child trials to let the device settle")
    parser.add_argument("--print-child-output", default="all", choices=["all", "failures", "none"],
                        help="how much child stdout/stderr to print before each RESULT")
    parser.add_argument("--only-format", default="", help="filter cases by format: cf32 or cs16")
    parser.add_argument("--only-channels", default="",
                        help="filter cases by channel style: default or explicit")
    parser.add_argument("--only-stream-args", default="",
                        help="filter cases by stream args: none or wire-cs16")
    parser.add_argument("--only-activate", default="",
                        help="filter cases by activate mode: default, zero, or num-elems")
    parser.add_argument("--only-write-call", default="",
                        help="filter cases by write call: simple, flags, or full")
    parser.add_argument("--only-layout", default="",
                        help="filter cases by layout: cf32, 2col, flat")
    parser.add_argument("--stop-on-ok", action="store_true")
    parser.add_argument("--stop-on-wedge", action="store_true",
                        help="stop after a first-write hang/abort to avoid cascading failures")
    parser.add_argument("--direct-tx-gfsk", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

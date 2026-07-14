#!/usr/bin/env python3
"""Native libxtrx TX probe wrapper around the upstream ``test_xtrx`` utility.

This intentionally bypasses SoapyXTRX. Use it when SoapySDR writeStream() aborts or hangs and we
need to know whether the native libxtrx TX path can start DMA and accept TX slices on the same
device/rate/bandwidth/gain settings.

Without ``-O``, the upstream utility transmits from an internally allocated host buffer. This
probe diagnoses native TX startup only; it does not transmit a defined spacecraft command frame.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HOST_FORMATS = {
    "float32": "8",
    "int16": "4",
    "int8": "2",
}

USABLE_STATUSES = {"OK", "OK_DRIVER_ERR", "OK_UNDERRUN"}

FAILURE_MARKERS = (
    ("xtrx-delayed-buffers", "TX DMA Current delayed buffers"),
    ("xtrx-timeout-skip", "TX DMA Current skip due to TO buffers"),
    ("xtrx-dma-timeout", "TX DMA TO"),
    ("xtrx-dma-error", "TX DMA ERROR"),
    ("xtrxll-dmatx-pointer-abort", "xtrxllpciebase_dmatx_get"),
    ("process-abort", "Aborted"),
    ("core-dumped", "core dumped"),
)

GDB_SIGNAL_RE = re.compile(
    r"^(?:Program|Thread\b[^\n]*)\s+"
    r"(?:received signal|terminated with signal)\s+(SIG[A-Z0-9]+)",
    re.MULTILINE,
)
POSIX_SIGNAL_NUMBERS = {
    "SIGILL": 4,
    "SIGABRT": 6,
    "SIGBUS": 7,
    "SIGFPE": 8,
    "SIGKILL": 9,
    "SIGSEGV": 11,
    "SIGPIPE": 13,
    "SIGALRM": 14,
    "SIGTERM": 15,
}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class Case:
    host_format: str
    tx_slice: int
    samples: int
    tx_skip: int
    tx_no_discard: bool

    def name(self) -> str:
        discard = "no-discard" if self.tx_no_discard else "discard"
        return (
            f"host_format={self.host_format} slice={self.tx_slice} samples={self.samples} "
            f"tx_skip={self.tx_skip} discard={discard}"
        )


@dataclass(frozen=True)
class Result:
    status: str
    returncode: int | None
    underruns: int
    reasons: tuple[str, ...]
    stdout: str
    stderr: str


def _decode_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _parse_csv_ints(text: str, *, option: str) -> list[int]:
    values: list[int] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw, 0)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option}: invalid integer {raw!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{option}: values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{option}: provide at least one value")
    return values


def _parse_formats(text: str) -> list[str]:
    items = [item.strip().lower() for item in str(text or "").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--host-formats: provide at least one format")
    if "all" in items:
        return list(HOST_FORMATS)
    bad = [item for item in items if item not in HOST_FORMATS]
    if bad:
        valid = "|".join((*HOST_FORMATS, "all"))
        raise argparse.ArgumentTypeError(f"--host-formats: invalid {bad!r}; use {valid}")
    return list(dict.fromkeys(items))


def _parse_discard_modes(text: str) -> list[bool]:
    items = [item.strip().lower() for item in str(text or "").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--tx-discard-modes: provide at least one mode")
    if "both" in items:
        return [False, True]
    aliases = {
        "discard": False,
        "default": False,
        "no-discard": True,
        "nodiscard": True,
    }
    bad = [item for item in items if item not in aliases]
    if bad:
        raise argparse.ArgumentTypeError(
            "--tx-discard-modes: invalid "
            f"{bad!r}; use discard,no-discard,both"
        )
    return list(dict.fromkeys(aliases[item] for item in items))


def _find_test_xtrx(explicit: str) -> str:
    if explicit:
        return explicit
    found = shutil.which("test_xtrx")
    if found:
        return found
    candidates = (
        "/usr/lib/xtrx/test_xtrx",
        "/usr/local/lib/xtrx/test_xtrx",
        "/usr/lib64/xtrx/test_xtrx",
        "/usr/lib/aarch64-linux-gnu/xtrx/test_xtrx",
        "/usr/lib/arm-linux-gnueabihf/xtrx/test_xtrx",
        "/opt/lib/xtrx/test_xtrx",
    )
    for path in candidates:
        if Path(path).is_file():
            return path
    return "test_xtrx"


def _find_gdb(explicit: str) -> str:
    if explicit:
        return explicit
    found = shutil.which("gdb")
    if found:
        return found
    raise RuntimeError("--gdb-backtrace requires gdb; install it or pass --gdb=/path/to/gdb")


def _underruns(text: str) -> int:
    match = re.search(r"TX STAT Underruns:\s*(\d+)", text)
    if not match:
        return 0
    return int(match.group(1))


def _reasons(text: str) -> tuple[str, ...]:
    found = [tag for tag, marker in FAILURE_MARKERS if marker in text]
    if " ERROR:" in text and not any(tag.startswith("xtrx-") for tag in found):
        found.append("driver-error")
    if "Success!" in text:
        found.append("success")
    return tuple(dict.fromkeys(found))


def _signal_reason(returncode: int) -> str:
    signum = -int(returncode)
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    return f"signal:{name}"


def _gdb_signal(text: str) -> tuple[int | None, str] | None:
    match = GDB_SIGNAL_RE.search(text)
    if match is None:
        return None
    name = match.group(1)
    signum = POSIX_SIGNAL_NUMBERS.get(name)
    if signum is None:
        signum = getattr(signal, name, None)
    return (int(signum) if signum is not None else None, name)


def _classify(cp: subprocess.CompletedProcess[str]) -> Result:
    stdout = cp.stdout or ""
    stderr = cp.stderr or ""
    text = f"{stderr}\n{stdout}"
    reasons = list(_reasons(text))
    underruns = _underruns(text)
    inferior_signal = _gdb_signal(text)
    if inferior_signal is not None:
        signum, name = inferior_signal
        status = f"SIGNAL_{signum}" if signum is not None else f"SIGNAL_{name}"
        reasons.extend((f"gdb-signal:{name}", f"signal:{name}"))
    elif cp.returncode < 0:
        status = f"SIGNAL_{-cp.returncode}"
        reasons.append(_signal_reason(cp.returncode))
    elif cp.returncode > 0:
        status = f"EXIT_{cp.returncode}"
    elif "Success!" not in text:
        status = "OK_NO_SUCCESS_MARKER"
    elif underruns:
        status = "OK_UNDERRUN"
    elif any(reason.startswith("xtrx-") or reason == "driver-error" for reason in reasons):
        status = "OK_DRIVER_ERR"
    else:
        status = "OK"
    return Result(status, cp.returncode, underruns, tuple(dict.fromkeys(reasons)), stdout, stderr)


def _classify_timeout(stdout: str, stderr: str, diagnostics: str = "") -> Result:
    if diagnostics:
        stderr = f"{stderr.rstrip()}\n\n{diagnostics}".lstrip()
    text = f"{stderr}\n{stdout}"
    return Result("HANG", None, _underruns(text), ("timeout", *_reasons(text)), stdout, stderr)


def _read_proc_file(pid: int, name: str) -> str:
    path = Path("/proc") / str(pid) / name
    try:
        text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"
    return text or "<empty>"


def _xtrx_interrupts() -> str:
    path = Path("/proc/interrupts")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"
    found = [line for line in lines if "xtrx" in line.lower()]
    return "\n".join(found) if found else "<no xtrx interrupt lines>"


def _collect_timeout_forensics(pid: int) -> str:
    sections = [f"# Timeout forensics pid={pid}"]
    for name in ("wchan", "syscall", "stack"):
        sections.append(f"## /proc/{pid}/{name}")
        sections.append(_read_proc_file(pid, name))
    sections.append("## /proc/interrupts xtrx")
    sections.append(_xtrx_interrupts())
    return "\n".join(sections)


def _signal_process_group(proc: subprocess.Popen[str], signum: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signum)
        elif signum == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _build_command(args, case: Case) -> list[str]:
    cmd = [
        _find_test_xtrx(args.test_xtrx),
        "-D",
        args.device,
        "-T",
        "-S",
        str(args.sample_rate),
        "-F",
        str(args.freq),
        "-B",
        str(args.bandwidth),
        "-G",
        str(args.tx_gain),
        "-H",
        HOST_FORMATS[case.host_format],
        "-N",
        str(case.samples),
        "-E",
        str(case.tx_slice),
        "-K",
        str(case.tx_skip),
        "-C",
        str(args.cycles),
        "-l",
        str(args.loglevel),
    ]
    if args.log_period > 0:
        cmd.extend(["-L", str(args.log_period)])
    if args.tx_siso:
        cmd.append("-I")
    if case.tx_no_discard:
        cmd.append("-U")
    if args.gdb_backtrace:
        cmd = [
            _find_gdb(args.gdb),
            "--batch",
            "--quiet",
            "-ex",
            "set pagination off",
            "-ex",
            "set confirm off",
            "-ex",
            "set print thread-events off",
            "-ex",
            "run",
            "-ex",
            "echo \\n# GDB thread backtraces\\n",
            "-ex",
            "thread apply all bt full",
            "--args",
            *cmd,
        ]
    if args.rt_priority > 0:
        chrt = shutil.which("chrt")
        if chrt:
            cmd = [chrt, "-f", str(args.rt_priority), *cmd]
        else:
            print("WARNING: --rt-priority requested but chrt was not found", file=sys.stderr)
    return cmd


def _run_case(args, case: Case, cmd: list[str] | None = None) -> Result:
    if cmd is None:
        cmd = _build_command(args, case)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        diagnostics = _collect_timeout_forensics(proc.pid) if args.timeout_forensics else ""
        _signal_process_group(proc, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout_kill_grace_s)
        except subprocess.TimeoutExpired:
            _signal_process_group(proc, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            if diagnostics:
                diagnostics = (
                    f"{diagnostics}\n## timeout cleanup\n"
                    f"SIGTERM did not exit within {args.timeout_kill_grace_s:.3g}s; sent SIGKILL."
                )
        stdout = _decode_output(stdout)
        stderr = _decode_output(stderr)
        return _classify_timeout(stdout, stderr, diagnostics)
    cp = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    return _classify(cp)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-xtrx", default="", help="path to test_xtrx; auto-search if omitted")
    parser.add_argument("--device", default="pcie:///dev/xtrx0")
    parser.add_argument("--freq", type=float, default=402_500_000.0)
    parser.add_argument("--sample-rate", type=float, default=480_000.0)
    parser.add_argument("--bandwidth", type=float, default=800_000.0)
    parser.add_argument("--tx-gain", type=float, default=0.0, help="TX PAD gain passed to -G")
    parser.add_argument("--host-formats", type=_parse_formats, default=["float32"],
                        help="comma list: float32,int16,int8,all")
    parser.add_argument("--slices", default="1024",
                        help="comma list of test_xtrx -E TX slice sizes")
    parser.add_argument("--samples", default="8192",
                        help="comma list of test_xtrx -N sample counts")
    parser.add_argument("--tx-skips", default="8192",
                        help="comma list of test_xtrx -K TX start-skip sample counts")
    parser.add_argument("--tx-discard-modes", type=_parse_discard_modes, default=[False],
                        help="comma list: discard,no-discard,both; no-discard passes -U")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--loglevel",
        type=int,
        default=5,
        help=(
            "libxtrxll log level; 5 makes its TX DMA status path read four registers, "
            "matching the upstream Yocto workaround"
        ),
    )
    parser.add_argument("--log-period", type=int, default=0,
                        help="test_xtrx -L log period; useful with large --cycles")
    parser.add_argument("--rt-priority", type=int, default=0,
                        help="run test_xtrx through chrt -f PRIORITY when available")
    parser.add_argument("--gdb", default="", help="path to gdb; auto-search if omitted")
    parser.add_argument(
        "--gdb-backtrace",
        action="store_true",
        help="run test_xtrx under batch gdb and print all thread backtraces on abort",
    )
    parser.add_argument("--warn-non-power-of-two", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="warn for TX slice sizes that are not powers of two")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--timeout-forensics", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="on timeout, capture /proc wait state and xtrx interrupt counters")
    parser.add_argument("--timeout-kill-grace-s", type=float, default=1.0,
                        help="seconds to wait after SIGTERM before SIGKILL on timeout")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle trial order to expose order-dependent device state")
    parser.add_argument("--seed", type=int, default=1, help="shuffle seed")
    parser.add_argument("--cooldown-s", type=float, default=0.25)
    parser.add_argument("--tx-siso", action=argparse.BooleanOptionalAction, default=True,
                        help="pass -I for TX SISO mode")
    parser.add_argument("--print-output", default="failures",
                        choices=["all", "failures", "none"])
    parser.add_argument("--stop-on-ok", action="store_true")
    parser.add_argument("--stop-on-usable", action="store_true",
                        help="stop after OK, OK_DRIVER_ERR, or OK_UNDERRUN")
    return parser


def _ordered_counts(counts: Counter[str]) -> str:
    order = ("OK", "OK_DRIVER_ERR", "OK_UNDERRUN", "SIGNAL_6", "HANG")

    def key(item: tuple[str, int]) -> tuple[int, str]:
        status, _ = item
        try:
            return (order.index(status), status)
        except ValueError:
            return (len(order), status)

    return " ".join(f"{status}={count}" for status, count in sorted(counts.items(), key=key))


def _print_diagnosis(
    cases: list[Case],
    status_by_case: dict[Case, Counter[str]],
    reasons_by_case: dict[Case, Counter[str]],
) -> None:
    total_statuses: Counter[str] = Counter()
    total_reasons: Counter[str] = Counter()
    best_case: Case | None = None
    best_score = (-1, -1, -1)
    for case in cases:
        statuses = status_by_case[case]
        reasons = reasons_by_case[case]
        total_statuses.update(statuses)
        total_reasons.update(reasons)
        clean = statuses.get("OK", 0)
        usable = sum(statuses.get(status, 0) for status in USABLE_STATUSES)
        total = sum(statuses.values())
        score = (usable, clean, -total)
        if score > best_score:
            best_score = score
            best_case = case

    total = sum(total_statuses.values())
    if total <= 0:
        return
    clean = total_statuses.get("OK", 0)
    usable = sum(total_statuses.get(status, 0) for status in USABLE_STATUSES)
    hard_failures = total - usable
    reason_text = ", ".join(
        f"{reason}({count})" for reason, count in total_reasons.most_common()
    ) or "-"

    print("\n# Overall", flush=True)
    print(
        f"trials={total} usable={usable} {_ordered_counts(total_statuses)} "
        f"reasons={reason_text}",
        flush=True,
    )
    if best_case is not None:
        statuses = status_by_case[best_case]
        print(
            f"best usable={sum(statuses.get(status, 0) for status in USABLE_STATUSES)}/"
            f"{sum(statuses.values())} :: {best_case.name()}",
            flush=True,
        )

    if total_reasons.get("xtrxll-dmatx-pointer-abort", 0):
        print("\n# Diagnosis", flush=True)
        print(
            "NATIVE_TX_DMA_POINTER_ABORT: libxtrxll aborted in "
            "xtrxllpciebase_dmatx_get after reading an impossible TX ring-pointer span.",
            flush=True,
        )
        print(
            "Upstream libxtrxll aborts when ((nwr - ncleared) & 0x3f) exceeds its "
            "32-buffer TX ring. Log level 5 selects the four-register read path from the "
            "reported Yocto workaround.",
            flush=True,
        )
    elif hard_failures and usable and clean == 0:
        print("\n# Diagnosis", flush=True)
        print(
            "NATIVE_TX_STARTUP_SEVERELY_UNSTABLE: native libxtrx produced only dirty "
            "successful completions; there were no clean OK trials.",
            flush=True,
        )
        print(
            "Treat this test_xtrx tuple as a failing native-startup stress case, not as a "
            "candidate TX recipe.",
            flush=True,
        )
        print(
            "More sample-rate, bandwidth, host-format, or slice-size tuning is unlikely to be "
            "the whole fix unless one exact case repeats cleanly.",
            flush=True,
        )
    elif hard_failures and usable:
        print("\n# Diagnosis", flush=True)
        print(
            "NATIVE_TX_STARTUP_UNSTABLE: native libxtrx sometimes starts TX DMA and sometimes "
            "aborts or hangs with the same RF settings.",
            flush=True,
        )
        print(
            "Do not treat sample rate, host format, or slice size as a complete fix unless the "
            "same case repeats cleanly.",
            flush=True,
        )
    elif hard_failures == total:
        print("\n# Diagnosis", flush=True)
        print(
            "NATIVE_TX_STARTUP_FAILS: every native TX attempt aborted or hung before a successful "
            "test_xtrx completion.",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    slices = _parse_csv_ints(args.slices, option="--slices")
    sample_counts = _parse_csv_ints(args.samples, option="--samples")
    tx_skips = _parse_csv_ints(args.tx_skips, option="--tx-skips")
    cases = [
        Case(host_format, tx_slice, samples, tx_skip, tx_no_discard)
        for host_format in args.host_formats
        for tx_slice in slices
        for samples in sample_counts
        for tx_skip in tx_skips
        for tx_no_discard in args.tx_discard_modes
    ]
    if args.warn_non_power_of_two:
        non_power = []
        for value in slices:
            if not _is_power_of_two(value):
                non_power.append(f"--slices={value}")
        if non_power:
            print(
                "WARNING: upstream issue reports suggest power-of-two TX slice sizes behave "
                f"better; non-power-of-two values: {', '.join(non_power)}",
                file=sys.stderr,
                flush=True,
            )
    trials = [(repeat_index, case) for repeat_index in range(1, args.repeat + 1) for case in cases]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(trials)

    status_by_case: dict[Case, Counter[str]] = {case: Counter() for case in cases}
    reasons_by_case: dict[Case, Counter[str]] = {case: Counter() for case in cases}
    worst = 0
    for index, (repeat_index, case) in enumerate(trials, start=1):
        print(f"\n[{index}/{len(trials)} repeat={repeat_index}/{args.repeat}] {case.name()}",
              flush=True)
        try:
            cmd = _build_command(args, case)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr, flush=True)
            return 2
        print("CMD:", " ".join(cmd), flush=True)
        result = _run_case(args, case, cmd)
        status_by_case[case][result.status] += 1
        reasons_by_case[case].update(result.reasons)
        if result.status == "HANG":
            worst = max(worst, 2)
        elif result.status != "OK":
            worst = max(worst, 1)

        show_output = (
            args.print_output == "all"
            or (args.print_output == "failures" and result.status != "OK")
        )
        if show_output and result.stdout:
            print(result.stdout.rstrip())
        if show_output and result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)

        reason_text = ",".join(result.reasons) if result.reasons else "-"
        print(
            f"RESULT: {result.status} underruns={result.underruns} reasons={reason_text}",
            flush=True,
        )
        if args.stop_on_ok and result.status == "OK":
            break
        if args.stop_on_usable and result.status in USABLE_STATUSES:
            break
        if args.cooldown_s > 0 and index < len(trials):
            time.sleep(args.cooldown_s)

    print("\n# Summary", flush=True)
    for case in cases:
        statuses = " ".join(f"{status}={count}" for status, count in status_by_case[case].items())
        reasons = ", ".join(
            f"{reason}({count})" for reason, count in reasons_by_case[case].most_common()
        ) or "-"
        print(f"{statuses or 'NO_TRIALS'} reasons={reasons} :: {case.name()}", flush=True)
    _print_diagnosis(cases, status_by_case, reasons_by_case)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Standalone 2-GFSK bench transmitter — the TX counterpart to a satnogs_fsk.py RX run.

Unlike the flowgraph apps (``cubesat_gfsk_ax25_tx.py`` / ``cubesat_gfsk_endurosat_bidir.py``), which
are orchestrator-driven (they connect to control/status/data sockets and wait for a transmit
command), this is a one-shot CLI: it builds ONE frame from a payload file using the SAME proven
``gfsk_ax25`` DSP and keys the SDR TX directly. For bench validation of the uplink (TX gain levels
and on-air bit order are the open bench items) or to loop a signal back into a satnogs_fsk.py RX.

Payload comes from ``--payload-file``. Framings:
  * ``endurosat`` — wraps the file as ONE chip-packet payload (preamble/sync/len/payload/CRC-16);
    the payload is opaque (e.g. an encrypted AirMAC frame) and MUST be <= 128 B.
  * ``ax25``      — the file as an AX.25 UI info field (<= 77 B).
  * ``raw``       — the file bytes AS-IS on the air (MSB-first bits -> GFSK by default, nothing
    added: no preamble/sync/len/CRC, no NRZI/scrambling). For a blob that is ALREADY a complete
    on-air frame (it must carry its own preamble/sync for the receiver to lock). Up to 32 KB.
    If the raw file is a packet train separated by zero-byte pads, pass ``--raw-zero-gap-bytes`` to
    turn long 0x00 runs into zero-amplitude IQ gaps instead of transmitting them as FSK "0" bits.

The 2-GFSK modulator needs an integer samples/symbol, so the SDR rate is snapped to the nearest
multiple of the symbol rate. The default is 2.0448 MHz = 213 samples/symbol at 9600 sym/s,
kept below a 2.048 MHz SDR-rate ceiling.

Examples:
  # Wrap a <=128 B command payload in an EnduroSat chip packet, 5x on the XTRX at 402.5 MHz:
  python tools/tx_gfsk.py --soapy-tx-device="driver=xtrx" --samp-rate=2044800 --tx-freq=402500000 \
      --bw=800000 --gain=30 \
      --framing=endurosat --payload-file=/tmp/uplink.bin --repeat=5 --repeat-gap-ms=200

  # Send a pre-framed raw packet train; 32+ zero bytes are inter-frame silence gaps:
  python tools/tx_gfsk.py --tx-freq=402500000 --samp-rate=2044800 --bw=800000 \
      --framing=raw --payload-file=/tmp/command.bin --raw-zero-gap-bytes=32 \
      --other-settings="PAD=-52"

  # No SDR — just render the IQ to a .cf32 (feed it to iq_analyze / a satnogs RX):
  python tools/tx_gfsk.py --framing=raw --payload-file=/tmp/command.bin --out-file=/tmp/tx.cf32

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
from gfsk_ax25 import ax25, endurosat, endurosat_link, gfsk  # noqa: E402

log = logging.getLogger("tx_gfsk")

DEFAULT_SYMBOL_RATE = 9600.0
DEFAULT_SAMPLE_RATE = 2_044_800.0
RAW_MAX_BYTES = 32_768  # raw cap (~27.3 s @ 9600 baud); bounds the in-memory IQ allocation
_TX_CHUNK = 1024
_TX_MAX_STALLS = 20  # consecutive writeStream timeouts before aborting a burst (bounded, no spin)
_TX_WRITE_TIMEOUT_US = 250_000
_TX_DEBUG_WRITES = 4
_TX_CS16_PEAK = 32767.0
XTRX_LOW_RATE_EXPLICIT_CHANNELS_HZ = 1_000_000.0
XTRX_VERY_LOW_RATE_CF32_HZ = 500_000.0
XTRX_YOCTO_LOGLEVEL = 5
# CA-FLOW-010: the XTRX ANALOG filter floor (~0.8 MHz). Below it the analog chain
# goes silent (bench-proven on RX; the TX probe applies the same lift). Channel
# selectivity belongs in DSP — the analog BW tracks the sample rate. Matches
# probe_soapy_tx_write.XTRX_MIN_TX_BW_HZ; change them together with HIL evidence.
XTRX_MIN_TX_BW_HZ = 800_000.0


def _configure_xtrx_loglevel(device: str, loglevel: int) -> None:
    if "xtrx" not in str(device).lower():
        return
    os.environ["SOAPY_XTRX_LOGLEVEL"] = str(int(loglevel))
    log.info(
        "tx: SOAPY_XTRX_LOGLEVEL=%d (5 selects the Yocto four-register TX DMA status path)",
        int(loglevel),
    )


def resolve_rate(samp_rate: float, symbol_rate: float) -> float:
    """Snap ``samp_rate`` to the nearest integer multiple of ``symbol_rate`` — the modulator needs
    an integer samples/symbol (``gfsk.modulate`` raises otherwise)."""
    sps = max(1, round(float(samp_rate) / float(symbol_rate)))
    return float(sps) * float(symbol_rate)


def parse_settings(text: str) -> list[tuple[str, float]]:
    """Parse satnogs-style ``--other-settings`` ("NAME=VAL,NAME=VAL") into per-element gain pairs
    (malformed skipped). Applied as ``setGain(TX, ch, NAME, VAL)`` (LMS7002M elements, e.g. PAD)."""
    pairs: list[tuple[str, float]] = []
    for item in (text or "").split(","):
        item = item.strip()
        if "=" not in item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if not name:
            continue
        with contextlib.suppress(ValueError):
            pairs.append((name, float(value)))
    return pairs


def _raw_bits(payload: bytes, *, bitorder: str) -> np.ndarray:
    """Raw bytes -> bits for the raw framing. ``bitorder`` is numpy's ``big``/``little``."""
    order = {"msb": "big", "lsb": "little"}.get(bitorder, bitorder)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder=order)


def _modulate_raw_bytes(
    payload: bytes,
    params: gfsk.GfskParams,
    *,
    bitorder: str,
) -> np.ndarray:
    return gfsk.modulate(_raw_bits(payload, bitorder=bitorder), params)


def _raw_zero_gap_iq(
    payload: bytes,
    params: gfsk.GfskParams,
    *,
    bitorder: str,
    min_gap_bytes: int,
) -> np.ndarray:
    """Render raw bytes, but convert runs of >= ``min_gap_bytes`` zero bytes into IQ silence.

    This is for captures/export files that already contain complete on-air frames separated by
    zero-byte pads. Plain raw mode transmits those pads as full-power FSK "0" symbols; this mode
    preserves their time duration but writes zero-amplitude samples for the gap.
    """
    if min_gap_bytes <= 0:
        return _modulate_raw_bytes(payload, params, bitorder=bitorder)
    sps = int(round(params.sps))
    if abs(params.sps - sps) > 1e-9:
        msg = f"sample_rate/symbol_rate must be integer for raw gaps (got {params.sps})"
        raise ValueError(msg)

    parts: list[np.ndarray] = []
    segment_count = gap_count = gap_bytes = 0
    pos = 0
    n = len(payload)
    scan = 0
    while scan < n:
        if payload[scan] != 0:
            scan += 1
            continue
        end = scan + 1
        while end < n and payload[end] == 0:
            end += 1
        run = end - scan
        if run >= min_gap_bytes:
            if pos < scan:
                parts.append(_modulate_raw_bytes(payload[pos:scan], params, bitorder=bitorder))
                segment_count += 1
            parts.append(np.zeros(run * 8 * sps, dtype=np.complex64))
            gap_count += 1
            gap_bytes += run
            pos = end
        scan = end
    if pos < n:
        parts.append(_modulate_raw_bytes(payload[pos:], params, bitorder=bitorder))
        segment_count += 1

    log.info(
        "raw gaps: %d non-gap segment(s), %d zero gap(s), %d gap byte(s) rendered as silence",
        segment_count, gap_count, gap_bytes,
    )
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.complex64)


def _is_xtrx_device(args) -> bool:
    return "xtrx" in str(getattr(args, "soapy_tx_device", "") or "").lower()


def _resolve_tx_bw(args, used_rate: float) -> float:
    bw = float(args.bw) if args.bw else float(used_rate)
    # CA-FLOW-010: restore the XTRX narrow-BW guard the WIP snapshot dropped —
    # an analog TX filter below the ~0.8 MHz floor silences the chain. Lift it
    # unless the operator explicitly forces narrow (--allow-narrow-bw).
    if (
        _is_xtrx_device(args)
        and bw < XTRX_MIN_TX_BW_HZ
        and not getattr(args, "allow_narrow_bw", False)
    ):
        log.warning(
            "tx: analog TX bandwidth %.0f Hz is below the XTRX analog floor — lifting to "
            "%.0f Hz (channel selectivity belongs in DSP; --allow-narrow-bw to force)",
            bw, XTRX_MIN_TX_BW_HZ,
        )
        return XTRX_MIN_TX_BW_HZ
    return bw


def _resolve_tx_chunk(requested: int, mtu: int) -> int:
    if mtu <= 0:
        return max(1, requested)
    if requested <= 0:
        return min(_TX_CHUNK, mtu)
    if requested > mtu:
        log.warning("tx: requested chunk %d > stream MTU %d; using MTU", requested, mtu)
        return mtu
    return max(1, requested)


def _resolve_tx_stream_channels(args, used_rate: float | None = None) -> str:
    text = str(args.tx_stream_channels or "auto").strip().lower()
    if text == "auto":
        if _is_xtrx_device(args) and used_rate is not None:
            if float(used_rate) <= XTRX_LOW_RATE_EXPLICIT_CHANNELS_HZ:
                log.info("tx: XTRX low-rate auto stream setup uses explicit channel list")
                return "explicit"
            log.info("tx: XTRX auto stream setup omits channel list")
        return "default"
    if text in {"default", "explicit"}:
        return text
    log.warning("tx: invalid --tx-stream-channels=%r; using default", args.tx_stream_channels)
    return "default"


def _resolve_tx_write_call(args) -> str:
    text = str(args.tx_write_call or "auto").strip().lower()
    if text in {"simple", "flags", "full"}:
        return text
    if text == "auto":
        if _is_xtrx_device(args):
            log.info(
                "tx: XTRX auto writeStream call uses simple form; the flags/full forms have "
                "stalled/aborted in bench runs"
            )
            return "simple"
        return "simple"
    log.warning("tx: invalid --tx-write-call=%r; using simple", args.tx_write_call)
    return "simple"


def _resolve_tx_activate_mode(args) -> str:
    text = str(args.tx_activate_elems or "auto").strip().lower()
    if text != "auto":
        return text
    if _is_xtrx_device(args):
        log.info(
            "tx: XTRX Soapy activation uses default numElems; this path is diagnostic only"
        )
        return "0"
    return "0"


def _resolve_tx_format(args, used_rate: float | None = None) -> str:
    text = str(args.tx_format or "auto").strip().lower()
    if text == "auto":
        if _is_xtrx_device(args):
            if used_rate is not None and float(used_rate) <= XTRX_VERY_LOW_RATE_CF32_HZ:
                log.info("tx: XTRX very-low-rate auto TX format uses CF32")
                return "cf32"
            log.info("tx: XTRX auto TX format uses native flat CS16")
            return "cs16"
        return "cf32"
    if text in {"cf32", "cs16"}:
        return text
    log.warning("tx: invalid --tx-format=%r; using cf32", args.tx_format)
    return "cf32"


def _resolve_tx_time_mode(args) -> str:
    text = str(args.tx_time_mode or "none").strip().lower()
    if text in {"none", "hw", "reset"}:
        if (
            text != "none"
            and _is_xtrx_device(args)
            and not getattr(args, "allow_xtrx_timed_tx", False)
        ):
            log.warning(
                "tx: XTRX timed TX (%s) is disabled by default because this SoapyXTRX path "
                "has stalled/aborted in bench runs; using none (pass --allow-xtrx-timed-tx "
                "to force timestamped writes)",
                text,
            )
            return "none"
        return text
    log.warning("tx: invalid --tx-time-mode=%r; using none", args.tx_time_mode)
    return "none"


def _use_overall_tx_gain(args) -> bool:
    if args.gain is None:
        return False
    if _is_xtrx_device(args) and not getattr(args, "allow_xtrx_overall_gain", False):
        log.warning(
            "tx: ignoring --gain for XTRX; SoapyXTRX TX overall gain falls through an unsafe "
            "generic path. Use --other-settings=PAD=<dB> instead, or pass "
            "--allow-xtrx-overall-gain to force the old behavior."
        )
        return False
    return True


def _resolve_activate_elems(
    mode: str,
    *,
    burst_samples: int,
    mtu: int,
    repeat: int,
) -> int:
    text = str(mode or "0").strip().lower()
    if text in {"0", "default", "none", "unspecified"}:
        return 0
    if text == "mtu":
        return max(0, int(mtu))
    if text == "burst":
        return max(0, int(burst_samples) * max(1, int(repeat)))
    try:
        return max(0, int(text))
    except ValueError:
        log.warning("tx: invalid --tx-activate-elems=%r; using 0", mode)
        return 0

def _iq_to_cs16(iq: np.ndarray, *, scale: float) -> np.ndarray:
    samples = np.asarray(iq, dtype=np.complex64)
    gain = max(0.0, min(1.0, float(scale)))
    # SoapySDR's Python buffer pointer helper treats a 2-D int16 array as a buffer with the wrong
    # element shape for SoapyXTRX. Use native interleaved CS16 layout: I0,Q0,I1,Q1,...
    out = np.empty(samples.size * 2, dtype=np.int16)
    out[0::2] = np.rint(np.clip(samples.real * gain, -1.0, 1.0) * _TX_CS16_PEAK).astype(
        np.int16
    )
    out[1::2] = np.rint(np.clip(samples.imag * gain, -1.0, 1.0) * _TX_CS16_PEAK).astype(
        np.int16
    )
    return out


def _get_hardware_time(dev) -> int:  # pragma: no cover (SoapySDR)
    try:
        return int(dev.getHardwareTime(""))
    except TypeError:
        return int(dev.getHardwareTime())


def _set_hardware_time(dev, time_ns: int) -> None:  # pragma: no cover (SoapySDR)
    try:
        dev.setHardwareTime(int(time_ns), "")
    except TypeError:
        dev.setHardwareTime(int(time_ns))


def _gain_range_text(dev, direction: int, ch: int, name: str | None = None) -> str:
    try:
        rng = dev.getGainRange(direction, ch, name) if name else dev.getGainRange(direction, ch)
    except Exception as e:  # noqa: BLE001 - driver capability probing is best-effort
        return f"(unavailable: {e})"
    return str(rng)


def _gain_value_text(dev, direction: int, ch: int, name: str | None = None) -> str:
    try:
        value = dev.getGain(direction, ch, name) if name else dev.getGain(direction, ch)
    except Exception as e:  # noqa: BLE001 - driver capability probing is best-effort
        return f"(unavailable: {e})"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _gain_value_float(dev, direction: int, ch: int, name: str | None = None) -> float | None:
    try:
        value = dev.getGain(direction, ch, name) if name else dev.getGain(direction, ch)
        return float(value)
    except Exception:  # noqa: BLE001 - best-effort driver readback
        return None


def _log_gain_capabilities(dev, direction: int, ch: int) -> None:  # pragma: no cover (SoapySDR)
    try:
        names = list(dev.listGains(direction, ch))
    except Exception as e:  # noqa: BLE001 - optional driver capability
        log.info("tx gains: list unavailable (%s)", e)
        return
    log.info("tx gains: elements=%s overall_range=%s", names, _gain_range_text(dev, direction, ch))
    for name in names:
        log.info(
            "tx gain element: %s range=%s current=%s",
            name,
            _gain_range_text(dev, direction, ch, name),
            _gain_value_text(dev, direction, ch, name),
        )


def _log_gain_readback(
    dev, direction: int, ch: int, settings: list[tuple[str, float]]
) -> None:  # pragma: no cover (SoapySDR)
    elems = ", ".join(
        f"{name}={_gain_value_text(dev, direction, ch, name)}" for name, _ in settings
    ) or "(none)"
    log.info("tx gains after set: overall=%s elements=%s",
             _gain_value_text(dev, direction, ch), elems)


def build_frame_iq(
    payload: bytes,
    *,
    framing: str,
    sample_rate: float,
    symbol_rate: float,
    mod_index: float,
    bt: float,
    dest: str = "CQ",
    src: str = "DSN",
    scramble: bool = True,
    nrzi: bool = True,
    raw_bitorder: str = "big",
    raw_zero_gap_bytes: int = 0,
) -> np.ndarray:
    """Build one frame's baseband 2-GFSK IQ. ``endurosat`` = chip packet (preamble/sync/len/payload/
    CRC-16, payload verbatim & opaque); ``ax25`` = the payload as an AX.25 UI info field; ``raw`` =
    the file bytes AS-IS (MSB-first bits by default → GFSK, no preamble/sync/len/CRC, no NRZI/
    scrambling), for a blob that is already a complete on-air bitstream."""
    if framing == "raw":
        params = gfsk.GfskParams(
            sample_rate_hz=sample_rate, symbol_rate_hz=symbol_rate, mod_index=mod_index, bt=bt
        )
        return _raw_zero_gap_iq(
            payload,
            params,
            bitorder=raw_bitorder,
            min_gap_bytes=max(0, int(raw_zero_gap_bytes)),
        )
    if framing == "endurosat":
        if len(payload) > endurosat_link.MAX_PAYLOAD:
            log.warning("payload %d B > EnduroSat max %d B — truncating",
                        len(payload), endurosat_link.MAX_PAYLOAD)
        return endurosat_link.transmit(
            payload[: endurosat_link.MAX_PAYLOAD], sample_rate,
            symbol_rate_hz=symbol_rate, mod_index=mod_index, bt=bt,
        )
    if framing == "ax25":
        if len(payload) > endurosat.AX25_INFO_MAX_BYTES:
            log.warning("payload %d B > AX.25 info max %d B — truncating",
                        len(payload), endurosat.AX25_INFO_MAX_BYTES)
        body = ax25.encode_ui(dest=dest, src=src, info=payload[: endurosat.AX25_INFO_MAX_BYTES])
        profile = endurosat.LinkProfile(
            scramble=scramble, nrzi=nrzi, mod_index=mod_index, bt=bt, symbol_rate_hz=symbol_rate,
        )
        return endurosat.transmit(body, sample_rate, profile=profile)
    msg = f"unknown framing {framing!r} (want endurosat|ax25|raw)"
    raise ValueError(msg)


def _configure_tx(dev, args, ch: int, used_rate: float) -> None:  # pragma: no cover (SoapySDR)
    from SoapySDR import SOAPY_SDR_TX

    dev.setSampleRate(SOAPY_SDR_TX, ch, used_rate)
    dev.setFrequency(SOAPY_SDR_TX, ch, float(args.tx_freq))
    # Apply the requested analog TX filter bandwidth. A driver without a settable BW ignores it.
    bw = _resolve_tx_bw(args, used_rate)
    with contextlib.suppress(Exception):
        dev.setBandwidth(SOAPY_SDR_TX, ch, bw)
    if args.antenna:
        with contextlib.suppress(Exception):
            dev.setAntenna(SOAPY_SDR_TX, ch, args.antenna)
    with contextlib.suppress(Exception):
        dev.setGainMode(SOAPY_SDR_TX, ch, False)  # manual gain for TX
    _log_gain_capabilities(dev, SOAPY_SDR_TX, ch)
    gain_settings = parse_settings(args.other_settings)
    use_overall_gain = _use_overall_tx_gain(args)
    if use_overall_gain:
        dev.setGain(SOAPY_SDR_TX, ch, float(args.gain))  # overall
    if use_overall_gain and gain_settings:
        log.info("tx: applying named gains after overall gain, so --other-settings wins")
    for name, value in gain_settings:
        current = _gain_value_float(dev, SOAPY_SDR_TX, ch, name)
        if current is not None and abs(current - value) < 1e-9:
            log.info("tx: gain %s already %g; not calling setGain", name, value)
            continue
        with contextlib.suppress(Exception):
            dev.setGain(SOAPY_SDR_TX, ch, name, value)  # per-element
    _log_gain_readback(dev, SOAPY_SDR_TX, ch, gain_settings)
    if args.ppm:
        with contextlib.suppress(Exception):
            dev.setFrequencyCorrection(SOAPY_SDR_TX, ch, float(args.ppm))
    log.info(
        "tx: dev=%s ch=%d freq=%.0f rate=%.0f bw=%.0f antenna=%s gain=%s other=%s",
        args.soapy_tx_device, ch, args.tx_freq, used_rate, bw, args.antenna or "(default)",
        args.gain if args.gain is not None else "(default)", args.other_settings or "(none)",
    )


def _write_stream_call(
    dev,
    stream,
    block: np.ndarray,
    num_elems: int,
    *,
    flags: int,
    time_ns: int,
    timeout_us: int,
    write_call: str,
):
    """Centralize the Python binding call shape so XTRX tests can isolate wrapper issues."""
    mode = str(write_call or "simple").lower()
    if mode == "simple":
        return dev.writeStream(stream, [block], num_elems)
    if mode == "flags":
        return dev.writeStream(stream, [block], num_elems, flags)
    if mode == "full":
        return dev.writeStream(stream, [block], num_elems, flags, int(time_ns), timeout_us)
    msg = f"unknown tx write call mode {write_call!r}"
    raise ValueError(msg)


def _write_burst(
    dev,
    stream,
    buf: np.ndarray,
    chunk: int,
    *,
    timeout_us: int,
    write_call: str,
    copy_chunks: bool,
    pace_sample_rate: float,
    write_sleep_us: int,
    first_time_ns: int | None,
    elem_stride: int = 1,
) -> int:  # pragma: no cover (SoapySDR)
    """Write one buffer, bounded (never spins on a stalled/erroring stream). END_BURST goes on the
    LAST DATA chunk so the driver flushes the tail and transmits — a separate 0-length END_BURST
    write BLOCKS on XTRX/LMS drivers (do not use it). ``chunk`` is the per-writeStream size (the
    stream MTU): writing MORE than the driver's packet size can segfault the driver. Returns samples
    accepted."""
    from SoapySDR import SOAPY_SDR_END_BURST, SOAPY_SDR_HAS_TIME, SOAPY_SDR_TIMEOUT

    stride = max(1, int(elem_stride))
    n = len(buf) // stride
    i = stalls = writes = 0
    burst_t0 = last_progress = time.monotonic()
    while i < n:
        num_elems = min(chunk, n - i)
        block = buf[i * stride : (i + num_elems) * stride]
        if copy_chunks:
            block = block.copy()
        flags = SOAPY_SDR_END_BURST if (i + num_elems) >= n else 0
        time_ns = 0
        if i == 0 and first_time_ns is not None:
            flags |= SOAPY_SDR_HAS_TIME
            time_ns = int(first_time_ns)
        writes += 1
        call_t0 = time.monotonic()
        if writes <= _TX_DEBUG_WRITES:
            log.info(
                "tx: writeStream enter #%d call=%s offset=%d len=%d flags=%d "
                "time_ns=%s timeout_us=%s",
                writes, write_call, i, num_elems, flags,
                time_ns if first_time_ns is not None and i == 0 else "(none)",
                timeout_us if write_call == "full" else "(binding default)",
            )
        sr = _write_stream_call(
            dev, stream, block, num_elems, flags=flags, time_ns=time_ns,
            timeout_us=timeout_us, write_call=write_call,
        )
        call_dt_ms = 1e3 * (time.monotonic() - call_t0)
        if writes <= _TX_DEBUG_WRITES:
            log.info(
                "tx: writeStream return #%d ret=%s flags=%s time=%.1f ms",
                writes, getattr(sr, "ret", "(unknown)"),
                getattr(sr, "flags", "(unknown)"), call_dt_ms,
            )
        if sr.ret > 0:
            i += sr.ret
            stalls = 0
            if i == sr.ret:
                log.info("tx: first writeStream call returned %d samples", sr.ret)
            now = time.monotonic()
            if pace_sample_rate > 0.0:
                target_elapsed = i / pace_sample_rate
                ahead_s = target_elapsed - (now - burst_t0)
                if ahead_s > 0.0:
                    time.sleep(ahead_s)
                    now = time.monotonic()
            if write_sleep_us > 0:
                time.sleep(write_sleep_us / 1_000_000.0)
                now = time.monotonic()
            if now - last_progress >= 1.0:
                log.info(
                    "tx: write progress %d/%d samples (%.1f%%), writes=%d, elapsed=%.1f s",
                    i, n, 100.0 * i / max(1, n), writes, now - burst_t0,
                )
                last_progress = now
        elif sr.ret == SOAPY_SDR_TIMEOUT:
            stalls += 1
            if stalls > _TX_MAX_STALLS:
                log.warning("tx: writeStream stalled x%d; aborting burst", stalls)
                break
        else:
            log.warning("tx: writeStream error ret=%d; aborting burst", sr.ret)
            break
    return i


def _transmit(
    dev,
    ch: int,
    iq,
    *,
    repeat: int,
    gap_s: float,
    sample_rate: float,
    tx_chunk: int,
    write_call: str,
    write_timeout_us: int,
    copy_chunks: bool,
    pace: bool,
    write_sleep_us: int,
    stream_channels: str,
    activate_elems_mode: str,
    tx_format: str,
    tx_scale: float,
    tx_time_mode: str,
    tx_time_lead_ms: float,
) -> None:  # pragma: no cover
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX

    fmt = str(tx_format or "cf32").lower()
    if fmt == "cs16":
        stream_format = getattr(SoapySDR, "SOAPY_SDR_CS16", None)
        if stream_format is None:
            raise RuntimeError("SoapySDR does not expose SOAPY_SDR_CS16 on this system")
        buf = _iq_to_cs16(iq, scale=tx_scale)
        elem_stride = 2
    else:
        stream_format = SOAPY_SDR_CF32
        buf = np.asarray(iq, dtype=np.complex64)
        elem_stride = 1
    stream_elems = len(buf) // elem_stride

    log.info(
        "tx: setupStream start format=%s ch=%d channels=%s buffer_dtype=%s shape=%s "
        "stream_elems=%d",
        fmt.upper(), ch, stream_channels, buf.dtype, buf.shape, stream_elems,
    )
    if stream_channels == "explicit":
        stream = dev.setupStream(SOAPY_SDR_TX, stream_format, [ch])
    else:
        stream = dev.setupStream(SOAPY_SDR_TX, stream_format)
    log.info("tx: setupStream returned stream=%r", stream)
    try:
        mtu = 0
        log.info("tx: getStreamMTU start")
        with contextlib.suppress(Exception):
            mtu = int(dev.getStreamMTU(stream))
        if mtu <= 0:
            msg = (
                f"tx stream MTU reported {mtu} (invalid). Refusing to write samples because "
                "XTRX/LMS drivers can stall or crash from this state; use a wider TX bandwidth "
                "and a stable sample rate, then retry."
            )
            raise RuntimeError(msg)
        chunk = _resolve_tx_chunk(tx_chunk, mtu)
        activate_elems = _resolve_activate_elems(
            activate_elems_mode,
            burst_samples=stream_elems,
            mtu=mtu,
            repeat=repeat,
        )
        lead_ns = int(max(0.0, float(tx_time_lead_ms)) * 1_000_000.0)
        first_time_ns = None
        if tx_time_mode != "none":
            if tx_time_mode == "reset":
                log.info("tx: setHardwareTime(0) start")
                _set_hardware_time(dev, 0)
                log.info("tx: setHardwareTime(0) returned")
            if write_call != "full":
                log.warning(
                    "tx: timed TX requires full writeStream call; overriding %s", write_call
                )
                write_call = "full"
        log.info(
            "tx: stream MTU=%d write_chunk=%d write_call=%s write_timeout_us=%d "
            "copy_chunks=%s pace=%s write_sleep_us=%d activate_elems=%d tx_time_mode=%s "
            "tx_time_lead_ms=%.1f",
            mtu, chunk, write_call, write_timeout_us, copy_chunks, pace, write_sleep_us,
            activate_elems, tx_time_mode, tx_time_lead_ms,
        )
        log.info("tx: activateStream start")
        if activate_elems > 0:
            dev.activateStream(stream, 0, 0, activate_elems)
        else:
            dev.activateStream(stream)
        log.info("tx: activateStream returned")
        if tx_time_mode != "none":
            hw_now = _get_hardware_time(dev)
            first_time_ns = hw_now + lead_ns
            log.info(
                "tx: timed first write mode=%s hw_now=%d lead_ns=%d first_time_ns=%d",
                tx_time_mode, hw_now, lead_ns, first_time_ns,
            )
        for r in range(repeat):
            log.info("tx: burst %d/%d write start (%d samples)", r + 1, repeat, stream_elems)
            burst_t0 = time.monotonic()
            sent = _write_burst(
                dev, stream, buf, chunk,
                timeout_us=max(1, int(write_timeout_us)),
                write_call=write_call,
                copy_chunks=copy_chunks,
                pace_sample_rate=sample_rate if pace else 0.0,
                write_sleep_us=max(0, int(write_sleep_us)),
                first_time_ns=first_time_ns if r == 0 else None,
                elem_stride=elem_stride,
            )
            log.info(
                "tx: burst %d/%d write returned %d/%d samples in %.1f s",
                r + 1, repeat, sent, stream_elems, time.monotonic() - burst_t0,
            )
            if sent != stream_elems:
                raise RuntimeError(
                    f"TX accepted only {sent}/{stream_elems} samples; refusing clean completion"
                )
            log.info("tx: readStreamStatus start")
            with contextlib.suppress(Exception):
                dev.readStreamStatus(stream, timeoutUs=200_000)
            log.info("tx: readStreamStatus returned")
            log.info("tx: burst %d/%d - %d/%d samples", r + 1, repeat, sent, stream_elems)
            if gap_s > 0.0 and r < repeat - 1:
                time.sleep(gap_s)
    finally:
        log.info("tx: deactivateStream start")
        with contextlib.suppress(Exception):
            dev.deactivateStream(stream)
        log.info("tx: deactivateStream returned")
        log.info("tx: closeStream start")
        with contextlib.suppress(Exception):
            dev.closeStream(stream)
        log.info("tx: closeStream returned")


def _release_soapy_device(soapy, dev) -> bool:
    """Best-effort explicit SoapySDR device release after all TX streams are closed."""
    def _disown() -> None:
        with contextlib.suppress(Exception):
            dev.thisown = False

    close = getattr(dev, "close", None)
    if callable(close):
        try:
            close()
        except Exception as e:  # noqa: BLE001 - teardown should not mask TX result
            log.warning("tx: device close() failed: %s", e)
        else:
            log.info("tx: device closed via device.close()")
            return True

    device_cls = getattr(soapy, "Device", None)
    unmake = getattr(device_cls, "unmake", None)
    if callable(unmake):
        try:
            unmake(dev)
        except Exception as e:  # noqa: BLE001 - driver/binding-dependent cleanup API
            log.warning("tx: SoapySDR.Device.unmake() failed: %s", e)
        else:
            _disown()
            log.info("tx: device released via SoapySDR.Device.unmake()")
            return True

    unmake = getattr(soapy, "Device_unmake", None)
    if callable(unmake):
        try:
            unmake(dev)
        except Exception as e:  # noqa: BLE001 - older SWIG bindings may expose this form
            log.warning("tx: SoapySDR.Device_unmake() failed: %s", e)
        else:
            _disown()
            log.info("tx: device released via SoapySDR.Device_unmake()")
            return True

    log.info("tx: no explicit Soapy device close API; releasing Python reference")
    return False


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tx_gfsk", description="Standalone 2-GFSK bench transmitter.")
    p.add_argument("--payload-file", required=True, help="raw payload bytes to transmit")
    p.add_argument("--framing", default="endurosat", choices=["endurosat", "ax25", "raw"])
    p.add_argument("--soapy-tx-device", default="driver=xtrx")
    p.add_argument(
        "--xtrx-loglevel",
        type=int,
        choices=range(8),
        default=XTRX_YOCTO_LOGLEVEL,
        help=(
            "SOAPY_XTRX_LOGLEVEL used before opening XTRX; 5 selects the validated "
            "Yocto four-register TX DMA status path"
        ),
    )
    p.add_argument("--samp-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--tx-freq", type=float, default=0.0, help="TX centre freq, Hz (needed to key)")
    p.add_argument("--bw", type=float, default=0.0, help="analog TX bandwidth, Hz (0=samp-rate)")
    p.add_argument("--allow-narrow-bw", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--antenna", default="")
    p.add_argument("--gain", type=float, default=None, help="overall TX gain, dB")
    p.add_argument("--gain-mode", default="", help="accepted for CLI parity; TX uses manual gain")
    p.add_argument(
        "--other-settings",
        default="",
        help='per-element gains, e.g. "PAD=-52" for minimum XTRX drive',
    )
    p.add_argument("--symbol-rate", type=float, default=DEFAULT_SYMBOL_RATE)
    p.add_argument("--mod-index", type=float, default=0.5)
    p.add_argument("--bt", type=float, default=0.5)
    p.add_argument("--raw-bitorder", default="msb", choices=["msb", "lsb"],
                   help="raw framing: bit order inside each byte (default: msb)")
    p.add_argument("--raw-zero-gap-bytes", type=int, default=0,
                   help="raw framing: render zero-byte runs of at least N bytes as IQ silence")
    p.add_argument("--dest", default="CQ", help="AX.25 destination callsign")
    p.add_argument("--src", default="DSN", help="AX.25 source callsign")
    p.add_argument("--no-scramble", action="store_true", help="AX.25: disable G3RUH scrambling")
    p.add_argument("--no-nrzi", action="store_true", help="AX.25: disable NRZI")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--repeat-gap-ms", type=float, default=0.0)
    p.add_argument("--tx-format", default="auto", choices=["auto", "cf32", "cs16"],
                   help="Soapy host sample format")
    p.add_argument("--tx-scale", type=float, default=1.0,
                   help="digital IQ scale before transmission, clamped to 0..1")
    p.add_argument("--tx-chunk", type=int, default=0,
                   help=f"Soapy writeStream length (0=default {_TX_CHUNK}, capped to MTU)")
    p.add_argument("--tx-write-call", default="auto", choices=["auto", "simple", "flags", "full"],
                   help="Python writeStream call shape: auto, simple, flags, or full timeout form")
    p.add_argument("--tx-stream-channels", default="auto", choices=["auto", "default", "explicit"],
                   help="setupStream channel style: auto, default omits list, explicit passes [ch]")
    p.add_argument("--tx-activate-elems", default="auto",
                   help="Soapy activateStream numElems hint: auto, 0, mtu, burst, or integer")
    p.add_argument("--tx-write-timeout-us", type=int, default=_TX_WRITE_TIMEOUT_US,
                   help="writeStream timeout per chunk, microseconds (only passed in full mode)")
    p.add_argument("--tx-copy-chunks", action="store_true",
                   help="copy each write chunk before passing it to SoapySDR")
    p.add_argument("--tx-pace", action="store_true",
                   help="pace writeStream calls to the configured TX sample rate")
    p.add_argument("--tx-write-sleep-us", type=int, default=0,
                   help="extra sleep after each accepted writeStream call")
    p.add_argument("--tx-time-mode", default="none", choices=["none", "hw", "reset"],
                   help="timestamp first TX write: none, hw=current time, reset=set time 0")
    p.add_argument("--tx-time-lead-ms", type=float, default=50.0,
                   help="future lead time for timed first TX write")
    p.add_argument("--allow-xtrx-timed-tx", action="store_true",
                   help="force XTRX timestamped TX writes despite observed SoapyXTRX aborts")
    p.add_argument("--allow-xtrx-overall-gain", action="store_true",
                   help="force XTRX overall setGain despite observed SoapyXTRX aborts")
    p.add_argument("--ppm", type=float, default=0.0)
    p.add_argument("--allow-truncate", action="store_true",
                   help="send only the first max-payload bytes if oversize (default: refuse)")
    p.add_argument("--out-file", default="", help="write IQ to this .cf32 instead of the SDR")
    return p


def _max_payload(framing: str) -> int:
    if framing == "endurosat":
        return endurosat_link.MAX_PAYLOAD
    if framing == "ax25":
        return endurosat.AX25_INFO_MAX_BYTES
    return RAW_MAX_BYTES  # raw


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    with contextlib.suppress(Exception):
        faulthandler.enable(all_threads=True)
    args = _build_argparser().parse_args(argv)

    payload_path = Path(args.payload_file)
    if not payload_path.is_file():
        log.error("payload file not found: %s", payload_path)
        return 2
    payload = payload_path.read_bytes()

    cap = _max_payload(args.framing)
    if len(payload) > cap and not args.allow_truncate:
        extra = (
            " (an EnduroSat frame carries <=128 B; split the data into per-frame payloads — "
            "fragmentation is the AirMAC/session layer's job)"
            if args.framing == "endurosat" else ""
        )
        log.error(
            "payload %d B exceeds the %s max %d B — refusing to send a truncated frame%s. "
            "Pass --allow-truncate to send the first %d B anyway.",
            len(payload), args.framing, cap, extra, cap,
        )
        return 2

    used_rate = resolve_rate(args.samp_rate, args.symbol_rate)
    if used_rate != args.samp_rate:
        log.info("samp-rate %.0f snapped to %.0f for integer %.0f samples/symbol",
                 args.samp_rate, used_rate, used_rate / args.symbol_rate)

    iq = build_frame_iq(
        payload, framing=args.framing, sample_rate=used_rate, symbol_rate=args.symbol_rate,
        mod_index=args.mod_index, bt=args.bt, dest=args.dest, src=args.src,
        scramble=not args.no_scramble, nrzi=not args.no_nrzi,
        raw_bitorder="little" if args.raw_bitorder == "lsb" else "big",
        raw_zero_gap_bytes=args.raw_zero_gap_bytes,
    )
    log.info(
        "built %s frame: payload=%d B, %.0f sym/s @ %.0f Hz → %d samples (%.1f ms)",
        args.framing, len(payload), args.symbol_rate, used_rate, len(iq), 1e3 * len(iq) / used_rate,
    )

    if args.out_file:
        iq.astype(np.complex64).tofile(args.out_file)
        log.info("wrote %d samples to %s (no SDR)", len(iq), args.out_file)
        return 0

    if args.tx_freq <= 0.0:
        log.error("--tx-freq is required to transmit (or use --out-file to render IQ only)")
        return 2

    try:
        import SoapySDR  # noqa: PLC0415 — bench-only; not needed for --out-file
    except ImportError as e:
        log.error("SoapySDR not available (%s); use --out-file to render IQ without a radio", e)
        return 3
    dev = None
    try:
        _configure_xtrx_loglevel(args.soapy_tx_device, args.xtrx_loglevel)
        dev = SoapySDR.Device(args.soapy_tx_device)
        _configure_tx(dev, args, args.channel, used_rate)
        _transmit(
            dev, args.channel, iq,
            repeat=max(1, args.repeat), gap_s=args.repeat_gap_ms / 1000.0,
            sample_rate=used_rate,
            tx_chunk=args.tx_chunk, write_call=_resolve_tx_write_call(args),
            write_timeout_us=args.tx_write_timeout_us,
            copy_chunks=args.tx_copy_chunks, pace=args.tx_pace,
            write_sleep_us=args.tx_write_sleep_us,
            stream_channels=_resolve_tx_stream_channels(args, used_rate),
            activate_elems_mode=_resolve_tx_activate_mode(args),
            tx_format=_resolve_tx_format(args, used_rate), tx_scale=args.tx_scale,
            tx_time_mode=_resolve_tx_time_mode(args), tx_time_lead_ms=args.tx_time_lead_ms,
        )
    except RuntimeError as e:
        log.error("%s", e)
        return 4
    except Exception:
        log.exception("tx failed")
        return 4
    finally:
        if dev is not None:
            _release_soapy_device(SoapySDR, dev)
            dev = None
            gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())

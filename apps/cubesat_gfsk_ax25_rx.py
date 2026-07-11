#!/usr/bin/env python3
"""Cubesat 2-GFSK / AX.25 receive flowgraph (EnduroSat-class UHF, 9k6).

Decodes a 2-GFSK, G3RUH-scrambled, AX.25-framed downlink (beacon + packets) and
emits one ``frame_received`` status event per valid frame plus the raw frame
bytes on the data socket (``data_format = "raw_bytes"`` -> gs-client's RAW_BITS
artifact spec, an explicit key in its ``spec_for_data_format`` map).

Two interchangeable engines, selected by ``--engine`` / ``GS_FLOWGRAPH_ENGINE``
env / params-file ``engine`` key (default ``gnuradio``):

* ``gnuradio`` — GNU Radio front-end (IQ -> bits) using the gr-soapy source. The
                 default + production hardware path (SatNOGS-proven: gr-soapy owns
                 stream activation/MTU/timeouts). Hands the bitstream to the SAME
                 tested ``framing.decode`` protocol layer.
* ``dsp``      — pure numpy/scipy demod (``gfsk_ax25`` library). Backup engine:
                 runs anywhere, fully unit-tested, used for file-IQ bench work and
                 as a fallback. IQ from SoapySDR (``--sdr-args driver=...``) via a
                 hand-rolled read loop, or a ``cf32`` file (``--sdr-args
                 file:/path.cf32``).

Both engines share the scrambler/NRZI/HDLC/AX.25 code, so only the IQ->bits
front-end differs. The ``gnuradio`` path is validated on the Linux bench (it
imports ``gnuradio``); the ``dsp`` path is exercised by the repo's pytest suite.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import threading
import time

import numpy as np
from _fallback_select import symbol_rate_hz_of
from _recorder import StreamRecorder, first_sample_probe
from _soapy import (
    capture_plan,
    merge_sdr_params,
    readback_soapy_settings,
    resample_ratio,
    sdr_env,
    sdr_ready_fields,
)
from _spawn_contract import (
    EngineFailure,
    await_first_samples,
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
    watch_engine_death,
)
from framings import LIVE_FRAMINGS as _LIVE_FRAMINGS
from framings import normalize_framing, valid_ax25_address

from gfsk_ax25 import ax25, endurosat, endurosat_link

VERSION = "0.1.0"

_DEFAULT_SAMPLE_RATE = 96_000  # >= 5x the 18.7 kHz channel; integer-ish sps
_DECODE_PERIOD_S = 2.0  # how often the dsp engine re-decodes the capture
_READ_CHUNK = 4096
_DEFAULT_SDR_GAIN_DB = 40.0  # manual RX gain when none is configured (0 dB = deaf)
# R-11: how long the source gets to deliver its FIRST samples before the pass
# fails closed. SDR open + stream activation is seconds; the supervisor's
# ready timeout is 30 s, so 15 s leaves room for the open itself.
_FIRST_SAMPLE_TIMEOUT_S = 15.0


def _select_engine(args, params: dict[str, object]) -> str:
    explicit = getattr(args, "engine", "") or ""
    env = os.environ.get("GS_FLOWGRAPH_ENGINE", "")
    from_params = str(params.get("engine", "")) if isinstance(params, dict) else ""
    requested = (explicit or env or from_params).lower()
    if requested:
        engine = requested
    elif _select_framing(params) == "endurosat":
        # The proven, lab-validated EnduroSat chip-packet receiver is the IQ-level
        # ensemble in endurosat_link (StreamDecoder), which lives in the dsp engine.
        # Default endurosat passes there rather than the bit-level gnuradio fallback.
        engine = "dsp"
    else:
        # Default to the gr-soapy (gnuradio) front-end: the SatNOGS-proven front-end
        # (gr-soapy handles stream activation/MTU/timeouts); dsp is the backup.
        engine = "gnuradio"
    if engine not in ("dsp", "gnuradio"):
        logging.getLogger("cubesat_gfsk_ax25_rx").warning(
            "unknown engine %r; falling back to gnuradio", engine
        )
        engine = "gnuradio"
    return engine


# Link layers THIS app can deframe. Anything else runs record-only (IQ capture,
# no deframe) — never silently a wrong link layer.
_APP_FRAMINGS = ("ax25", "endurosat")
# ``_LIVE_FRAMINGS`` (imported from the framings registry — single-sourced with the post-pass
# decoder) is the LIGHT set this app runs LIVE, both at once (docs/13). The backend/SatYAML
# `framing` label is a HINT, not an exclusive filter: a pass labelled "ax25" whose real traffic is
# EnduroSat (or vice-versa) is captured in ONE pass. Both are CRC-gated, so the wrong deframer
# yields nothing (no false frames); the only cost is a second light 2-GFSK demod. Heavier /
# non-light framings (ccsds_tm, kiss, the gr-satellites catalog) are handled post-pass on the
# recorded .cf32 — see apps/iq_decode.py.
_warned_framings: set[str] = set()  # one WARNING per unknown label per process


def _build_live_decoder(
    framing: str,
    sample_rate: float,
    params: dict[str, object],
    profile: endurosat.LinkProfile,
) -> object:
    """One IQ ``StreamDecoder`` for a live framing. Both light framings share the pass's 2-GFSK
    PHY and differ only in the link layer (HDLC/AX.25 vs EnduroSat chip-packet) + its bit-level
    descrambling/NRZI, so running both on the same IQ costs a second light demod, nothing more."""
    if framing == "endurosat":
        sym_hz = symbol_rate_hz_of(params, default=endurosat_link.DEFAULT_SYMBOL_RATE_HZ)
        return endurosat_link.StreamDecoder(
            sample_rate,
            symbol_rate_hz=sym_hz,
            mod_index=float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
            bt=float(params.get("bt", endurosat_link.DEFAULT_BT)),
        )
    return endurosat.StreamDecoder(sample_rate, profile=profile, recover_timing=True)


def _select_framing(params: dict[str, object]) -> str | None:
    """``"ax25"`` (default, compat) | ``"endurosat"`` (chip-packet, the real
    Gen-2 link) | ``None`` (record-only: IQ capture continues, no deframe).

    The label — GS_FLOWGRAPH_FRAMING env or the params ``framing`` key — arrives
    VERBATIM from gs-client (backend/SatYAML vocabulary, e.g. "AX.25 G3RUH",
    "EnduroSat AirMAC"). It is routed through ``framings.normalize_framing``,
    the system's SINGLE normalization point (docs/10 P0-2) — this app keeps no
    second exact-token vocabulary. A missing/blank label keeps the app's
    historical AX.25 default; a label that doesn't normalize to a framing this
    app implements must NOT silently become ax25 (a wrong link layer): the pass
    runs record-only, with one WARNING. EnduroSat payloads are the opaque
    (encrypted) AirMAC frames — the orchestrator handles those.
    """
    label = str(
        os.environ.get("GS_FLOWGRAPH_FRAMING", "")
        or (params.get("framing", "") if isinstance(params, dict) else "")
    ).strip()
    if not label:
        return "ax25"
    local = normalize_framing(label)
    if local in _APP_FRAMINGS:
        return local
    if label not in _warned_framings:
        _warned_framings.add(label)
        logging.getLogger("cubesat_gfsk_ax25_rx").warning(
            "framing %r is not one this app deframes (%s); recording IQ without deframing",
            label,
            "/".join(_APP_FRAMINGS),
        )
    return None


def _profile_from_params(params: dict[str, object]) -> endurosat.LinkProfile:
    return endurosat.LinkProfile(
        scramble=bool(params.get("scramble", True)),
        nrzi=bool(params.get("nrzi", True)),
        mod_index=float(params.get("mod_index", endurosat.MOD_INDEX)),  # type: ignore[arg-type]
        bt=float(params.get("bt", endurosat.BT)),  # type: ignore[arg-type]
        # baud/baudrate/symbol_rate_hz are interchangeable (all the symbol rate).
        symbol_rate_hz=symbol_rate_hz_of(params, default=endurosat.SYMBOL_RATE_HZ),
    )


def _append_frame_record(output_dir: str, body: bytes, framing: str, ui=None) -> None:
    """Append one decoded frame to ``<output_dir>/frames.jsonl`` — the deframed payload
    plus (for endurosat) the full on-wire frame. Written alongside the IQ files (same
    pass dir), so gs-client uploads it to object storage when configured, else the
    IQ-retention reaper cleans it. Never raises."""
    rec: dict[str, object] = {
        "ts": round(time.time(), 3),
        "framing": framing,
        "len": len(body),
        "crc_ok": True,
        "payload_hex": body.hex(),  # deframed link payload (the AirMAC frame for endurosat)
    }
    if framing == "endurosat":
        with contextlib.suppress(Exception):
            rec["frame_hex"] = endurosat_link.frame_bytes(body).hex()  # full on-wire frame
    if ui is not None:
        rec.update({"dest": ui.dest, "src": ui.src, "info_hex": ui.info.hex()})
    try:
        with open(os.path.join(output_dir, "frames.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        logging.getLogger("cubesat_gfsk_ax25_rx").warning("frames.jsonl write failed: %s", e)


def _endurosat_deframe_bits(bits) -> list[bytes]:  # pragma: no cover (bench)
    """EnduroSat chip-packet deframe over GR-recovered bits, trying both polarities
    (the bit-level fallback; the dsp engine's IQ-level StreamDecoder is preferred)."""
    arr = np.asarray(bits, dtype=np.uint8)
    return endurosat_link.deframe(arr) or endurosat_link.deframe(1 - arr)


async def _emit_frame(sockets, body: bytes, *, framing: str = "ax25", output_dir=None) -> None:
    """Send a frame_received status event + raw frame bytes on the data socket, and
    (when ``output_dir`` is given) append a {raw, deframed} record to frames.jsonl.

    For ``endurosat`` framing the body is the opaque (encrypted AirMAC) payload,
    so we do not attempt an AX.25 parse — the orchestrator decodes it.
    """
    # Run-both means the AX.25 deframer runs on EVERY pass (incl. EnduroSat/unknown-labelled), and
    # its 16-bit FCS passes a random noise chunk ~1/65536 of the time. The registry deframe rejects
    # those by callsign structure; the live StreamDecoder/framing.decode path does not, so apply the
    # SAME guard here before emitting — else a spurious "ax25" frame pollutes the product.
    if framing == "ax25" and not valid_ax25_address(body):
        return
    ui = ax25.decode_ui(body) if framing == "ax25" else None
    event: dict[str, object] = {
        "event": "frame_received",
        "framing": framing,
        "frame": {
            "bytes_b64": base64.b64encode(body).decode("ascii"),
            "len": len(body),
            "crc_ok": True,
        },
    }
    if ui is not None:
        event["frame"].update({"dest": ui.dest, "src": ui.src, "info_len": len(ui.info)})  # type: ignore[union-attr]
    await send_event(sockets.status_writer, event)
    try:
        sockets.data_writer.write(body)
        await sockets.data_writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        logging.getLogger("cubesat_gfsk_ax25_rx").warning("data socket closed; frame not stored")
    if output_dir:  # persist {raw, deframed} alongside the IQ (S3/temp handled by gs-client)
        _append_frame_record(output_dir, body, framing, ui)


# ----------------------------------------------------------------------
# dsp engine: SoapySDR / file IQ source -> numpy demod
# ----------------------------------------------------------------------


def _open_iq_source(args, params=None, report=None):
    """Return a blocking iterator of complex64 chunks (SoapySDR or cf32 file).
    ``report`` (optional dict) is filled with the R-21 identity/settings record
    during source setup — device, requested vs applied vs read-back — so the
    ready event can carry what the front-end ACTUALLY runs."""
    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        path = sdr_args[len("file:") :]
        if report is not None:
            report["device"] = sdr_args
            report["source"] = "file"
        return _file_iq_chunks(path)
    return _soapy_iq_chunks(args, params or {}, report)  # pragma: no cover (needs hardware)


def _file_iq_chunks(path: str):
    with open(path, "rb") as f:
        while True:
            raw = f.read(_READ_CHUNK * 8)  # complex64 = 8 bytes
            if not raw:
                return
            yield np.frombuffer(raw, dtype=np.complex64)


def _sdr_channel(args) -> int:
    """RX channel index from ``--sdr-port`` when it is a plain integer; else 0.
    A non-numeric port (e.g. ``RX1``) is treated as an antenna name, not a channel."""
    port = str(getattr(args, "sdr_port", "") or "").strip()
    return int(port) if port.isdigit() else 0


def _select_rx_antenna(dev, soapy_rx, ch, args, params, log) -> None:  # pragma: no cover
    """Select + set the RX antenna. Priority: ``params['sdr_antenna']`` → the
    ``--sdr-port`` label if the driver lists it → the first non-``NONE`` antenna.
    The available list + the chosen one are logged so a wrong physical port (the
    classic "0-byte capture / deaf radio") is obvious from the journal."""
    available = [str(a) for a in dev.listAntennas(soapy_rx, ch)]
    port = str(getattr(args, "sdr_port", "") or "").strip()
    chosen = None
    for cand in (params.get("sdr_antenna"), port):
        if isinstance(cand, str) and cand in available:
            chosen = cand
            break
    if chosen is None and available:
        chosen = next((a for a in available if a.upper() != "NONE"), available[0])
    if chosen is not None:
        dev.setAntenna(soapy_rx, ch, chosen)
    log.info("dsp SDR: RX antenna=%s (available=%s, requested port=%r)", chosen, available, port)


def _resample_poly(x, up: int, down: int):  # pragma: no cover (needs scipy + hardware)
    """Polyphase resample a complex64 chunk by ``up/down`` (capture rate → modem rate).
    scipy handles the anti-alias filter; per-chunk boundary transients are negligible at
    the large capture/modem ratio and the modem's sync tolerates them."""
    from scipy.signal import resample_poly  # noqa: PLC0415

    return resample_poly(x, up, down).astype(np.complex64)


def _soapy_iq_chunks(args, params, report=None):  # pragma: no cover (needs hardware/SoapySDR)
    import SoapySDR  # lazy: only the dsp+hardware path needs it
    from SoapySDR import (
        SOAPY_SDR_CF32,
        SOAPY_SDR_OVERFLOW,
        SOAPY_SDR_RX,
        SOAPY_SDR_TIMEOUT,
    )

    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    ch = _sdr_channel(args)
    rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    freq = float(args.center_freq_hz)

    params = merge_sdr_params(params)  # station GS_SDR_* antenna/gain/agc defaults
    env = sdr_env()
    lo = env["lo_offset_hz"]
    # Capture at the SDR's supported rate (XTRX floor ~2.1 Msps) and resample each
    # chunk down to ``rate`` for the modem (the file-IQ path is already at ``rate``).
    sdr_rate, decimate = capture_plan(env["capture_rate_hz"], rate)
    up, down = resample_ratio(sdr_rate, rate) if decimate else (1, 1)

    dev = SoapySDR.Device(args.sdr_args)
    dev.setSampleRate(SOAPY_SDR_RX, ch, sdr_rate)
    # Widen the ANALOG RX filter to ~the capture rate so a large LO offset (the +lo carrier below)
    # isn't rolled off before the ADC — the XTRX analog floor is ~0.8 MHz; channel selectivity is
    # done in DSP. Guarded: a driver without a settable analog BW just ignores it.
    with contextlib.suppress(Exception):  # noqa: BLE001 — driver may lack a settable analog BW
        dev.setBandwidth(SOAPY_SDR_RX, ch, sdr_rate)
    # LO offset: tune the analog LO off-carrier (RF) + the baseband CORDIC back (BB)
    # so the DC spike sits at +lo, not on the signal. The dsp NCO handles Doppler at
    # baseband, so the LO is set once here.
    if lo:
        try:
            dev.setFrequency(SOAPY_SDR_RX, ch, "RF", freq + lo)
            dev.setFrequency(SOAPY_SDR_RX, ch, "BB", -lo)
        except Exception:  # noqa: BLE001 — driver without RF/BB split → direct tune
            dev.setFrequency(SOAPY_SDR_RX, ch, freq)
    else:
        dev.setFrequency(SOAPY_SDR_RX, ch, freq)
    _select_rx_antenna(dev, SOAPY_SDR_RX, ch, args, params, log)
    if env["ppm"]:
        with contextlib.suppress(Exception):
            dev.setFrequencyCorrection(SOAPY_SDR_RX, ch, env["ppm"])
    if env["dc_removal"]:
        with contextlib.suppress(Exception):
            dev.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

    # Front-end gain (mirrors the gnuradio engines / configure_soapy_source):
    # GS_SDR_AGC → hardware AGC; else per-element staging (GS_SDR_GAINS, e.g.
    # LNA=30,TIA=9,PGA=3) wins; else overall sdr_gain_db; else AGC off + a sane manual
    # default (a 0 dB front-end is effectively deaf and was never set before).
    gain_db = params.get("sdr_gain_db")
    gains = params.get("sdr_gains")
    if params.get("sdr_agc"):
        with contextlib.suppress(Exception):
            dev.setGainMode(SOAPY_SDR_RX, ch, True)  # hardware AGC
        gain_str = "AGC"
    elif isinstance(gains, dict) and gains:
        with contextlib.suppress(Exception):
            dev.setGainMode(SOAPY_SDR_RX, ch, False)
        for gname, gval in gains.items():
            if not isinstance(gname, str) or isinstance(gval, bool):
                continue
            if isinstance(gval, (int, float)):
                dev.setGain(SOAPY_SDR_RX, ch, gname, float(gval))  # per-element
        gain_str = ",".join(f"{k}={v:g}" for k, v in gains.items())
    elif isinstance(gain_db, (int, float)) and not isinstance(gain_db, bool):
        dev.setGain(SOAPY_SDR_RX, ch, float(gain_db))
        gain_str = f"{float(gain_db):.1f} dB"
    else:
        with contextlib.suppress(Exception):
            dev.setGainMode(SOAPY_SDR_RX, ch, False)  # disable AGC
        dev.setGain(SOAPY_SDR_RX, ch, float(_DEFAULT_SDR_GAIN_DB))
        gain_str = f"{_DEFAULT_SDR_GAIN_DB:.1f} dB (default)"
    log.info(
        "dsp SDR: %s ch=%d capture=%.0f→%.0f freq=%.0f lo_offset=%.0f gain=%s ppm=%.2f "
        "dc_removal=%s — opening RX stream",
        args.sdr_args, ch, sdr_rate, rate, freq, lo, gain_str, env["ppm"], env["dc_removal"],
    )
    # R-21: requested vs read-back settings for the ready event. The readback
    # runs AFTER every setter above, so "actual" is what the hardware settled on.
    actual = readback_soapy_settings(dev, channel=ch, direction=SOAPY_SDR_RX)
    log.info("dsp SDR readback: %s", actual)
    if report is not None:
        report.update(
            device=str(args.sdr_args),
            source="soapy",
            requested={
                "sample_rate_hz": sdr_rate, "frequency_hz": freq, "lo_offset_hz": lo,
                "gain": gain_str, "ppm": env["ppm"], "dc_removal": env["dc_removal"],
            },
            actual=actual,
        )

    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
    dev.activateStream(stream)
    buff = np.empty(_READ_CHUNK, dtype=np.complex64)
    total = reads = timeouts = overflows = errors = 0
    try:
        while True:
            # 1 s timeout so a stalled SDR surfaces as logged timeouts, not a silent
            # spin (the old loop swallowed every ret<=0 and recorded nothing).
            sr = dev.readStream(stream, [buff], len(buff), timeoutUs=1_000_000)
            if sr.ret > 0:
                total += sr.ret
                reads += 1
                if reads == 1:
                    log.info("dsp SDR: streaming — first %d samples received", sr.ret)
                elif reads % 5000 == 0:
                    log.info("dsp SDR: %.2f Msamples read so far", total / 1e6)
                if decimate:  # resample capture-rate chunk down to the modem rate
                    yield _resample_poly(buff[: sr.ret], up, down)
                else:
                    yield buff[: sr.ret].copy()
            elif sr.ret == SOAPY_SDR_TIMEOUT:
                timeouts += 1
                if timeouts in (1, 10, 100) or timeouts % 1000 == 0:
                    log.warning(
                        "dsp SDR: readStream TIMEOUT x%d — no samples from %s "
                        "(check antenna/clock/driver)", timeouts, args.sdr_args,
                    )
            elif sr.ret == SOAPY_SDR_OVERFLOW:
                overflows += 1  # 'O' — host not draining fast enough; keep going
            else:
                errors += 1
                if errors in (1, 10) or errors % 1000 == 0:
                    log.warning("dsp SDR: readStream error ret=%d flags=%d", sr.ret, sr.flags)
    finally:
        log.info(
            "dsp SDR: closing stream — total=%.2f Ms reads=%d timeouts=%d overflows=%d errors=%d",
            total / 1e6, reads, timeouts, overflows, errors,
        )
        dev.deactivateStream(stream)
        dev.closeStream(stream)


async def _run_dsp_engine(args, sockets, params, started, stop_requested, profile, doppler) -> None:
    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    out_dir = getattr(args, "output_dir", None)  # frames.jsonl alongside the IQ
    # Run BOTH light framings live (the `framing` label is only a hint now). Each frame is emitted
    # tagged with the framing that produced it; the wrong deframer stays silent (CRC-gated).
    declared = _select_framing(params)  # hint only — logged in the ready event; may be None
    decoders: list[tuple[str, object]] = [
        (name, _build_live_decoder(name, sample_rate, params, profile)) for name in _LIVE_FRAMINGS
    ]

    # Digital Doppler NCO. ``doppler`` is shared with the command handler, so a
    # set_doppler mid-pass is picked up here on the next chunk; nco_phase keeps
    # the correction phase-continuous across chunks.
    nco_phase = 0.0

    # RX: stream + record from spawn (arm — before AOS); don't gate on cmd:start.
    # ``started`` still tracks the cmd:start/stop lifecycle for the status events.
    _ = started

    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_running_loop()
    # Pre-demod IQ capture: record the RAW chunk (before the Doppler NCO / demod).
    recorder = StreamRecorder.maybe_start(args, sample_rate_hz=sample_rate)
    sdr_report: dict[str, object] = {}

    def _reader() -> None:
        try:
            for chunk in _open_iq_source(args, params, sdr_report):
                if stop_requested.is_set():
                    break
                arr = np.asarray(chunk, dtype=np.complex64)
                if recorder is not None:
                    recorder.write(arr)
                loop.call_soon_threadsafe(queue.put_nowait, arr)
        except Exception:
            log.exception("IQ source error")
        finally:
            if recorder is not None:
                recorder.close()  # close the cf32; views derived post-pass by iq_views
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Daemon thread, NOT run_in_executor: a wedged SoapySDR readStream must not
    # block interpreter exit on the fail-closed path (executor threads are
    # joined at exit; a daemon thread is not).
    reader_thread = threading.Thread(target=_reader, name="iq-reader", daemon=True)
    reader_thread.start()

    # R-11: 'ready' is PROOF, not process startup — it goes out only after the
    # source delivered its first samples (device open + settings applied +
    # stream active + data flowing). A source that cannot open or stays silent
    # fails the pass HERE (EngineFailure → the amain death watch emits an
    # ``error`` event and the app exits nonzero), not at LOS with 0 bytes.
    try:
        first_chunk = await asyncio.wait_for(queue.get(), timeout=_FIRST_SAMPLE_TIMEOUT_S)
    except TimeoutError:
        stop_requested.set()
        msg = f"IQ source delivered no samples within {_FIRST_SAMPLE_TIMEOUT_S:.0f}s"
        raise EngineFailure(msg) from None
    if first_chunk is None:
        msg = "IQ source ended before the first sample (open/configure failed?)"
        raise EngineFailure(msg)

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            # "raw_bytes" is an explicit key in gs-client's spec_for_data_format map
            # (same RAW_BITS spec the old undeclared "raw_bits" label reached only
            # via the unknown-label fallback — docs/10 LOW-7 label drift).
            "data_format": "raw_bytes",
            "sample_rate": int(sample_rate),
            "symbol_rate": int(profile.symbol_rate_hz),
            "engine": "dsp",
            # Now a list: every light framing tried live. `framing_hint` is the backend/SatYAML
            # label we were told (informational — no longer an exclusive filter).
            "framing": ",".join(name for name, _ in decoders),
            "framing_hint": declared or "none",
            "flowgraph_version": VERSION,
            **sdr_ready_fields(
                device=str(sdr_report.get("device", args.sdr_args or "")),
                requested=sdr_report.get("requested"),  # type: ignore[arg-type]
                applied=sdr_report.get("applied"),  # type: ignore[arg-type]
                actual=sdr_report.get("actual"),  # type: ignore[arg-type]
                stream_active=True,
                first_samples=True,
            ),
        },
    )

    async def _decode_loop() -> None:
        decode_errors = 0
        while not stop_requested.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_requested.wait(), _DECODE_PERIOD_S)
            if stop_requested.is_set():
                break
            # Decode OFF the event loop (docs/10 MED-3): a long demod must never
            # stall command/status handling. The decoders lock their chunk
            # hand-off, so pushes from the loop thread stay safe meanwhile. Each
            # light framing drains independently and its frames are tagged with it.
            for fname, dec in decoders:
                try:
                    bodies = await asyncio.to_thread(dec.decode_new)
                except Exception:  # noqa: BLE001 — docs/J HIGH-1: a decoder bug must
                    # not end live decode. Unhandled, this task's exception would
                    # vanish into gather(return_exceptions=True) below and every
                    # later drain of the pass would be silently dropped. Log it
                    # (rate-limited: one bad window tends to mean many) and keep
                    # draining — the decoder's buffer hand-off already completed, so
                    # the next drain starts clean.
                    decode_errors += 1
                    if decode_errors <= 3 or decode_errors % 50 == 0:
                        log.exception(
                            "%s decode_new failed (error #%d); decode loop continues",
                            fname,
                            decode_errors,
                        )
                    continue
                for body in bodies:
                    await _emit_frame(sockets, body, framing=fname, output_dir=out_dir)

    decode_task = asyncio.create_task(_decode_loop(), name="decode-loop")
    chunk: np.ndarray | None = first_chunk  # the proof chunk is data — decode it too
    try:
        while True:
            if chunk is None:
                break
            off = doppler["hz"]
            if off:
                n = np.arange(len(chunk))
                ph = nco_phase - 2.0 * np.pi * off * n / sample_rate
                chunk = (chunk * np.exp(1j * ph)).astype(np.complex64)
                nco_phase = float(ph[-1]) if len(ph) else nco_phase
            for _, dec in decoders:
                dec.push(chunk)
            chunk = await queue.get()
    finally:
        stop_requested.set()
        # AWAIT (never cancel) the decode task: a cancel could discard frames a
        # decode_new already consumed from the buffer in its worker thread. The
        # loop wakes promptly on stop_requested; then flush the remainder of each.
        # The reader join is BOUNDED (daemon thread): a wedged SDR read must not
        # hang teardown past the supervisor's stop deadline.
        await asyncio.to_thread(reader_thread.join, 5.0)
        await asyncio.gather(decode_task, return_exceptions=True)
        for fname, dec in decoders:
            try:
                leftovers = await asyncio.to_thread(dec.flush)
            except Exception:  # noqa: BLE001 — docs/J HIGH-1: never die invisibly
                log.exception("%s final flush failed; last-window frames not emitted", fname)
                leftovers = []
            for body in leftovers:
                await _emit_frame(sockets, body, framing=fname, output_dir=out_dir)


# ----------------------------------------------------------------------
# gnuradio engine: GR front-end -> shared framing.decode (bench)
# ----------------------------------------------------------------------


async def _run_gnuradio_engine(  # pragma: no cover (bench)
    args, sockets, params, started, stop_requested, profile, doppler
) -> None:
    """Bench engine: a GNU Radio top_block recovers the bitstream; the SAME
    ``framing.decode`` turns bits into AX.25 frames. Built lazily so this file
    imports on hosts without GNU Radio."""
    from gnuradio_gfsk import build_rx_top_block  # bench-only helper module

    from gfsk_ax25 import framing

    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    declared = _select_framing(params)  # hint only (see the dsp engine) — no longer exclusive
    out_dir = getattr(args, "output_dir", None)  # frames.jsonl alongside the IQ
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    ctx = build_rx_top_block(args, profile, sample_rate, params)
    # RX: stream + record from spawn (don't gate on cmd:start — that's TX keying);
    # cmd:start still fires 'started' and cmd:stop ends the pass.
    _ = started
    # R-11: start the graph BEFORE declaring ready — 'ready' means an ACTIVE
    # stream with first-sample proof (the recorder's unbuffered cf32 grows the
    # moment the SDR delivers), not process startup. No recorder → the proof is
    # reported unavailable (None), never fabricated.
    ctx.start()
    probe = first_sample_probe(getattr(ctx, "recorder", None))
    first: bool | None = None
    if probe is not None:
        first = await await_first_samples(probe, timeout_s=_FIRST_SAMPLE_TIMEOUT_S)
        if not first:
            ctx.stop()
            msg = f"gr-soapy stream active but no samples within {_FIRST_SAMPLE_TIMEOUT_S:.0f}s"
            raise EngineFailure(msg)
    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            # "raw_bytes": explicit gs-client spec_for_data_format key (LOW-7 drift).
            "data_format": "raw_bytes",
            "sample_rate": int(sample_rate),
            "engine": "gnuradio",
            # Both light framings run on the ONE recovered bitstream (demod once, deframe many).
            "framing": ",".join(_LIVE_FRAMINGS),
            "framing_hint": declared or "none",
            "flowgraph_version": VERSION,
            **sdr_ready_fields(
                device=str(args.sdr_args or ""),
                requested=merge_sdr_params(params),
                applied=getattr(ctx, "sdr_applied", None),
                actual=readback_soapy_settings(ctx.src),
                stream_active=True,
                first_samples=first,
            ),
        },
    )
    last_doppler = 0.0
    # Carry the tail bits across drain boundaries so a frame straddling one isn't lost
    # (~5-10 % of frames otherwise, at 2000-bit AX.25 frames per ~2 s drain @9k6). Dedup is
    # POSITIONAL: subtract exactly the frames the carried tail ALONE re-decodes (with
    # multiplicity) — a payload-set dedup would permanently suppress genuine repeat beacons,
    # which re-decode out of the tail every drain.
    tail = np.empty(0, dtype=np.uint8)
    tail_bits = 4096  # ~2 max-length AX.25 frames

    def _decode(bits_arr, framing_name):
        # One recovered bitstream, deframed by each light link layer (demod once, deframe many):
        # EnduroSat chip-packet or AX.25/HDLC. Both are CRC-gated → the wrong one yields nothing.
        if framing_name == "endurosat":
            return list(_endurosat_deframe_bits(bits_arr))
        return list(framing.decode(bits_arr, scramble=profile.scramble, nrzi=profile.nrzi))

    async def _emit_new(bits_arr, prev_tail_arr) -> None:
        # Emit new frames for EVERY light framing, each tagged, with the SAME positional tail-carry
        # dedup per framing (subtract exactly what the carried tail alone re-decodes).
        for fname in _LIVE_FRAMINGS:
            frames = _decode(bits_arr, fname)
            if prev_tail_arr.size:
                for body in _decode(prev_tail_arr, fname):  # already emitted last drain
                    if body in frames:
                        frames.remove(body)
            for body in frames:
                await _emit_frame(sockets, body, framing=fname, output_dir=out_dir)

    try:
        while not stop_requested.is_set():
            await asyncio.sleep(_DECODE_PERIOD_S)
            if doppler["hz"] != last_doppler:
                last_doppler = doppler["hz"]
                ctx.set_doppler(last_doppler)  # retune the SoapySDR source
            fresh = ctx.drain_bits()  # np.uint8 hard bits recovered by GR
            prev_tail = tail
            bits = np.concatenate([prev_tail, fresh]) if prev_tail.size else fresh
            if bits.size:
                tail = bits[-tail_bits:].copy()
            await _emit_new(bits, prev_tail)
    finally:
        ctx.stop()
        ctx.wait()
        # Final drain at stop (docs/J LOW-4, mirroring the dsp engine's flush):
        # bits recovered in the last <=_DECODE_PERIOD_S — the LOS end of the pass
        # — still sit in the GR sink when stop lands. Stop/wait first so GR has
        # flushed its pipeline, then decode once more with the same tail-carry
        # dedup as the loop body. A decode error here must not break teardown.
        try:
            fresh = ctx.drain_bits()
            bits = np.concatenate([tail, fresh]) if tail.size else fresh
            await _emit_new(bits, tail)
        except Exception:  # noqa: BLE001 — docs/J HIGH-1: never die invisibly
            log.exception("final GR drain failed; frames from the last window not emitted")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


async def amain(args) -> int:
    log = logging.getLogger("cubesat_gfsk_ax25_rx")
    params = load_params(args)
    engine = _select_engine(args, params)
    profile = _profile_from_params(params)
    log.info("engine=%s profile=%s", engine, profile)

    sockets = await connect_spawn_sockets(args)
    started = asyncio.Event()
    stop_requested = asyncio.Event()
    doppler = {"hz": 0.0}

    async def _on_start(_cmd: dict[str, object]) -> None:
        started.set()
        await send_event(sockets.status_writer, {"event": "started"})

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        started.set()  # release any waiter
        stop_reason["value"] = str(cmd.get("reason", "command"))

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        if isinstance(off, (int, float)):
            doppler["hz"] = float(off)  # shared with the running engine

    engine_fn = _run_dsp_engine if engine == "dsp" else _run_gnuradio_engine
    engine_task = asyncio.create_task(
        engine_fn(args, sockets, params, started, stop_requested, profile, doppler),
        name=f"engine-{engine}",
    )
    # R-11: an engine that dies (failed SDR open, silent source, DSP crash)
    # fails the pass — error event + nonzero exit — instead of idling behind
    # a live command loop with nothing captured.
    watch_engine_death(engine_task, sockets.status_writer, sockets.control_reader, stop_requested)
    async def _shutdown_engine() -> None:
        """Idempotent engine teardown — settle the engine task fully."""
        stop_requested.set()
        started.set()
        await asyncio.gather(engine_task, return_exceptions=True)

    handlers = {"start": _on_start, "stop": _on_stop, "set_doppler": _on_set_doppler}
    try:
        reason = await run_command_loop(sockets.control_reader, handlers)
        # P0-08: cleanup BEFORE the explicit stopped ack; then exit 0. EOF is
        # transport loss (no ack; exit nonzero).
        await _shutdown_engine()
        if reason == "stop":
            await send_event(
                sockets.status_writer,
                {"event": "stopped", "reason": stop_reason["value"]},
            )
            return 0
        log.warning("control EOF without stop — transport loss; exiting nonzero (P0-08)")
        return 1
    finally:
        await _shutdown_engine()
        await sockets.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="cubesat_gfsk_ax25_rx",
        description="2-GFSK / AX.25 (9k6) cubesat receiver — dsp | gnuradio engines.",
    )
    parser.add_argument("--engine", default="", choices=["", "dsp", "gnuradio"])
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    logging.basicConfig(level=logging.INFO)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cubesat 2-GFSK / AX.25 transmit flowgraph (EnduroSat-class UHF uplink, 9k6).

The mirror of ``cubesat_gfsk_ax25_rx.py``: builds an AX.25 UI frame from the
uplink payload, runs the shared ``gfsk_ax25`` transmit chain (HDLC -> NRZI ->
G3RUH -> 2-GFSK), and sinks the baseband IQ to the SDR (or a ``cf32`` file for
bench use). Same ``--engine {dsp,gnuradio}`` selection as the receiver.

Uplink payload sources, in order: params ``uplink_b64`` (base64), params
``uplink_file`` path, or ``<output-dir>/uplink.bin``. Callsigns come from params
``dest`` / ``src`` (default ``CQ`` / ``DSN``).

NOTE: keying the PA is the orchestrator's safety FSM responsibility (Document A
A.6); this flowgraph only produces/sends the modulated baseband and never
asserts PTT itself.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
from _fallback_select import symbol_rate_hz_of
from _soapy import (
    apply_corrections,
    configure_soapy_source,
    merge_sdr_params_tx,
    readback_soapy_settings,
    sdr_env,
    sdr_ready_fields,
)
from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)

from gfsk_ax25 import ax25, endurosat, endurosat_link

VERSION = "0.1.0"
_DEFAULT_SAMPLE_RATE = 96_000


def _select_engine(args, params: dict[str, object]) -> str:
    engine = (
        (getattr(args, "engine", "") or "")
        or os.environ.get("GS_FLOWGRAPH_ENGINE", "")
        or (str(params.get("engine", "")) if isinstance(params, dict) else "")
        or "dsp"
    ).lower()
    return engine if engine in ("dsp", "gnuradio") else "dsp"


def _select_framing(params: dict[str, object]) -> str:
    """ax25 (default) | endurosat (chip-packet). For endurosat the uplink payload
    is the already-built (encrypted AirMAC) frame, sent verbatim in the packet."""
    framing = (
        os.environ.get("GS_FLOWGRAPH_FRAMING", "")
        or (str(params.get("framing", "")) if isinstance(params, dict) else "")
        or "ax25"
    ).lower()
    return framing if framing in ("ax25", "endurosat") else "ax25"


def _uplink_payload(args, params: dict[str, object]) -> bytes:
    b64 = params.get("uplink_b64")
    if isinstance(b64, str) and b64:
        return base64.b64decode(b64)
    for candidate in (params.get("uplink_file"), Path(args.output_dir or ".") / "uplink.bin"):
        if candidate and Path(candidate).exists():
            return Path(candidate).read_bytes()
    return b""


def _build_frame_iq(args, params: dict[str, object], profile) -> np.ndarray:
    payload = _uplink_payload(args, params)
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    if _select_framing(params) == "endurosat":
        # Uplink payload is the already-built (encrypted AirMAC) frame; wrap it in
        # the EnduroSat chip packet (preamble + sync + len + CRC-16) at 9600 sym/s
        # (endurosat_link defaults), honouring params overrides if present.
        sym_hz = symbol_rate_hz_of(params, default=endurosat_link.DEFAULT_SYMBOL_RATE_HZ)
        return endurosat_link.transmit(
            payload[: endurosat_link.MAX_PAYLOAD],
            sample_rate,
            symbol_rate_hz=sym_hz,
            mod_index=float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
            bt=float(params.get("bt", endurosat_link.DEFAULT_BT)),
        )
    body = ax25.encode_ui(
        dest=str(params.get("dest", "CQ")),
        src=str(params.get("src", "DSN")),
        info=payload[: endurosat.AX25_INFO_MAX_BYTES],
    )
    return endurosat.transmit(body, sample_rate, profile=profile)


def _sink_iq(
    args,
    iq: np.ndarray,
    params: dict[str, object] | None = None,
    on_first_accept=None,
    should_abort=None,
):
    """Sink the burst; returns a ``_soapy_tx.BurstResult`` describing what was
    ACTUALLY accepted (R-16 — completion events must not fabricate success).
    ``should_abort`` is polled between chunks so a pass stop cancels an
    in-flight burst instead of radiating it to completion."""
    from _soapy_tx import BurstResult

    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        data = iq.astype(np.complex64)
        if should_abort is not None and should_abort():
            return BurstResult(accepted=0, total=len(data), outcome="cancelled",
                               detail="aborted by caller")
        Path(sdr_args[len("file:") :]).write_bytes(data.tobytes())
        if on_first_accept is not None:
            on_first_accept()
        return BurstResult(accepted=len(data), total=len(data), outcome="complete")
    return _soapy_sink(args, iq, params, on_first_accept, should_abort)  # pragma: no cover


class _SoapyDeviceAdapter:
    """Adapt a raw ``SoapySDR.Device`` to the gr-soapy ``set_*`` surface that
    ``configure_soapy_source``/``apply_corrections`` drive, bound to one direction/stream.
    Lets this direct-SoapySDR TX path reuse the SAME antenna/gain/env configuration as the
    GNU Radio engines instead of transmitting at the ~0 dB device default (``_soapy``'s
    "hears-nothing trap" — on TX: radiates nothing)."""

    def __init__(self, dev, direction: int) -> None:
        self._dev = dev
        self._direction = direction

    def set_antenna(self, channel: int, name: str) -> None:
        self._dev.setAntenna(self._direction, channel, name)

    def set_gain_mode(self, channel: int, automatic: bool) -> None:
        self._dev.setGainMode(self._direction, channel, automatic)

    def set_gain(self, channel: int, *args: object) -> None:
        # (value) = overall gain, (name, value) = per-element — mirror gr-soapy's overloads.
        self._dev.setGain(self._direction, channel, *args)

    def set_frequency_correction(self, channel: int, ppm: float) -> None:
        self._dev.setFrequencyCorrection(self._direction, channel, ppm)


def configure_tx_sink(dev, direction: int, params, sample_rate: float) -> dict:
    """Apply TX-direction settings + ppm correction to a raw SoapySDR TX device,
    mirroring the sink treatment the FM TX app / ``gnuradio_gfsk.transmit_gnuradio``
    give their gr-soapy sinks. Without this the default engine transmitted with NO
    gain configured (deaf-in-reverse: radiates nothing). Returns the applied dict.

    R-22: settings come from :func:`merge_sdr_params_tx` — per-pass ``sdr_tx_*``
    over station ``GS_SDR_TX_*`` ONLY. RX-oriented names (``GS_SDR_ANTENNA``
    ``LNAW``, ``GS_SDR_GAINS`` ``LNA/TIA/PGA``) never reach the TX endpoint;
    on LMS7/XTRX they'd raise in ``setAntenna``/``setGain`` and kill the pass.
    With nothing TX-specific configured the sane manual default gain still
    applies (the deaf-TX trap), TX-direction-addressed.

    Also sets the ANALOG TX filter to the SAMPLE rate, never the narrow channel width —
    below the device filter floor (~0.8 MHz on the XTRX) the analog path goes silent
    (station hardware rule; same fix as the FM TX app).

    NOTE: TX gain LEVELS are BENCH-PENDING — validate actual PA drive on the bench before a
    real uplink; this only ensures the front-end is configured at all."""
    endpoint = _SoapyDeviceAdapter(dev, direction)
    applied = configure_soapy_source(endpoint, merge_sdr_params_tx(params))
    applied.update(apply_corrections(endpoint, ppm=sdr_env()["ppm"], dc_removal=False))
    with contextlib.suppress(Exception):  # bandwidth setting is optional per driver
        dev.setBandwidth(direction, 0, float(sample_rate))
    return applied


def _probe_tx_device(args, params) -> dict:  # pragma: no cover (needs SoapySDR)
    """R-11 explicit TX readiness: prove the configured TX device OPENS and
    takes the front-end settings BEFORE reporting ready — fail closed at spawn,
    not at KEYED_READY with the PA energized. Opens, configures, reads back,
    closes; no stream is set up, nothing radiates (and the PA is off — keying
    is the orchestrator's safety FSM, never this app). Assumes this app owns
    the device at spawn, the same assumption ``_soapy_sink`` makes at transmit."""
    import SoapySDR
    from _stream import hardware_rate_for, require_sample_rate
    from SoapySDR import SOAPY_SDR_TX

    dev = SoapySDR.Device(args.sdr_args)
    try:
        sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
        # R-14: probe at the SUPPORTED hardware rate (integer multiple of the
        # modem rate) — probing the raw modem rate would fail on XTRX-class
        # hardware even though the transmit path is fine.
        hw_rate, _factor = hardware_rate_for(sample_rate, sdr_env()["capture_rate_hz"])
        dev.setSampleRate(SOAPY_SDR_TX, 0, hw_rate)
        require_sample_rate(dev, SOAPY_SDR_TX, 0, hw_rate)
        dev.setFrequency(SOAPY_SDR_TX, 0, float(args.center_freq_hz))
        applied = configure_tx_sink(dev, SOAPY_SDR_TX, params, hw_rate)
        actual = readback_soapy_settings(dev, channel=0, direction=SOAPY_SDR_TX)
        return {"applied": applied, "actual": actual}
    finally:
        with contextlib.suppress(Exception):  # release before the transmit-time reopen
            SoapySDR.Device.unmake(dev)


def _tx_spawn_probe(args, params) -> tuple[bool, dict[str, object]]:
    """Resolve the app's TX readiness condition at spawn. Non-hardware sinks
    (``file:`` bench mode / empty args) skip the device probe — recorded as
    such, never implied verified. Returns ``(ready_ok, ready_fields)``."""
    sdr_args = str(args.sdr_args or "")
    if not sdr_args or sdr_args.startswith("file:"):
        fields = sdr_ready_fields(
            device=sdr_args or "none", requested=None, applied=None, actual=None,
            stream_active=False, first_samples=None,
        )
        fields["tx_ready"] = "non-hardware-sink"
        return True, fields
    try:
        report = _probe_tx_device(args, params)
    except Exception as e:  # noqa: BLE001 — unopenable hardware fails closed
        logging.getLogger("cubesat_gfsk_ax25_tx").exception(
            "TX device probe failed for %r", sdr_args
        )
        return False, {"code": "tx-device-probe-failed", "detail": repr(e)}
    fields = sdr_ready_fields(
        device=sdr_args,
        requested=merge_sdr_params_tx(params),
        applied=report.get("applied"),  # type: ignore[arg-type]
        actual=report.get("actual"),  # type: ignore[arg-type]
        stream_active=False,  # TX opens its stream per burst; ready proves the DEVICE
        first_samples=None,
    )
    fields["tx_ready"] = "device-verified"
    return True, fields


def _soapy_sink(  # pragma: no cover
    args, iq: np.ndarray, params=None, on_first_accept=None, should_abort=None
):
    """P0-07: uses the shared bounded TX transport (apps/_soapy_tx.py) — MTU
    chunking, bounded zero/timeout/error handling, END_BURST on the final data
    chunk, per-burst deadline — instead of the fixed-4096 ret>0-only loop that
    could spin forever or oversize a native driver write."""
    import SoapySDR
    from _soapy_tx import query_tx_mtu, write_burst
    from _stream import hardware_rate_for, require_sample_rate, upsample_burst
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX

    log = logging.getLogger("cubesat_gfsk_ax25_tx")
    dev = SoapySDR.Device(args.sdr_args)
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    # R-14: the DEVICE runs at a supported integer multiple of the modem rate
    # (XTRX-class TX can't stream ~96 kHz); the burst is upsampled to match.
    # The readback is validated — a silently-clamped rate garbles the burst.
    hw_rate, factor = hardware_rate_for(sample_rate, sdr_env()["capture_rate_hz"])
    dev.setSampleRate(SOAPY_SDR_TX, 0, hw_rate)
    require_sample_rate(dev, SOAPY_SDR_TX, 0, hw_rate)
    dev.setFrequency(SOAPY_SDR_TX, 0, float(args.center_freq_hz))
    applied = configure_tx_sink(dev, SOAPY_SDR_TX, params, hw_rate)
    log.info("TX sink configured (modem=%.0f hw=%.0f x%d): %s",
             sample_rate, hw_rate, factor, applied)
    stream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    dev.activateStream(stream)
    try:
        buf = upsample_burst(iq.astype(np.complex64), factor)
        # Deadline: the burst's real-time duration plus generous margin — a TX
        # that cannot drain a one-burst buffer inside this is wedged, not slow.
        deadline_s = max(10.0, 3.0 * len(buf) / max(1.0, hw_rate))
        result = write_burst(
            dev,
            stream,
            buf,
            mtu=query_tx_mtu(dev, stream),
            deadline_s=deadline_s,
            on_first_accept=on_first_accept,
            should_abort=should_abort,
        )
        if not result.complete:
            log.warning(
                "TX burst incomplete: %d/%d samples (%s: %s)",
                result.accepted, result.total, result.outcome, result.detail,
            )
        return result
    finally:
        dev.deactivateStream(stream)
        dev.closeStream(stream)


async def amain(args) -> int:
    log = logging.getLogger("cubesat_gfsk_ax25_tx")
    params = load_params(args)
    engine = _select_engine(args, params)
    profile = endurosat.LinkProfile(
        scramble=bool(params.get("scramble", True)),
        nrzi=bool(params.get("nrzi", True)),
    )
    log.info("engine=%s", engine)

    sockets = await connect_spawn_sockets(args)
    stop_requested = asyncio.Event()

    # R-11: explicit TX readiness — a hardware sink is probed (open + configure
    # + readback, nothing radiates) BEFORE ready; configured-but-unopenable
    # hardware fails closed at spawn, not at KEYED_READY with the PA energized.
    tx_ok, tx_fields = await asyncio.to_thread(_tx_spawn_probe, args, params)
    if not tx_ok:
        await send_event(sockets.status_writer, {"event": "error", **tx_fields})
        await sockets.aclose()
        return 1

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "none",
            "engine": engine,
            "flowgraph_version": VERSION,
            **tx_fields,
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        await send_event(sockets.status_writer, {"event": "started"})
        if engine == "gnuradio":  # pragma: no cover (bench)
            from gnuradio_gfsk import transmit_gnuradio

            # Audit round 2: this used to emit a HARDCODED
            # {"samples": 0, "outcome": "complete"} whatever the engine did — a
            # fabricated success for a burst that may have radiated nothing, and an
            # engine exception was swallowed by the command loop on top. Report the
            # REAL burst result, and fail loudly if the engine raises.
            try:
                burst = await asyncio.to_thread(transmit_gnuradio, args, params, profile)
            except Exception as e:
                logging.getLogger("cubesat_gfsk_ax25_tx").exception(
                    "gnuradio TX engine failed"
                )
                await send_event(
                    sockets.status_writer,
                    {"event": "error", "code": "tx-engine-failed", "detail": repr(e)},
                )
                return
            await send_event(
                sockets.status_writer,
                {
                    "event": "transmit_complete",
                    "samples": int(getattr(burst, "accepted", 0)),
                    "outcome": str(getattr(burst, "outcome", "error")),
                    "detail": str(getattr(burst, "detail", "")),
                },
            )
            return
        # R-16: transmit_started is emitted only when the stream provably
        # accepts samples (bridged threadsafe from the sink worker thread),
        # and transmit_complete reports the ACCEPTED count + a bounded
        # explicit outcome — never a fabricated zero-sample success.
        loop = asyncio.get_running_loop()

        def _first_accept() -> None:
            loop.call_soon_threadsafe(
                lambda: loop.create_task(
                    send_event(sockets.status_writer, {"event": "transmit_started"})
                )
            )

        iq = _build_frame_iq(args, params, profile)
        result = await asyncio.to_thread(
            _sink_iq, args, iq, params,
            on_first_accept=_first_accept,
            should_abort=stop_requested.is_set,  # a pass stop cancels the burst
        )
        await send_event(
            sockets.status_writer,
            {
                "event": "transmit_complete",
                "samples": int(result.accepted),
                "outcome": result.outcome,
                "detail": result.detail,
            },
        )

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        stop_reason["value"] = str(cmd.get("reason", "command"))

    handlers = {"start": _on_start, "stop": _on_stop}
    try:
        reason = await run_command_loop(sockets.control_reader, handlers)
        if reason == "stop":
            # P0-08: dispatch ended on the accepted stop; the explicit stopped
            # ack follows cleanup (nothing to tear down here beyond sockets)
            # and the process exits 0.
            await send_event(
                sockets.status_writer,
                {"event": "stopped", "reason": stop_reason["value"]},
            )
            return 0
        log.warning("control EOF without stop — transport loss; exiting nonzero (P0-08)")
        return 1
    finally:
        stop_requested.set()
        await sockets.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="cubesat_gfsk_ax25_tx",
        description="2-GFSK / AX.25 (9k6) cubesat transmitter — dsp | gnuradio engines.",
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

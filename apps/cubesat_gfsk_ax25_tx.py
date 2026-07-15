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
import contextlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
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
from _uplink_frame import (
    ENGINES,
    UnknownEngine,
    UplinkFrame,
    build_uplink_frame,
    preflight,
    select_framing,
)

from gfsk_ax25 import endurosat, gfsk

VERSION = "0.1.0"
log = logging.getLogger("cubesat_gfsk_ax25_tx")

_DEFAULT_SAMPLE_RATE = 96_000


def _select_engine(args, params: dict[str, object]) -> str:
    """The TX engine. ABSENT -> dsp (the documented default). PRESENT but unknown -> RAISE.

    Round 7: an unknown engine silently became `dsp`, and the ENVIRONMENT outranked params.json — so
    a typo, or a stale GS_FLOWGRAPH_ENGINE in someone's shell, quietly changed which modulator keyed
    the PA. params.json is the station's VALIDATED configuration and it wins."""
    from_params = str(params.get("engine", "")) if isinstance(params, dict) else ""
    chosen = (
        (getattr(args, "engine", "") or "")
        or from_params
        or os.environ.get("GS_FLOWGRAPH_ENGINE", "")
    ).strip().lower()
    if not chosen:
        return "dsp"
    if chosen not in ENGINES:
        msg = (
            f"unknown engine {chosen!r} — refusing to fall back to dsp. A "
            f"silent fallback means the "
            f"PA is keyed by a modulator nobody asked for. Known: {sorted(ENGINES)}"
        )
        raise UnknownEngine(msg)
    return chosen


def _select_framing(params: dict[str, object]) -> str:
    """ax25 (default) | endurosat (chip-packet). For endurosat the uplink payload
    is the already-built (encrypted AirMAC) frame, sent verbatim in the packet.

    R2-43: the CHOICE lives here, but the framing itself lives in _uplink_frame, which BOTH engines
    go through. The gnuradio engine used to build its own frame and always chose AX.25."""
    return select_framing(params, env=os.environ.get("GS_FLOWGRAPH_FRAMING", ""))


def _build_frame(args, params: dict[str, object], profile) -> UplinkFrame:
    """The one framed uplink both engines modulate. See apps/_uplink_frame.py."""
    frame = build_uplink_frame(args, params, profile, framing_name=_select_framing(params))
    log.info(
        "uplink frame: framing=%s payload=%dB from %s | %d bits @ %.0f sym/s (h=%.2f bt=%.2f)",
        frame.framing, frame.payload_len, frame.payload_source,
        frame.bits.size, frame.symbol_rate_hz, frame.mod_index, frame.bt,
    )
    if frame.payload_len == 0:
        log.warning(
            "uplink payload is EMPTY (no uplink_b64, no uplink_file, no uplink.bin) — the burst "
            "would key the PA to transmit a frame with no content"
        )
    return frame


def _build_frame_iq(args, params: dict[str, object], profile) -> np.ndarray:
    """dsp engine: modulate the SHARED frame's bits. Nothing here decides what to transmit."""
    frame = _build_frame(args, params, profile)
    return gfsk.modulate(
        frame.bits,
        gfsk.GfskParams(
            sample_rate_hz=frame.sample_rate_hz,
            symbol_rate_hz=frame.symbol_rate_hz,
            mod_index=frame.mod_index,
            bt=frame.bt,
        ),
    )


def _preflight_and_build_iq(args, params: dict[str, object], profile, engine: str):
    """Validate EVERYTHING and produce the FINAL IQ — before `ready`, before the PA.

    Round 8. This is the difference between "the numbers look fine" and "we can actually transmit":
    it IMPORTS the selected engine (so a missing GNU Radio fails the spawn instead of the burst) and
    it MODULATES (so a payload cannot change between validation and transmission — the frame used
    on air is this frame, not a rebuilt one)."""
    frame = preflight(args, params, profile, engine=engine, framing_name=_select_framing(params))
    if engine == "gnuradio":
        from gnuradio_gfsk import modulate_gnuradio  # noqa: PLC0415 — the point IS to import it now

        iq = modulate_gnuradio(frame)
    else:
        iq = gfsk.modulate(
            frame.bits,
            gfsk.GfskParams(
                sample_rate_hz=frame.sample_rate_hz,
                symbol_rate_hz=frame.symbol_rate_hz,
                mod_index=frame.mod_index,
                bt=frame.bt,
            ),
        )
    if iq.size == 0:
        msg = "the modulator produced ZERO samples — refusing to key the PA"
        raise ValueError(msg)
    if not np.all(np.isfinite(iq.view(np.float32))):
        msg = "the modulator produced NON-FINITE IQ (NaN/Inf) — refusing to key the PA"
        raise ValueError(msg)
    log.info(
        "TX preflight: %s engine produced %d IQ samples — cached for the burst",
        engine, iq.size,
    )
    return iq


async def emit_burst(
    status_writer, args, params: dict[str, object], profile, engine: str, *,
    should_abort=None, iq=None, cs16=None,
) -> None:
    """Modulate, key the burst through the ONE validated sink, and report it honestly.

    Module-level (not a closure) so it is DRIVEABLE BY A TEST — the previous version was
    nested inside the app's main coroutine, which is how a TX-safety defect survived: the
    only "test" possible was a source-text grep.

    Both engines share this path (audit round 2). The gnuradio engine used to open its own
    soapy sink and run its own graph: it never emitted ``transmit_started``, so safety
    stayed in KEYED_READY and the orchestrator's immediate de-key — which fires only from
    KEYED — never ran. The PA stayed energized and T/R stayed on TX until LOS. It also
    counted SOURCE items instead of samples the SDR accepted, skipped the shared payload
    selection, and never validated the hardware rate. The engine now only MODULATES.

    R-16: ``transmit_started`` is emitted only when the stream PROVABLY accepts a sample
    (bridged threadsafe from the sink worker thread); ``transmit_complete`` reports the
    ACCEPTED count and a bounded outcome — never a fabricated success.
    """
    loop = asyncio.get_running_loop()

    def _first_accept() -> None:
        loop.call_soon_threadsafe(
            lambda: loop.create_task(
                send_event(status_writer, {"event": "transmit_started"})
            )
        )

    try:
        # ROUND 8: use the IQ that PREFLIGHT built and validated. Rebuilding it here meant the frame
        # on air was not the frame that was checked — a file payload could change or vanish in
        # between, and the modulator's own failures (missing engine, bt=0, non-finite params) landed
        # after `ready`, with the PA keyed.
        if iq is not None:
            pass
        elif engine == "gnuradio":  # pragma: no cover (bench: needs GNU Radio)
            from gnuradio_gfsk import modulate_gnuradio  # noqa: PLC0415

            # R2-43: the engine gets the SHARED frame. It no longer resolves the payload (it knew
            # about only one of the three sources) and it no longer chooses the framing (it always
            # chose AX.25, even for an EnduroSat uplink). It modulates. That is all it does.
            iq = await asyncio.to_thread(modulate_gnuradio, _build_frame(args, params, profile))
        else:
            iq = _build_frame_iq(args, params, profile)
        # (3a) the HARDWARE sink uses cs16 (the pre-key FINAL flat CS16); the file path uses iq.
        result = await asyncio.to_thread(
            _sink_iq, args, iq, params,
            on_first_accept=_first_accept,
            should_abort=should_abort,
            cs16=cs16,
        )
    except Exception as e:
        # AUDIT ROUND 4 (P0): emitting `tx-failed` and RETURNING NORMALLY is not enough.
        # The burst handler is the `start` command handler, so returning cleanly leaves the
        # app alive and the pass running — while the PA/T-R chain is still energized
        # (KEYED_READY with no accepted sample, or KEYED after one). RAISE, so the command
        # loop's handler-failure path ends dispatch and the app exits nonzero: gs-client
        # sees a crashed engine, forces the PA off, and fails the pass. The error event is
        # still emitted first so the reason survives.
        logging.getLogger("cubesat_gfsk_ax25_tx").exception("TX burst failed")
        with contextlib.suppress(Exception):
            await send_event(
                status_writer, {"event": "error", "code": "tx-failed", "detail": repr(e)}
            )
        raise
    await send_event(
        status_writer,
        {
            "event": "transmit_complete",
            "samples": int(result.accepted),
            "outcome": result.outcome,
            "detail": result.detail,
        },
    )


def _sink_iq(
    args,
    iq: np.ndarray,
    params: dict[str, object] | None = None,
    on_first_accept=None,
    should_abort=None,
    *,
    cs16: np.ndarray | None = None,
):
    """Sink the burst; returns a ``_soapy_tx.BurstResult`` describing what was
    ACTUALLY accepted (R-16 — completion events must not fabricate success).
    ``should_abort`` is polled between chunks so a pass stop cancels an
    in-flight burst instead of radiating it to completion.

    The FILE/bench path writes the modem-rate ``iq`` as cf32 (for offline decode). The HARDWARE
    path uses ``cs16`` — the FINAL flat CS16 buffer built PRE-KEY (3a) in ``_prepare_tx_cs16`` —
    so ``_soapy_sink`` does no DSP inside the keyed window."""
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
    return _soapy_sink(args, params, cs16, on_first_accept, should_abort)  # pragma: no cover


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

    XTRX-SAFE GAIN (3c, probe-verified): the overall ``setGain`` overload is REMOVED entirely
    — it aborts SoapyXTRX (tools/probe_soapy_tx_write.py). TX drive is a NAMED per-element gain
    (the PAD element via ``sdr_tx_gains``/``GS_SDR_TX_GAINS``) and is REQUIRED: if none is
    configured this RAISES :class:`_soapy_tx.TxGainConfigError` — a deaf/would-crash TX is a
    configuration error, refused, not papered over with the unsafe overall gain. The overall
    ``sdr_gain_db`` key is stripped so ``configure_soapy_source`` can never reach that overload.

    Also sets the ANALOG TX filter to the SAMPLE rate, never the narrow channel width —
    below the device filter floor (~0.8 MHz on the XTRX) the analog path goes silent
    (station hardware rule; same fix as the FM TX app).

    NOTE: TX gain LEVELS are BENCH-PENDING — validate actual PA drive on the bench before a
    real uplink; this only ensures the front-end is configured at all."""
    from _soapy_tx import named_tx_gains

    endpoint = _SoapyDeviceAdapter(dev, direction)
    tx_settings = merge_sdr_params_tx(params)
    # (3c) REQUIRE a named per-element gain (PAD); never apply the overall setGain overload.
    named = named_tx_gains(tx_settings)  # raises TxGainConfigError when none is configured
    tx_only: dict[str, object] = {"sdr_gains": named}
    if isinstance(tx_settings.get("sdr_antenna"), str):
        tx_only["sdr_antenna"] = tx_settings["sdr_antenna"]
    applied = configure_soapy_source(endpoint, tx_only, default_gain_db=None)
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
    args, params, cs16, on_first_accept=None, should_abort=None
):
    """Open the XTRX TX stream and write the ALREADY-FINAL flat CS16 burst — (3a) NO DSP here.
    ``cs16`` is the hardware-rate flat CS16 buffer built PRE-KEY (in ``_prepare_tx_cs16`` at
    preflight), so the keyed window only configures + opens + writes; there is no modulation,
    resample, pack, or large allocation on this path.

    Conformed to the XTRX bench-probe shape (tools/probe_soapy_tx_write.py):
    ``setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16)`` with NO explicit channel list; (3d) the stream
    MTU is queried BEFORE activateStream; NO sleep between activate and the first write; write_burst
    sends it with the 3-arg call + one bounded readStreamStatus check (no flags/END_BURST/timed
    writes). (3f) TX deactivate + close are each attempted independently on exit. (3h) neither the
    absent write timeout nor the deadline can interrupt a HUNG native writeStream — the deadline
    only bounds time between writes; the real backstop is the orchestrator's keyed-window PA-off."""
    import SoapySDR
    from _soapy_tx import BurstResult, query_tx_mtu, write_burst
    from _stream import hardware_rate_for, require_sample_rate
    from SoapySDR import SOAPY_SDR_CS16, SOAPY_SDR_TX

    log = logging.getLogger("cubesat_gfsk_ax25_tx")
    buf = np.ascontiguousarray(np.asarray(cs16, dtype=np.int16))
    n_complex = int(buf.size // 2)
    if buf.size == 0:  # (3g) an empty burst is an error, not a silent success
        return BurstResult(accepted=0, total=0, outcome="error", detail="empty burst buffer")
    dev = SoapySDR.Device(args.sdr_args)
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    # R-14: the DEVICE runs at a supported integer multiple of the modem rate; the staged CS16
    # was already resampled to THIS hardware rate pre-key (same deterministic hardware_rate_for).
    # The readback is validated — a silently-clamped rate garbles the burst.
    hw_rate, factor = hardware_rate_for(sample_rate, sdr_env()["capture_rate_hz"])
    dev.setSampleRate(SOAPY_SDR_TX, 0, hw_rate)
    require_sample_rate(dev, SOAPY_SDR_TX, 0, hw_rate)
    dev.setFrequency(SOAPY_SDR_TX, 0, float(args.center_freq_hz))
    applied = configure_tx_sink(dev, SOAPY_SDR_TX, params, hw_rate)  # (3c) requires named PAD gain
    log.info("TX sink configured (modem=%.0f hw=%.0f x%d): %s",
             sample_rate, hw_rate, factor, applied)
    stream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16)  # CS16, no [0] channel list
    try:
        mtu = query_tx_mtu(dev, stream)  # (3d) query MTU BEFORE activateStream
        dev.activateStream(stream)  # no post-activate sleep (probe rule)
        # Deadline: the burst's real-time duration plus generous margin — a TX
        # that cannot drain a one-burst buffer inside this is wedged, not slow.
        deadline_s = max(10.0, 3.0 * n_complex / max(1.0, hw_rate))
        result = write_burst(
            dev,
            stream,
            buf,
            mtu=mtu,
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
        # (3f) each cleanup step is attempted INDEPENDENTLY — one failing must not skip the other.
        with contextlib.suppress(Exception):
            dev.deactivateStream(stream)
        with contextlib.suppress(Exception):
            dev.closeStream(stream)


def _prepare_tx_cs16(args, iq: np.ndarray) -> np.ndarray | None:
    """PRE-KEY (3a): build the FINAL hardware-rate flat CS16 buffer for the HARDWARE sink, so the
    keyed window (``_soapy_sink``) does NO DSP and NO large allocation. Returns None for the
    file/bench sink (which writes modem-rate cf32 for offline decode) and for an empty burst.
    Resamples to the same deterministic hardware rate ``_soapy_sink`` will set, then packs to CS16,
    validating non-empty + finite. Raises ValueError on an empty / non-finite waveform so the caller
    fails the spawn (tx-preflight-failed) rather than keying."""
    from _soapy_tx import to_cs16
    from _stream import hardware_rate_for, upsample_burst

    sdr_args = str(args.sdr_args or "")
    if not sdr_args or sdr_args.startswith("file:"):
        return None  # bench/file path writes modem-rate cf32 in _sink_iq; nothing to pre-pack
    sample_rate = float(args.sample_rate or _DEFAULT_SAMPLE_RATE)
    _hw_rate, factor = hardware_rate_for(sample_rate, sdr_env()["capture_rate_hz"])
    hw = upsample_burst(np.asarray(iq, dtype=np.complex64), int(factor))
    if hw.size == 0:
        msg = "TX preflight: the hardware-rate waveform is empty — refusing to key"
        raise ValueError(msg)
    if not np.all(np.isfinite(hw.view(np.float32))):
        msg = "TX preflight: hardware-rate waveform has non-finite IQ (NaN/Inf) — refusing to key"
        raise ValueError(msg)
    return to_cs16(hw)


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

    # Round 7: PREFLIGHT THE ENTIRE TRANSMISSION BEFORE `ready`.
    #
    # The engine, the framing, the payload length and — above all — the sample-rate/symbol-rate
    # ratio were only ever discovered by the MODULATOR, which runs after `ready`, after the T/R
    # relay
    # has been thrown and after the PA has been keyed. The canonical AX.25 TX waveform shipped at
    # 96 kHz against 12480 sym/s (7.69 samples/symbol): it could never have transmitted, and it
    # would
    # have found that out with the PA hot.
    #
    # Build the real frame here. If anything about it is unflyable, fail the spawn.
    #
    # ROUND 8: preflight must LOAD THE ENGINE and BUILD THE IQ, not merely check numbers.
    #
    #   * GNU Radio was imported only inside the burst path, AFTER `ready`. On a host without it the
    #     sequence was: ready(engine=gnuradio) -> started -> ModuleNotFoundError -> tx-failed, with
    #     the T/R relay thrown and the PA keyed for a modulator that does not exist.
    #   * the preflighted frame was then DISCARDED and rebuilt after keying, so a file payload could
    #     change or vanish between validation and transmission.
    #
    # Build the actual IQ here, once, and hand THAT to the burst.
    try:
        prevalidated_iq = await asyncio.to_thread(
            _preflight_and_build_iq, args, params, profile, engine
        )
        # (3a) build the FINAL hardware-rate flat CS16 buffer PRE-KEY (hardware sink only), so the
        # keyed window does no DSP/allocation. None for the file/bench sink (writes modem cf32).
        prevalidated_cs16 = await asyncio.to_thread(_prepare_tx_cs16, args, prevalidated_iq)
    except ValueError as e:
        log.error("TX preflight FAILED — refusing to declare ready: %s", e)
        await send_event(
            sockets.status_writer,
            {"event": "error", "code": "tx-preflight-failed", "detail": str(e)},
        )
        await sockets.aclose()
        return 1
    except ImportError as e:
        log.error("TX preflight: engine %r is NOT AVAILABLE on this host: %s", engine, e)
        await send_event(
            sockets.status_writer,
            {"event": "error", "code": "tx-engine-unavailable", "detail": str(e)},
        )
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

    # SWEEP-1 (#6): the single burst runs as a BACKGROUND task, not awaited inline. run_command_loop
    # dispatches handlers serially, so awaiting the whole burst inside `start` blocked the loop for
    # the entire transmission: a mid-burst `stop` could not be dequeued, `stop_requested` never got
    # set, and should_abort (wired to it) could never fire, so the in-flight abort was dead code
    # (and a hung native writeStream wedged the whole loop). Mirrors the bidir app's ROUND-11 P0-4
    # fix. The AUDIT ROUND 4 P0 contract (a burst FAILURE must fail the pass, nonzero exit) is
    # preserved by the done-callback + the burst_error check below.
    burst_task: dict[str, asyncio.Task[None] | None] = {"t": None}
    burst_error: list[BaseException] = []

    def _on_burst_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return  # success: wait for the orchestrator's stop, exactly as the inline path did
        # AUDIT ROUND 4 (P0): emit_burst already emitted its `error` event before raising. Record
        # the failure and force the command loop to END NOW (feed the control reader EOF) so amain
        # returns nonzero even if the orchestrator has not yet sent `stop` — matching the old inline
        # path, which surfaced the raise immediately as "handler-failed".
        burst_error.append(exc)
        stop_requested.set()
        with contextlib.suppress(Exception):
            sockets.control_reader.feed_eof()

    async def _on_start(_cmd: dict[str, object]) -> None:
        await send_event(sockets.status_writer, {"event": "started"})
        if burst_task["t"] is not None:
            return  # one-shot: ignore a duplicate start
        task = asyncio.create_task(
            emit_burst(
                sockets.status_writer, args, params, profile, engine,
                should_abort=stop_requested.is_set,
                iq=prevalidated_iq,  # ROUND 8: the frame we CHECKED is the frame we transmit
                cs16=prevalidated_cs16,  # (3a) the FINAL flat CS16 built pre-key (hardware sink)
            ),
            name="ax25-tx-burst",
        )
        burst_task["t"] = task
        task.add_done_callback(_on_burst_done)

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        stop_reason["value"] = str(cmd.get("reason", "command"))

    handlers = {"start": _on_start, "stop": _on_stop}
    try:
        reason = await run_command_loop(sockets.control_reader, handlers, sockets.status_writer)
        # SWEEP-3 (P1 #3): request the in-flight burst to ABORT before awaiting it. run_command_loop
        # returns on `stop` (stop_requested already set by _on_stop), on EOF (transport loss =
        # orchestrator gone = authority revoked), or on a handler failure — in ALL of these the pass
        # is ending, so the burst's cooperative should_abort must fire. Without setting it here, an
        # EOF / handler-failed exit awaited the burst to COMPLETION and kept writing to the keyed TX
        # stream after authority was revoked (violating _soapy_tx's never-write-after-revoke
        # contract). Matches the bidir _shutdown_engine, which sets stop_requested before awaiting.
        stop_requested.set()
        # Settle the in-flight burst so its transmit_complete/error is flushed and any failure is
        # observed BEFORE we decide the exit code or ack the stop.
        t = burst_task["t"]
        if t is not None:
            with contextlib.suppress(Exception):
                await t
        if burst_error:
            # AUDIT ROUND 4 (P0): the burst failed — fail the pass with a nonzero exit, exactly as
            # the old inline "handler-failed" path did (the error event was already emitted).
            logging.getLogger(__name__).error(
                "TX burst failed (%r) — exiting nonzero so the pass fails", burst_error[0]
            )
            return 1
        if reason == "handler-failed":
            # A command handler raised. The app must NOT return 0: gs-client's supervisor
            # classifies a clean exit as a normal stop, and the pass would complete as if
            # the command had been executed (audit — the TX apps transmit inside handlers).
            _log_handler_failure = logging.getLogger(__name__)
            _log_handler_failure.error(
                "a control-command handler failed — exiting nonzero so the pass fails"
            )
            return 1
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
        t = burst_task["t"]
        if t is not None and not t.done():
            # Teardown with the burst still in flight (e.g. control EOF mid-burst): request abort,
            # then cancel and reap so no task is left running past process exit.
            t.cancel()
            with contextlib.suppress(BaseException):
                await t
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

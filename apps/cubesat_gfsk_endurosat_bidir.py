#!/usr/bin/env python3
"""Persistent bidirectional EnduroSat 2-GFSK flowgraph — split-freq, half-duplex uplink + downlink.

ONE process owns the XTRX for the whole pass. It runs a **continuous RX demod** on the downlink
(emitting one ``frame_received`` per decoded EnduroSat chip packet, plus periodic ``signal`` RSSI),
and on a ``transmit_frame`` / ``transmit_payload_file`` control command it **bursts** the uplink on
the TX stream (emitting ``transmit_started`` → ``transmit_complete``). Uplink and downlink are
different frequencies; the single antenna is half-duplex behind a T/R switch.

Division of responsibility (Document A / docs/13):
* The **orchestrator** drives the T/R antenna switch (via the safety FSM) and decides when + how
  many times to transmit (the dynamic, elevation-gated send policy lives there, not here).
* This flowgraph **never touches the T/R lines and never keys the PA**. It only produces modulated
  baseband and demodulates the downlink. During a TX burst the RX stream is paused inside the I/O
  layer (the antenna is on the PA path anyway, so RX would be garbage).

**Uplink TX is RAW (Rev A):** the uplink file is ALREADY a complete EnduroSat packet train
(``[AA×5][7E]…[CRC]`` per packet, several concatenated, zero-byte-padded) — the framing is implicit
in the file. So the TX path modulates the bytes **VERBATIM** (raw MSB-first 2-GFSK via
``gfsk.modulate_bytes``), NOT re-wrapping them; long ``0x00`` pad runs can optionally become silence
gaps (``uplink_zero_gap_bytes``). The **RX/downlink** path is unchanged — it still deframes
EnduroSat via the proven ``gfsk_ax25.endurosat_link`` StreamDecoder. dsp engine only (no gr-soapy).

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
from _fallback_select import symbol_rate_hz_of
from _recorder import StreamRecorder
from _soapy_tx import BurstResult
from _spawn_contract import (
    build_argparser,
    connect_spawn_sockets,
    load_params,
    run_command_loop,
    send_event,
)

from gfsk_ax25 import endurosat_link, gfsk

VERSION = "0.1.0"
_DEFAULT_SAMPLE_RATE = 96_000  # multiple of 9600 → integer samples/symbol (endurosat_link requires)
_DECODE_PERIOD_S = 2.0  # how often the RX decoder is drained
_SIGNAL_PERIOD_S = 1.0  # how often an RSSI signal event is emitted
_READ_CHUNK = 4096
# R-15: samples read in this window after a TX burst reactivates the RX stream
# are the front-end's settling transient and are discarded, not decoded.
_RX_SETTLE_S = 0.05
_UPLINK_MAX_BYTES = 65_536  # sanity cap on a raw uplink train (~54 s @ 9600); guards against GB IQ
_log = logging.getLogger("cubesat_gfsk_endurosat_bidir")


# ----------------------------------------------------------------------
# EnduroSat modem parameters + frame build/demod (pure — unit-testable)
# ----------------------------------------------------------------------


def _decoder_kwargs(params: dict[str, object]) -> dict[str, float]:
    """The 2-GFSK modem parameters, from params with endurosat_link defaults (baud / baudrate /
    symbol_rate_hz are interchangeable — see symbol_rate_hz_of)."""
    return {
        "symbol_rate_hz": symbol_rate_hz_of(params, default=endurosat_link.DEFAULT_SYMBOL_RATE_HZ),
        "mod_index": float(params.get("mod_index", endurosat_link.DEFAULT_MOD_INDEX)),
        "bt": float(params.get("bt", endurosat_link.DEFAULT_BT)),
    }


def resolve_sample_rate(args, params: dict[str, object]) -> float:
    """The device/modem sample rate, SNAPPED to an integer multiple of the symbol rate.

    The 2-GFSK modulator (``gfsk.modulate``) requires an integer samples/symbol and RAISES otherwise
    — and the spawn-contract default ``--sample-rate`` is 2_000_000, which is NOT a 9600-multiple
    (208.33 sps). Left unsnapped, every uplink build would raise mid-command and be swallowed by the
    control loop (no transmit_complete → orchestrator stalls). RX (the capture-rate-robust
    StreamDecoder) is unaffected by the snap. TX and RX share one rate (one XTRX)."""
    requested = float(args.sample_rate or 0) or float(_DEFAULT_SAMPLE_RATE)
    sym = symbol_rate_hz_of(params, default=endurosat_link.DEFAULT_SYMBOL_RATE_HZ)
    sps = max(1, round(requested / sym))
    return float(sps) * sym  # exact integer samples/symbol


def build_uplink_iq(payload: bytes, sample_rate: float, params: dict[str, object]) -> np.ndarray:
    """The uplink file → 2-GFSK IQ, modulated VERBATIM (raw, MSB-first).

    The uplink file is ALREADY a complete EnduroSat packet train — ``[AA×5][7E]…[CRC]`` per packet,
    several concatenated, zero-byte-padded between them (the framing is implicit in the file). So we
    do NOT wrap it (no extra preamble/sync/len/CRC) and do NOT truncate — the bytes go on the air
    exactly as received. ``params["uplink_zero_gap_bytes"]`` (default 0 = off) optionally turns runs
    of that many ``0x00`` pad bytes into zero-amplitude silence gaps between packets (helps the
    receiver re-lock per packet) instead of full-power FSK ``0`` symbols."""
    if len(payload) > _UPLINK_MAX_BYTES:
        msg = f"uplink payload {len(payload)} B exceeds the {_UPLINK_MAX_BYTES} B sanity cap"
        raise ValueError(msg)
    kw = _decoder_kwargs(params)
    gp = gfsk.GfskParams(
        sample_rate_hz=sample_rate,
        symbol_rate_hz=kw["symbol_rate_hz"],
        mod_index=kw["mod_index"],
        bt=kw["bt"],
    )
    gap = int(params.get("uplink_zero_gap_bytes", 0) or 0)
    return gfsk.modulate_bytes_zero_gaps(payload, gp, bitorder="big", min_gap_bytes=max(0, gap))


def tx_doppler_hz(doppler_downlink_hz: float, downlink_hz: float, uplink_hz: float) -> float:
    """Uplink Doppler PRE-COMPENSATION (Hz) to add to the TX baseband, from the current DOWNLINK
    Doppler the orchestrator pushed (``set_doppler`` computes it against the downlink centre).

    Two corrections vs. the downlink value:

      * **Scale by frequency** — Doppler is proportional to the carrier, so the uplink shift is
        ``doppler_downlink * uplink_hz / downlink_hz`` (a 401 MHz downlink and a 449 MHz uplink do
        NOT share a shift).
      * **Opposite sign** — on the downlink the *received* signal carries ``+doppler`` and RX
        de-rotates it. On the uplink WE are the transmitter and the satellite is the receiver: to
        have the burst arrive at the satellite's nominal uplink frequency after it acquires Doppler
        on the way up, we transmit pre-shifted by ``-doppler_uplink``.

    Returns 0.0 when the downlink frequency is unusable (avoids a divide-by-zero) or there is no
    Doppler to apply."""
    if downlink_hz <= 0.0 or not doppler_downlink_hz:
        return 0.0
    return -doppler_downlink_hz * (uplink_hz / downlink_hz)


def apply_nco(iq: np.ndarray, freq_hz: float, sample_rate: float) -> np.ndarray:
    """Frequency-shift a complex baseband array by ``freq_hz`` (a positive value shifts up). Used to
    layer the uplink Doppler pre-compensation onto the raw modulated burst, keeping
    ``build_uplink_iq`` a pure verbatim modulator."""
    if not freq_hz or sample_rate <= 0.0:
        return np.asarray(iq, dtype=np.complex64)
    buf = np.asarray(iq, dtype=np.complex64)
    n = np.arange(len(buf))
    ph = 2.0 * np.pi * freq_hz * n / sample_rate
    return (buf * np.exp(1j * ph)).astype(np.complex64)


def demod_capture(iq: np.ndarray, sample_rate: float, params: dict[str, object]) -> list[bytes]:
    """Decode every EnduroSat frame in a complete IQ array (push-all + flush). Used by the tests and
    available for offline analysis; the live RX path streams the same StreamDecoder chunkwise."""
    decoder = endurosat_link.StreamDecoder(sample_rate, **_decoder_kwargs(params))
    decoder.push(np.asarray(iq, dtype=np.complex64))
    frames = list(decoder.decode_new())
    frames.extend(decoder.flush())
    return frames


# ----------------------------------------------------------------------
# I/O seam: one device shared by RX + TX. The SoapySDR path is bench-only;
# FileBidirIo is the unit-testable path (RX from a cf32, TX to a cf32).
# ----------------------------------------------------------------------


class BidirIo(Protocol):
    def rx_chunks(self) -> Iterator[np.ndarray]:
        """Blocking generator of complex64 downlink chunks; ends (StopIteration) on source EOF."""
        ...

    def transmit_burst(
        self, iq: np.ndarray, *, on_first_accept: Callable[[], None] | None = None
    ) -> BurstResult:
        """Send one uplink burst; returns the shared transport's BurstResult
        (accepted count + explicit outcome — R-16). ``on_first_accept`` fires
        when the sink provably takes samples. Pauses RX internally if the
        underlying device is shared (so RX and TX never touch it concurrently)."""
        ...

    def close(self) -> None: ...


class FileBidirIo:
    """Testable/bench-file I/O: RX from a ``.cf32`` file, TX appended to a ``.cf32`` file.

    ``rx_path`` None or missing → no downlink (RX yields nothing). ``tx_path`` None → transmitted IQ
    is discarded (kept in :attr:`sent_samples` for assertions)."""

    def __init__(self, rx_path: str | None, tx_path: str | None = None) -> None:
        self._rx_path = rx_path
        self._tx_path = tx_path
        self.sent_samples = 0

    def rx_chunks(self) -> Iterator[np.ndarray]:
        if not self._rx_path or not Path(self._rx_path).is_file():
            return
        with open(self._rx_path, "rb") as f:
            while True:
                raw = f.read(_READ_CHUNK * 8)  # complex64 = 8 bytes
                whole = len(raw) - (len(raw) % 8)  # floor a torn final read to whole samples
                if whole <= 0:
                    return
                yield np.frombuffer(raw[:whole], dtype=np.complex64)

    def transmit_burst(
        self, iq: np.ndarray, *, on_first_accept: Callable[[], None] | None = None
    ) -> BurstResult:
        buf = np.asarray(iq, dtype=np.complex64)
        if self._tx_path:
            with open(self._tx_path, "ab") as f:
                f.write(buf.tobytes())
        if len(buf) and on_first_accept is not None:
            on_first_accept()
        self.sent_samples += len(buf)
        return BurstResult(accepted=len(buf), total=len(buf), outcome="complete")

    def close(self) -> None:
        return None


# ----------------------------------------------------------------------
# Status emitters
# ----------------------------------------------------------------------


async def emit_frame(sockets, body: bytes) -> None:
    """One ``frame_received`` status event + the raw frame bytes on the data socket. For endurosat
    the body is the opaque (encrypted AirMAC) payload — no AX.25 parse here.

    Both writes suppress a dead-peer error: a broken status/data socket must never propagate up and
    kill the RX loop (or, worse, leave the reader parked on a full queue → teardown deadlock)."""
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        await send_event(
            sockets.status_writer,
            {
                "event": "frame_received",
                "framing": "endurosat",
                "frame": {
                    "bytes_b64": base64.b64encode(body).decode("ascii"),
                    "len": len(body),
                    "crc_ok": True,
                },
            },
        )
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        sockets.data_writer.write(body)
        await sockets.data_writer.drain()


async def emit_signal(sockets, rssi_dbm: float) -> None:
    # Suppress a dead status socket — a signal write must never kill the RX consumer loop.
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        await send_event(
            sockets.status_writer,
            {"event": "signal", "rssi_dbm": round(rssi_dbm, 1), "lock": False},
        )


def _rssi_dbm(chunk: np.ndarray) -> float:
    """Relative RSSI in dB from mean power (dBFS-ish; a soft 'something's there' hint for the
    orchestrator's listen window — not a calibrated level)."""
    power = float(np.mean(np.abs(chunk.astype(np.complex64)) ** 2))
    return 10.0 * math.log10(power) if power > 0.0 else -140.0


# ----------------------------------------------------------------------
# Core: continuous RX demod + on-command TX burst, sharing one device
# ----------------------------------------------------------------------


class _TxController:
    """Builds + bursts one EnduroSat uplink packet on a worker thread (never blocking the event
    loop), bracketed by ``transmit_started`` / ``transmit_complete``. ``tx_active`` lets a shared
    device park RX for the whole build+burst window. Serialized so overlapping transmit commands
    can't interleave bursts."""

    def __init__(
        self,
        io: BidirIo,
        *,
        doppler: dict[str, float] | None = None,
        downlink_hz: float = 0.0,
        uplink_hz: float = 0.0,
    ) -> None:
        self._io = io
        self._lock = asyncio.Lock()
        self.tx_active = threading.Event()
        # Shared with run_rx: the orchestrator's set_doppler push lands here (downlink Doppler).
        # With no doppler dict or a 0 downlink freq the pre-comp is inert (tx_doppler_hz → 0).
        self._doppler = doppler if doppler is not None else {"hz": 0.0}
        self._downlink_hz = downlink_hz
        self._uplink_hz = uplink_hz

    def _build_tx_iq(
        self, payload: bytes, sample_rate: float, params: dict[str, object]
    ) -> np.ndarray:
        """Raw verbatim modulation + uplink Doppler pre-compensation (runs on a worker thread)."""
        iq = build_uplink_iq(payload, sample_rate, params)
        tx_dop = tx_doppler_hz(self._doppler["hz"], self._downlink_hz, self._uplink_hz)
        return apply_nco(iq, tx_dop, sample_rate)

    async def transmit(
        self, sockets, payload: bytes, sample_rate: float, params: dict[str, object]
    ) -> int:
        """R-16: ``transmit_started`` fires only when the sink ACCEPTS its first
        sample (bridged threadsafe from the burst worker), never on command
        receipt/IQ build; ``transmit_complete`` ALWAYS fires and carries the
        accepted count plus an explicit bounded outcome — a build/burst failure
        is outcome="error", not a nominal zero-sample success. Exceptions are
        logged, not propagated, so the orchestrator's half-duplex loop never
        stalls on them."""
        async with self._lock:
            loop = asyncio.get_running_loop()

            def _first_accept() -> None:
                loop.call_soon_threadsafe(
                    lambda: loop.create_task(
                        send_event(sockets.status_writer, {"event": "transmit_started"})
                    )
                )

            accepted = 0
            outcome = "error"
            detail = ""
            self.tx_active.set()
            try:
                iq = await asyncio.to_thread(self._build_tx_iq, payload, sample_rate, params)
                result = await asyncio.to_thread(
                    self._io.transmit_burst, iq, on_first_accept=_first_accept
                )
                accepted = int(result.accepted)
                outcome = result.outcome
                detail = result.detail
            except Exception as e:  # noqa: BLE001 — must still emit transmit_complete
                _log.exception("bidir TX: uplink build/send failed")
                detail = repr(e)
            finally:
                self.tx_active.clear()
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "transmit_complete",
                        "samples": accepted,
                        "outcome": outcome,
                        "detail": detail,
                    },
                )
            return accepted


async def run_rx(
    args,
    sockets,
    params: dict[str, object],
    io: BidirIo,
    *,
    stop_requested: asyncio.Event,
    doppler: dict[str, float],
    tx: _TxController,
) -> None:
    """Continuous downlink demod: reader thread → queue → StreamDecoder, a decode loop draining
    frames, and periodic RSSI. Mirrors the RX app's dsp engine but endurosat-only. RX pauses while a
    TX burst is in flight (shared antenna/device)."""
    sample_rate = resolve_sample_rate(args, params)
    decoder = endurosat_link.StreamDecoder(sample_rate, **_decoder_kwargs(params))
    recorder = StreamRecorder.maybe_start(args, sample_rate_hz=sample_rate)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=64)
    nco_phase = 0.0
    last_signal = 0.0

    def _put(item: np.ndarray | None) -> None:
        # Backpressure: block the reader thread until the item is enqueued, so a fast source (a
        # bench cf32 replay outpacing the consumer) can't overflow the bounded queue or lose the
        # terminator. INTERRUPTIBLE: if the consumer stops draining (e.g. it died) we must not park
        # here forever — poll stop_requested and bail (dropping the item) so teardown can't hang.
        fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while True:
            try:
                fut.result(timeout=0.1)
                return
            except concurrent.futures.TimeoutError:
                if stop_requested.is_set():
                    fut.cancel()  # tearing down and nobody is draining — abandon this put
                    return

    def _reader() -> None:
        try:
            for chunk in io.rx_chunks():
                if stop_requested.is_set():
                    break
                if tx.tx_active.is_set():
                    continue  # antenna on the PA path — RX is meaningless during a burst
                arr = np.asarray(chunk, dtype=np.complex64)
                if recorder is not None:
                    recorder.write(arr)
                _put(arr)
        except Exception:
            _log.exception("bidir RX: IQ source error")
        finally:
            if recorder is not None:
                recorder.close()
            with contextlib.suppress(Exception):
                _put(None)

    reader_task = loop.run_in_executor(None, _reader)

    async def _decode_loop() -> None:
        errors = 0
        while not stop_requested.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_requested.wait(), _DECODE_PERIOD_S)
            if stop_requested.is_set():
                break
            try:
                bodies = await asyncio.to_thread(decoder.decode_new)
            except Exception:  # noqa: BLE001 — a decoder bug must not end live decode (docs/J HIGH-1)
                errors += 1
                if errors <= 3 or errors % 50 == 0:
                    _log.exception("bidir RX: decode_new failed (#%d); continuing", errors)
                continue
            for body in bodies:
                await emit_frame(sockets, body)

    decode_task = asyncio.create_task(_decode_loop(), name="bidir-decode")
    try:
        while True:
            # Terminate on the None sentinel OR on stop+drained — do NOT depend solely on the
            # sentinel: at teardown the interruptible _put may drop it (stop set + full queue), so a
            # sentinel-only exit could block forever on an empty queue. The reader stops producing
            # once stop is set, so "stop and empty" means no more data will ever arrive.
            if stop_requested.is_set() and queue.empty():
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if chunk is None:
                break
            off = doppler["hz"]
            if off:
                n = np.arange(len(chunk))
                ph = nco_phase - 2.0 * np.pi * off * n / sample_rate
                chunk = (chunk * np.exp(1j * ph)).astype(np.complex64)
                nco_phase = float(ph[-1]) if len(ph) else nco_phase
            now = time.monotonic()
            if now - last_signal >= _SIGNAL_PERIOD_S:
                last_signal = now
                await emit_signal(sockets, _rssi_dbm(chunk))
            decoder.push(chunk)
    finally:
        stop_requested.set()
        await asyncio.gather(reader_task, decode_task, return_exceptions=True)
        try:
            leftovers = await asyncio.to_thread(decoder.flush)
        except Exception:  # noqa: BLE001 — never die invisibly at teardown (docs/J HIGH-1)
            _log.exception("bidir RX: final flush failed; last-window frames not emitted")
            leftovers = []
        for body in leftovers:
            await emit_frame(sockets, body)


def _uplink_payload_from_cmd(cmd: dict[str, object], args, params: dict[str, object]) -> bytes:
    """Resolve the uplink bytes for a transmit command, in order: command ``bytes_b64`` → command /
    params ``payload_file`` / ``uplink_file`` → ``<output-dir>/uplink.bin``. Empty → no payload."""
    b64 = cmd.get("bytes_b64")
    if isinstance(b64, str) and b64:
        # validate=True so a malformed field falls through to the file candidates rather than
        # silently decoding stray characters into unintended (transmitted) bytes.
        with contextlib.suppress(Exception):
            return base64.b64decode(b64, validate=True)
    candidates = [
        cmd.get("payload_file"),
        params.get("uplink_file"),
        Path(args.output_dir or ".") / "uplink.bin",
    ]
    for candidate in candidates:
        if candidate and Path(str(candidate)).is_file():
            return Path(str(candidate)).read_bytes()
    return b""


# ----------------------------------------------------------------------
# Bench SDR I/O (one device, RX + TX streams) — not unit-covered
# ----------------------------------------------------------------------


def _open_bidir_io(args, params: dict[str, object]) -> BidirIo:
    """File I/O for ``--sdr-args file:...`` (rx) / bench; otherwise a shared SoapySDR device."""
    sdr_args = str(args.sdr_args or "")
    if sdr_args.startswith("file:"):
        rx = sdr_args[len("file:") :]
        tx = str(getattr(args, "output_dir", "") or ".") + "/uplink_tx.cf32"
        return FileBidirIo(rx, tx)
    return _SoapyBidirIo(args, params)  # pragma: no cover (needs hardware/SoapySDR)


class _SoapyBidirIo:  # pragma: no cover (needs hardware/SoapySDR)
    """One SoapySDR device, RX stream (downlink) + TX stream (uplink). RX and TX serialize on a lock
    so the device is never touched concurrently; a TX burst sets ``tx_active`` so the RX generator
    parks. Uplink freq comes from params ``uplink_hz`` (the split-freq TX tune); downlink is
    ``--center-freq-hz``. TX gain levels are BENCH-PENDING (validate PA drive before a real uplink).
    """

    def __init__(self, args, params: dict[str, object]) -> None:
        import SoapySDR
        from _soapy import apply_corrections, configure_soapy_source, merge_sdr_params, sdr_env
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX

        self._sd = SoapySDR
        self._RX, self._TX, self._CF32 = SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CF32
        self._lock = threading.Lock()
        self.tx_active = threading.Event()
        rate = resolve_sample_rate(args, params)  # integer sps — must match the modulator's IQ rate
        self._rate = rate
        dev = SoapySDR.Device(args.sdr_args)
        self._dev = dev
        env = sdr_env()
        merged = merge_sdr_params(params)
        # RX on the downlink.
        dev.setSampleRate(SOAPY_SDR_RX, 0, rate)
        dev.setFrequency(SOAPY_SDR_RX, 0, float(args.center_freq_hz))
        with contextlib.suppress(Exception):
            dev.setBandwidth(SOAPY_SDR_RX, 0, rate)
        # TX on the uplink freq (split-freq). params['uplink_hz'] falls back to the RX centre.
        uplink_hz = float(params.get("uplink_hz", args.center_freq_hz) or args.center_freq_hz)
        self._uplink_hz = uplink_hz
        dev.setSampleRate(SOAPY_SDR_TX, 0, rate)
        dev.setFrequency(SOAPY_SDR_TX, 0, uplink_hz)
        with contextlib.suppress(Exception):
            dev.setBandwidth(SOAPY_SDR_TX, 0, rate)

        class _EP:
            def __init__(self, d: object, direction: int) -> None:
                self._d, self._dir = d, direction

            def set_antenna(self, ch: int, name: str) -> None:
                self._d.setAntenna(self._dir, ch, name)

            def set_gain_mode(self, ch: int, auto: bool) -> None:
                self._d.setGainMode(self._dir, ch, auto)

            def set_gain(self, ch: int, *a: object) -> None:
                self._d.setGain(self._dir, ch, *a)

            def set_frequency_correction(self, ch: int, ppm: float) -> None:
                self._d.setFrequencyCorrection(self._dir, ch, ppm)

        configure_soapy_source(_EP(dev, SOAPY_SDR_RX), merged)
        # Deliberately DO NOT push the RX-oriented ``merged`` params (GS_SDR_ANTENNA/GS_SDR_GAINS —
        # e.g. antenna "LNAW", gain elements LNA/TIA/PGA) onto the TX endpoint: those names are
        # RX-only on LMS7/XTRX-class devices and setAntenna/setGain would RAISE, aborting the shared
        # device init and losing the DOWNLINK too (not just the uplink). TX freq/rate/bandwidth are
        # set explicitly above and the ppm correction below; TX antenna + PA-drive gain staging is
        # BENCH-PENDING and must be configured with TX-appropriate element names once validated.
        apply_corrections(_EP(dev, SOAPY_SDR_RX), ppm=env["ppm"], dc_removal=env["dc_removal"])
        # The uplink LO shares the XTRX reference, so the SAME ppm correction must ride the TX chain
        # — otherwise the calibrated reference error (ppm * uplink_hz, ~kHz) rides on top of the
        # Doppler pre-comp and pushes the burst off the satellite's narrow AirMAC receive window.
        # DC-offset removal is an RX-only concern, so it is not applied here.
        apply_corrections(_EP(dev, SOAPY_SDR_TX), ppm=env["ppm"], dc_removal=False)
        self._rx_stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
        dev.activateStream(self._rx_stream)
        # R-15: after a TX burst reactivates RX, samples before this monotonic
        # deadline are the front-end's settling transient and are discarded.
        self._rx_settle_until = 0.0
        _log.info("bidir SoapySDR: rx=%.0f tx=%.0f rate=%.0f", args.center_freq_hz, uplink_hz, rate)

    def rx_chunks(self) -> Iterator[np.ndarray]:
        from SoapySDR import SOAPY_SDR_OVERFLOW, SOAPY_SDR_TIMEOUT

        buff = np.empty(_READ_CHUNK, dtype=np.complex64)
        while True:
            if self.tx_active.is_set():
                time.sleep(0.001)
                continue
            with self._lock:
                sr = self._dev.readStream(self._rx_stream, [buff], len(buff), timeoutUs=200_000)
            if sr.ret > 0:
                if time.monotonic() < self._rx_settle_until:
                    continue  # R-15: post-reactivation settling transient — discard
                yield buff[: sr.ret].copy()
            elif sr.ret in (SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW):
                continue
            else:
                _log.warning("bidir RX readStream ret=%d", sr.ret)

    def transmit_burst(self, iq: np.ndarray, *, on_first_accept=None):
        """One uplink burst through the shared bounded TX transport (P0-07).

        R-15: a REAL half-duplex transition — the RX stream is DEACTIVATED
        before the TX stream activates (break-before-make at the stream level;
        the antenna-side T/R relay is gs-client's safety sequencer), and
        reactivated afterwards with a settle window during which the reader
        discards samples (front-end transient, not downlink).

        Returns a ``_soapy_tx.BurstResult`` (R-16: the caller's events must
        report the ACCEPTED count and an explicit outcome, never a nominal
        zero-sample completion)."""
        from _soapy_tx import query_tx_mtu, write_burst

        buf = np.asarray(iq, dtype=np.complex64)
        self.tx_active.set()
        try:
            with self._lock:
                with contextlib.suppress(Exception):
                    self._dev.deactivateStream(self._rx_stream)  # break RX first (R-15)
                tx = self._dev.setupStream(self._TX, self._CF32, [0])
                self._dev.activateStream(tx)
                try:
                    # Deadline: burst real-time duration + generous margin — a
                    # writer that cannot drain one burst in this is wedged; it
                    # must not hold the device lock through ~21 one-second
                    # timeouts (P0-07).
                    deadline_s = max(5.0, 3.0 * len(buf) / max(1.0, self._rate))
                    result = write_burst(
                        self._dev,
                        tx,
                        buf,
                        mtu=query_tx_mtu(self._dev, tx),
                        deadline_s=deadline_s,
                        on_first_accept=on_first_accept,
                    )
                    if not result.complete:
                        _log.warning(
                            "bidir TX burst incomplete: %d/%d (%s: %s)",
                            result.accepted, result.total, result.outcome, result.detail,
                        )
                    with contextlib.suppress(Exception):
                        self._dev.readStreamStatus(tx, timeoutUs=200_000)  # let the burst flush
                finally:
                    self._dev.deactivateStream(tx)
                    self._dev.closeStream(tx)
                    with contextlib.suppress(Exception):
                        self._dev.activateStream(self._rx_stream)  # make RX again (R-15)
                    # Reader discards until here — reactivation transient.
                    self._rx_settle_until = time.monotonic() + _RX_SETTLE_S
            return result
        finally:
            self.tx_active.clear()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._dev.deactivateStream(self._rx_stream)
            self._dev.closeStream(self._rx_stream)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


async def amain(args) -> int:
    params = load_params(args)
    sample_rate = resolve_sample_rate(args, params)  # integer samples/symbol (shared by RX + TX)
    io = _open_bidir_io(args, params)
    sockets = await connect_spawn_sockets(args)
    stop_requested = asyncio.Event()
    doppler = {"hz": 0.0}
    # Uplink Doppler pre-comp needs both carriers: downlink = --center-freq-hz, uplink = params
    # uplink_hz (split-freq TX tune; falls back to the downlink centre for a same-freq link).
    downlink_hz = float(args.center_freq_hz or 0.0)
    uplink_hz = float(params.get("uplink_hz", downlink_hz) or downlink_hz)
    tx = _TxController(io, doppler=doppler, downlink_hz=downlink_hz, uplink_hz=uplink_hz)
    # If the device exposes a shared tx_active flag (SoapySDR path), let the TX controller drive it
    # so the RX generator parks during a burst.
    io_flag = getattr(io, "tx_active", None)
    if isinstance(io_flag, threading.Event):
        tx.tx_active = io_flag

    await send_event(
        sockets.status_writer,
        {
            "event": "ready",
            "data_format": "raw_bytes",
            "sample_rate": int(sample_rate),
            "symbol_rate": int(_decoder_kwargs(params)["symbol_rate_hz"]),
            "engine": "dsp",
            "framing": "endurosat",
            "direction": "bidirectional",
            "flowgraph_version": VERSION,
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        await send_event(sockets.status_writer, {"event": "started"})

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        await send_event(
            sockets.status_writer,
            {"event": "stopped", "reason": str(cmd.get("reason", "command"))},
        )

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        if isinstance(off, (int, float)) and not isinstance(off, bool):
            doppler["hz"] = float(off)

    async def _on_transmit(cmd: dict[str, object]) -> None:
        # Handles both "transmit_frame" (inline bytes_b64) and "transmit_payload_file" (a file). The
        # current orchestrator sends "transmit_frame" for BOTH (ControlWriter.transmit_payload_file
        # writes cmd="transmit_frame" with a payload_file field); the second key is future-proofing.
        payload = _uplink_payload_from_cmd(cmd, args, params)
        if not payload:
            _log.warning("bidir TX: transmit command with no payload; ignoring")
            return
        await tx.transmit(sockets, payload, sample_rate, params)  # build+burst, bracketed by events

    rx_task = asyncio.create_task(
        run_rx(args, sockets, params, io, stop_requested=stop_requested, doppler=doppler, tx=tx),
        name="bidir-rx",
    )
    handlers = {
        "start": _on_start,
        "stop": _on_stop,
        "set_doppler": _on_set_doppler,
        "transmit_frame": _on_transmit,
        "transmit_payload_file": _on_transmit,
    }
    try:
        await run_command_loop(sockets.control_reader, handlers)
    finally:
        stop_requested.set()
        await asyncio.gather(rx_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            io.close()
        await sockets.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser(
        prog="cubesat_gfsk_endurosat_bidir",
        description="Bidirectional EnduroSat 2-GFSK (9k6): downlink demod + uplink burst.",
    )
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

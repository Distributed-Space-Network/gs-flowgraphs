#!/usr/bin/env python3
"""Persistent bidirectional EnduroSat 2-GFSK flowgraph — split-freq, half-duplex uplink + downlink.

ONE process owns the XTRX for the whole pass. It runs a **continuous RX demod** on the downlink
(emitting one ``frame_received`` per decoded EnduroSat chip packet, plus periodic ``signal`` RSSI),
and on command it **bursts** the uplink on the TX stream. Uplink and downlink are different
frequencies; the single antenna is half-duplex behind a T/R switch.

**The uplink is a TWO-STEP HANDSHAKE, and the split is a safety boundary (round 10):**

1. ``prepare_transmit`` — PRE-KEY, station safe, antenna on the LNA, PA cold. The payload is
   resolved (read off disk), validated against hard bounds, and fully modulated into cached
   baseband IQ. Answers ``tx_prepared`` (licence to key) or ``tx_prepare_failed`` (do NOT key).
2. ``transmit_frame`` — POST-KEY, PA hot. Rotates the cached IQ by the current Doppler and pushes
   it. Builds NOTHING. An unstaged frame is REFUSED, not built.

That split exists because the old single-step flow did the disk read, the framing, the GFSK
modulation and the ``np.repeat`` allocation *after* the orchestrator had already keyed the PA and
thrown the antenna onto it. A bad payload — an oversize blob, a nonsense baud from the backend —
was therefore discovered with the station radiating, and was recovered by a FORCED DISARM: the
emergency path, entered on a routine bad input. Now every step that can fail happens while the
station is cold, and the only DSP left inside the keyed window is a fixed-size rotation.

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
import contextlib
import hashlib
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from _fallback_select import SYMBOL_RATE_KEYS, symbol_rate_hz_of
from _recorder import StreamRecorder
from _soapy_tx import BurstResult, to_cs16
from _spawn_contract import (
    EngineFailure,
    build_argparser,
    connect_spawn_sockets,
    frame_received_event,
    load_params,
    run_command_loop,
    send_event,
    watch_engine_death,
)
from _stream import (
    StreamingDecimator,
    apply_nco_chunk,
    hardware_rate_for,
    make_backpressure_put,
    require_sample_rate,
    upsample_burst,
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
# Finding #17: a data-socket peer that stays CONNECTED but stops reading (its TCP
# receive buffer fills) makes an unbounded ``data_writer.drain()`` await forever —
# ConnectionReset/BrokenPipe never fire — so the decode loop cannot observe a stop
# and engine teardown wedges until the supervisor SIGTERMs us (the stop-protocol
# circular-wait this codebase treats as a defect). Bounding the drain turns a
# wedged-but-open peer into a routine peer failure: drop the frame body, keep the
# RX loop alive, let teardown proceed.
_DATA_DRAIN_TIMEOUT_S = 5.0
_UPLINK_MAX_BYTES = 65_536  # sanity cap on a raw uplink train (~54 s @ 9600); guards against GB IQ
# ROUND 10 — the allocation ceilings the standalone TX app already had and this one did not.
# gfsk.modulate() does np.repeat(symbols, sps): sps and the total sample count are what actually
# decide how much memory a burst asks for. A 64 kB payload is only "small" at a sane sps.
_MAX_SPS = 1024
# ~1.6 GB of complex64 — far past any real burst, short of an OOM kill:
_MAX_IQ_SAMPLES = 200_000_000
# ROUND 12 (11th audit, P0): the MODEM cap above is not the memory that gets allocated. The burst is
# upsampled to the HARDWARE rate before it goes out — samples * factor — and resample_poly
# materializes that whole buffer (plus temporaries) AFTER the PA is keyed. A burst under the modem
# cap can still be ~0.5 GB of hardware IQ. This bounds the hardware-rate sample count, checked
# BEFORE keying, so the post-key allocation is within a reviewed ceiling. ~64M complex64 ≈ 512 MiB
# of output; resample_poly's transient sits on top of it, keeping peak well under an OOM.
_MAX_HARDWARE_IQ_SAMPLES = 64_000_000
_MIN_SYMBOL_RATE_HZ = 1200.0  # protocol floor; the REST backend has offered baud=10
_MAX_SYMBOL_RATE_HZ = 10_000_000.0
_MAX_SAMPLE_RATE_HZ = 100_000_000.0
# ROUND 11 (P0-5) — THE RF DURATION BOUND. The round-10 caps bound the MODEM-rate IQ, but the burst
# is UPSAMPLED to the hardware rate AFTER the PA is keyed, and the RF DURATION = payload_bits / baud
# is independent of the sample rate. A 64 kB payload at the 1200-baud floor is 437 SECONDS of RF —
# past gs-client's burst-completion timeout, so gs-client gives up while the PA keeps radiating for
# 6 more minutes, and the post-key upsample allocates ~7 GB. Capping the DURATION is the
# load-bearing check: it bounds air time and, at any sane rate, the allocation with it.
#
# ROUND 11 (re-check, D5): this is the ENGINE'S hard ceiling — no burst longer than this is ever
# built. The ONE SHARED DEADLINE is then established at run time: tx_prepared reports the actual
# burst duration, and the orchestrator sizes its completion wait to that reported duration (not a
# second hardcoded constant). So the engine's cap and the orchestrator's wait cannot drift apart.
_MAX_BURST_SECONDS = 30.0
_log = logging.getLogger("cubesat_gfsk_endurosat_bidir")


class UplinkRejected(ValueError):
    """The staged uplink cannot be flown. Raised ONLY on the pre-key path, never while keyed."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


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


def validate_uplink(
    payload: bytes, sample_rate: float, params: dict[str, object], *, hardware_factor: int = 1
) -> int:
    """Prove this burst is FLYABLE, and return the exact IQ sample count it will allocate.

    ROUND 10. Every check here used to be absent from the bidirectional path — the app's only guard
    was the 64 kB byte cap, and the REAL cost of a burst is not its byte count but ``sps`` and the
    total sample count, because ``gfsk.modulate`` does ``np.repeat(symbols, sps)``. A 9600-baud
    payload at a 100 MHz sample rate is 10416 sps; a 64 kB payload then asks numpy for ~43 GB. The
    old code discovered that INSIDE the keyed window, where the failure is a forced disarm.

    This runs BEFORE the PA is keyed and raises UplinkRejected, which the caller turns into a
    ``tx_prepare_failed`` event. Nothing here touches hardware.
    """
    if not payload:
        raise UplinkRejected("empty-payload", "the uplink payload is empty — nothing to transmit")
    if len(payload) > _UPLINK_MAX_BYTES:
        raise UplinkRejected(
            "payload-too-large",
            f"uplink payload {len(payload)} B exceeds the {_UPLINK_MAX_BYTES} B cap",
        )

    # ROUND 11 (P1): REJECT an explicit garbage baud, do not silently coerce it. symbol_rate_hz_of
    # treats a present-but-invalid baud (NaN, Inf, <=0) as "absent" and falls back to the 9600
    # default — which is right for a DEMOD that must keep going, but wrong for a TX COMMAND: a
    # station told to transmit at baud=NaN must be REFUSED, not quietly retuned. The rest of the DSP
    # sees only the coerced value, so the check has to happen here against the raw command.
    for key in SYMBOL_RATE_KEYS:
        if key in params:
            try:
                raw = float(params[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise UplinkRejected(
                    "symbol-rate-unusable", f"the commanded {key}={params[key]!r} is not a number"
                ) from None
            if not (math.isfinite(raw) and raw > 0.0):
                raise UplinkRejected(
                    "symbol-rate-unusable",
                    f"the commanded {key}={params[key]!r} is not a usable baud",
                )
            break

    kw = _decoder_kwargs(params)
    sym = float(kw["symbol_rate_hz"])
    mod_index = float(kw["mod_index"])
    bt = float(kw["bt"])

    for name, value in (("symbol_rate", sym), ("sample_rate", sample_rate),
                        ("mod_index", mod_index), ("bt", bt)):
        if not math.isfinite(value):
            raise UplinkRejected("non-finite-parameter", f"{name} is not finite: {value!r}")

    if not _MIN_SYMBOL_RATE_HZ <= sym <= _MAX_SYMBOL_RATE_HZ:
        raise UplinkRejected(
            "symbol-rate-unusable",
            f"symbol rate {sym} Hz is outside [{_MIN_SYMBOL_RATE_HZ}, {_MAX_SYMBOL_RATE_HZ}]",
        )
    if not 0.0 < sample_rate <= _MAX_SAMPLE_RATE_HZ:
        raise UplinkRejected(
            "sample-rate-unusable",
            f"sample rate {sample_rate} Hz is outside (0, {_MAX_SAMPLE_RATE_HZ}]",
        )
    if not 0.0 < mod_index <= 10.0:
        raise UplinkRejected("modulation-unusable", f"mod_index {mod_index} is outside (0, 10]")
    if not 0.0 < bt <= 10.0:
        raise UplinkRejected("modulation-unusable", f"bt {bt} is outside (0, 10]")

    ratio = sample_rate / sym
    sps = int(round(ratio))
    if sps < 1 or abs(ratio - sps) > 1e-9:
        raise UplinkRejected(
            "non-integer-sps",
            f"sample_rate/symbol_rate = {ratio!r} is not an integer samples/symbol",
        )
    if sps > _MAX_SPS:
        raise UplinkRejected(
            "sps-too-large", f"samples/symbol {sps} exceeds the {_MAX_SPS} cap"
        )

    # ROUND 11 (P0-5): THE RF DURATION BOUND, checked BEFORE keying. This is independent of the
    # sample rate — duration = payload_bits / baud — and it is the real cap on air time and on the
    # post-key hardware-rate upsample. A burst that passes the sps/sample caps can still be minutes
    # of RF at a low baud; this refuses it while the station is still cold.
    duration_s = (len(payload) * 8) / sym
    if duration_s > _MAX_BURST_SECONDS:
        raise UplinkRejected(
            "burst-too-long",
            f"this burst is {duration_s:.1f} s of RF at {sym:.0f} baud, past the "
            f"{_MAX_BURST_SECONDS:.0f} s cap — it would outlast the orchestrator's completion "
            f"deadline and radiate unattended",
        )

    # The modulator emits one symbol per BIT (2-GFSK), each repeated sps times.
    samples = len(payload) * 8 * sps
    if samples > _MAX_IQ_SAMPLES:
        raise UplinkRejected(
            "iq-too-large",
            f"this burst would allocate {samples} IQ samples "
            f"(~{samples * 8 / 1e9:.1f} GB), past the {_MAX_IQ_SAMPLES} cap",
        )

    # ROUND 12 (P0): bound the HARDWARE-rate sample count that resample_poly materializes AFTER the
    # PA is keyed. samples above is the modem-rate count; the transmit path upsamples it by
    # hardware_factor. Reject here, cold, if that post-key buffer would exceed the reviewed ceiling.
    factor = max(1, int(hardware_factor))
    hw_samples = samples * factor
    if hw_samples > _MAX_HARDWARE_IQ_SAMPLES:
        raise UplinkRejected(
            "hardware-iq-too-large",
            f"this burst upsamples to {hw_samples} hardware IQ samples (x{factor}, "
            f"~{hw_samples * 8 / 1e9:.1f} GB), past the {_MAX_HARDWARE_IQ_SAMPLES} cap — allocated "
            f"AFTER keying",
        )
    return samples


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


_NCO_CHUNK = 1 << 20  # 1 Mi samples: the working set of the rotation, independent of burst length


def apply_nco(iq: np.ndarray, freq_hz: float, sample_rate: float) -> np.ndarray:
    """Frequency-shift a complex baseband array by ``freq_hz`` (a positive value shifts up). Used to
    layer the uplink Doppler pre-compensation onto the raw modulated burst, keeping
    ``build_uplink_iq`` a pure verbatim modulator.

    ROUND 10 — CHUNKED, because this is the ONE piece of DSP that runs INSIDE the keyed window.
    ``np.exp(1j*ph)`` produces a complex128 temporary, so the naive whole-array form held ~7x the
    burst's complex64 size at peak — for a max-size burst that is several GB, transiently, with
    the PA hot. Rotating a fixed-size chunk at a time bounds the working set to ``_NCO_CHUNK``
    regardless of burst length. The per-chunk arithmetic is byte-for-byte the whole-array
    computation this replaced — same operation order, same float64 phase on the GLOBAL sample
    index (float32 phase would lose precision as the index grows), so the seam cannot glitch."""
    buf = np.asarray(iq, dtype=np.complex64)
    if not freq_hz or sample_rate <= 0.0:
        return buf
    out = np.empty_like(buf)
    for start in range(0, len(buf), _NCO_CHUNK):
        stop = min(start + _NCO_CHUNK, len(buf))
        n = np.arange(start, stop, dtype=np.float64)
        ph = 2.0 * np.pi * float(freq_hz) * n / float(sample_rate)
        out[start:stop] = (buf[start:stop] * np.exp(1j * ph)).astype(np.complex64)
    return out


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
        self,
        cs16: np.ndarray,
        *,
        on_first_accept: Callable[[], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> BurstResult:
        """Send one uplink burst. ``cs16`` is the FINAL flat CS16 buffer built entirely
        pre-key (3a) — Doppler applied, resampled, packed — so this call does NO DSP and
        NO large allocation; it only opens the TX stream and writes. Returns the shared
        transport's BurstResult (accepted count + explicit outcome — R-16).
        ``on_first_accept`` fires when the sink provably takes samples; ``should_abort``
        is polled between chunks so a pass stop cancels an in-flight burst
        (outcome="cancelled") instead of radiating it to completion. Pauses RX internally
        if the underlying device is shared (so RX and TX never touch it concurrently)."""
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
        # Bench/file I/O runs at the modem rate — no resample — so the pre-key staging
        # upsamples by 1 (the staged buffer is already the final CS16 at this rate).
        self.tx_upsample_factor = 1

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
        self,
        cs16: np.ndarray,
        *,
        on_first_accept: Callable[[], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> BurstResult:
        # (3a) receives the FINAL flat CS16 buffer built pre-key — this path does NO DSP,
        # it just writes the bytes (a raw CS16 [I0,Q0,...] dump for bench use).
        buf = np.ascontiguousarray(np.asarray(cs16, dtype=np.int16))
        n_complex = int(buf.size // 2)
        if should_abort is not None and should_abort():
            return BurstResult(accepted=0, total=n_complex, outcome="cancelled",
                               detail="aborted by caller")
        if buf.size == 0:  # (3g) an empty burst is an error, not a silent success
            return BurstResult(accepted=0, total=0, outcome="error", detail="empty burst buffer")
        if self._tx_path:
            with open(self._tx_path, "ab") as f:
                f.write(buf.tobytes())
        if on_first_accept is not None:
            on_first_accept()
        self.sent_samples += n_complex
        return BurstResult(accepted=n_complex, total=n_complex, outcome="complete")

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
        # R-18: the shared builder carries frame.id + crc_ok (the chip-packet
        # deframer is CRC-16 gated) so the orchestrator counts the frame valid.
        await send_event(
            sockets.status_writer,
            frame_received_event(body, crc_ok=True, framing="endurosat"),
        )
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        sockets.data_writer.write(body)
        # Finding #17: bound the drain. A wedged-but-open consumer (connected, not
        # reading, send buffer full) makes an unbounded drain() await forever without
        # ever raising ConnectionReset/BrokenPipe, stalling the decode loop and blocking
        # teardown until SIGTERM. A stall is a peer failure: drop this frame body and
        # keep going, exactly as a dead peer is handled above.
        try:
            await asyncio.wait_for(
                sockets.data_writer.drain(), timeout=_DATA_DRAIN_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            _log.warning(
                "bidir RX: data-socket drain stalled >%.1fs (wedged-but-open peer); "
                "dropping frame body and continuing",
                _DATA_DRAIN_TIMEOUT_S,
            )


async def emit_signal(sockets, rssi_dbm: float) -> None:
    # Suppress a dead status socket — a signal write must never kill the RX consumer loop.
    # R-17: the level is UNCALIBRATED dBFS-style IQ power, and says so — the
    # field name stays rssi_dbm for protocol shape, but source + calibrated
    # tell the consumer exactly what it is. lock is never fabricated.
    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
        await send_event(
            sockets.status_writer,
            {
                "event": "signal",
                "rssi_dbm": round(rssi_dbm, 1),
                "lock": False,
                "source": "iq-power",
                "calibrated": False,
            },
        )


def _rssi_dbm(chunk: np.ndarray) -> float:
    """Relative RSSI in dB from mean power (dBFS-ish; a soft 'something's there' hint for the
    orchestrator's listen window — not a calibrated level)."""
    power = float(np.mean(np.abs(chunk.astype(np.complex64)) ** 2))
    return 10.0 * math.log10(power) if power > 0.0 else -140.0


# ----------------------------------------------------------------------
# Core: continuous RX demod + on-command TX burst, sharing one device
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _StagedBurst:
    """A validated, fully-built FINAL burst, waiting for the key.

    (3a) ALL waveform DSP is done here, PRE-KEY: disk read, framing, modulation, Doppler
    pre-compensation, resampling to the hardware rate, the flat-CS16 pack, and the finite /
    non-empty / sample-count validation. The cache therefore holds the FINAL flat CS16
    hardware-rate buffer, Doppler ALREADY applied — so the keyed window does NO DSP and NO
    large allocation, only a stream open + write of ``cs16``.

    Doppler uses the app's last-known ``set_doppler`` offset snapshotted at STAGE time. This
    reverses the earlier "Doppler NCO at emission time" scheme: a single offset for the short
    burst is accepted so that nothing is computed after the PA is hot. HANDOFF: the orchestrator
    must send ``set_doppler`` BEFORE ``prepare_transmit`` for the pre-comp to reflect the pass.
    """

    frame_id: str
    payload_sha256: str
    cs16: np.ndarray          # FINAL flat CS16 hardware-rate buffer, Doppler applied
    complex_samples: int      # hardware-rate complex-sample count (== cs16.size // 2)
    modem_samples: int        # modem-rate complex count (predicted / RF-duration basis)


class _TxController:
    """Stages, then bursts one EnduroSat uplink packet.

    ROUND 10 — THE INVARIANT: **no uplink IQ is ever built while the PA is keyed.**

    The old flow was: gs-client keys the PA (orchestrator prepare_to_key) → sends transmit_frame →
    THIS class read the payload file off disk, framed it, ran the GFSK modulator and the np.repeat
    allocation, and only then pushed samples. Every one of those steps can fail or block, and all of
    them ran with the antenna on the PA and the PA energized. A bad payload was recovered by a
    FORCED DISARM — the emergency path, reached on a routine bad input.

    Now the orchestrator must ``prepare_transmit`` first: that validates and BUILDS the burst while
    the station is still safe, and acknowledges ``tx_prepared``. Only then does it key and send
    ``transmit_frame``, which transmits the CACHED samples. If nothing is staged, transmit REFUSES —
    it does not fall back to building, because that fallback is exactly the hazard.
    """

    def __init__(
        self,
        io: BidirIo,
        *,
        sample_rate: float,
        doppler: dict[str, float] | None = None,
        downlink_hz: float = 0.0,
        uplink_hz: float = 0.0,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        self._io = io
        # One XTRX, one rate, snapped at startup. The staged IQ and the burst MUST be built and
        # rotated at the same rate; making it an attribute rather than a per-command argument is
        # what makes that structural instead of a convention.
        self._sample_rate = float(sample_rate)
        self._lock = asyncio.Lock()
        self.tx_active = threading.Event()
        # Polled between burst chunks: a pass stop cancels an in-flight burst
        # (P0-07/P0-08 stop semantics) instead of radiating it to completion.
        self._should_abort = should_abort
        # Shared with run_rx: the orchestrator's set_doppler push lands here (downlink Doppler).
        # With no doppler dict or a 0 downlink freq the pre-comp is inert (tx_doppler_hz → 0).
        self._doppler = doppler if doppler is not None else {"hz": 0.0}
        self._downlink_hz = downlink_hz
        self._uplink_hz = uplink_hz
        # The one staged burst, built pre-key. Guarded by _lock together with the burst itself, so a
        # stage can never land between a transmit's staged-lookup and its send.
        self._staged: _StagedBurst | None = None

    # -- pre-key -------------------------------------------------------------------------------

    async def prepare(
        self,
        sockets,
        frame_id: str,
        payload: bytes,
        sample_rate: float,
        params: dict[str, object],
    ) -> bool:
        """Validate + BUILD the burst while the station is still SAFE, and acknowledge it.

        Emits ``tx_prepared`` (the orchestrator's licence to key) or ``tx_prepare_failed`` (the
        orchestrator must NOT key). Never raises: a rejected uplink is a routine outcome that must
        leave the app alive and the station unkeyed, not kill the control loop.
        """
        async with self._lock:
            # A failed stage must not leave a PREVIOUS burst armed — the orchestrator would then key
            # for frame B and radiate frame A. Drop it before we try.
            self._staged = None
            try:
                # ROUND 12: pass the TX upsample factor so validate_uplink bounds the HARDWARE-rate
                # allocation (samples * factor), not just the modem-rate one. 1 for file/bench I/O.
                hw_factor = int(getattr(self._io, "tx_upsample_factor", 1) or 1)
                modem_samples = await asyncio.to_thread(
                    validate_uplink, payload, sample_rate, params, hardware_factor=hw_factor
                )
                # (3a) build the COMPLETE final hardware-rate flat CS16 buffer HERE, cold:
                # modulate -> Doppler (stage-time offset) -> resample -> pack -> finite/non-empty.
                cs16 = await asyncio.to_thread(
                    self._build_final_cs16, payload, sample_rate, params, hw_factor
                )
            except UplinkRejected as e:
                _log.error("bidir TX: uplink REJECTED pre-key (%s): %s", e.code, e.detail)
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "tx_prepare_failed",
                        "frame_id": frame_id,
                        "code": e.code,
                        "detail": e.detail,
                    },
                )
                return False
            except Exception as e:  # noqa: BLE001 — a build failure must not kill the app
                _log.exception("bidir TX: uplink build failed pre-key")
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "tx_prepare_failed",
                        "frame_id": frame_id,
                        "code": "build-failed",
                        "detail": repr(e),
                    },
                )
                return False

            complex_samples = int(cs16.size // 2)  # hardware-rate complex samples now cached
            self._staged = _StagedBurst(
                frame_id=frame_id,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                cs16=cs16,
                complex_samples=complex_samples,
                modem_samples=int(modem_samples),
            )
            # ROUND 11 (re-check, D5): report the RF DURATION of the burst, so the orchestrator can
            # size its completion deadline to THIS burst instead of a second, independent constant.
            # duration = modem samples / modem rate (== payload_bits/baud; upsampling changes the
            # sample count, not the RF time).
            duration_s = (modem_samples / sample_rate) if sample_rate > 0 else 0.0
            await send_event(
                sockets.status_writer,
                {
                    "event": "tx_prepared",
                    "frame_id": frame_id,
                    # hardware-rate complex samples that will actually be streamed (Doppler +
                    # resample already applied); predicted_samples stays the modem-rate count.
                    "samples": complex_samples,
                    "predicted_samples": int(modem_samples),
                    "payload_bytes": len(payload),
                    "payload_sha256": self._staged.payload_sha256,
                    "sample_rate": int(sample_rate),
                    "duration_s": round(duration_s, 3),
                },
            )
            return True

    def _build_final_cs16(
        self, payload: bytes, sample_rate: float, params: dict[str, object], hw_factor: int
    ) -> np.ndarray:
        """(3a) The COMPLETE pre-key DSP, run cold in a worker thread: modulate the uplink,
        apply the uplink Doppler pre-compensation (using the app's last-known ``set_doppler``
        offset, snapshotted now), resample to the hardware rate, and pack to ONE contiguous flat
        CS16 buffer. Validates non-empty + finite BEFORE returning. Raises :class:`UplinkRejected`
        on an empty or non-finite waveform so ``prepare`` answers ``tx_prepare_failed`` (never a
        keyed failure). Nothing here touches hardware, and the keyed window re-does NONE of it."""
        iq = build_uplink_iq(payload, sample_rate, params)  # modem-rate complex64
        # Doppler at STAGE time (single offset for the short burst): downlink offset -> uplink
        # pre-comp, applied at the modem rate, exactly as the emission-time NCO used to.
        tx_dop = tx_doppler_hz(self._doppler["hz"], self._downlink_hz, self._uplink_hz)
        iq = apply_nco(iq, tx_dop, sample_rate)
        hw = upsample_burst(iq, int(hw_factor))  # resample to the device hardware rate
        if hw.size == 0:
            raise UplinkRejected("empty-waveform", "the built waveform is empty — nothing to send")
        if not np.all(np.isfinite(hw.view(np.float32))):
            raise UplinkRejected(
                "non-finite-waveform", "the built waveform has non-finite IQ (NaN/Inf)"
            )
        return to_cs16(hw)

    # -- post-key ------------------------------------------------------------------------------

    async def transmit(self, sockets, frame_id: str) -> int:
        """Burst the ALREADY-BUILT samples. The PA is keyed when this runs.

        R-16: ``transmit_started`` fires only when the sink ACCEPTS its first sample (bridged
        threadsafe from the burst worker); ``transmit_complete`` ALWAYS fires and carries the
        accepted count plus an explicit bounded outcome. Exceptions are logged, not propagated, so
        the orchestrator's half-duplex loop never stalls on them.

        ROUND 10: if no matching burst was staged, this REFUSES. It does not build one. Building
        here is the hazard the handshake exists to remove, so the fallback would reintroduce it on
        exactly the path (a missed/failed prepare) where it is most likely to fire.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()

            staged = self._staged
            # One-shot: consume the stage whatever happens, so a stale burst can never be re-flown
            # against a later, different frame_id.
            self._staged = None

            if staged is None or staged.frame_id != frame_id:
                have = "nothing" if staged is None else f"frame {staged.frame_id!r}"
                detail = (
                    f"no staged IQ for frame {frame_id!r} (staged: {have}) — the PA is KEYED and "
                    f"this app will NOT build a burst while keyed; prepare_transmit first"
                )
                _log.error("bidir TX: %s", detail)
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "transmit_complete",
                        "frame_id": frame_id,
                        "samples": 0,
                        "outcome": "error",
                        "detail": detail,
                    },
                )
                return 0

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
                # (3a) NO DSP in the keyed window: the cached buffer is already the FINAL flat CS16
                # (Doppler applied, resampled, packed) built cold at stage time. The keyed window
                # only opens the TX stream and writes these bytes — no modulation, no rotation, no
                # resample, no allocation.
                result = await asyncio.to_thread(
                    lambda: self._io.transmit_burst(
                        staged.cs16, on_first_accept=_first_accept, should_abort=self._should_abort
                    )
                )
                accepted = int(result.accepted)
                outcome = result.outcome
                detail = result.detail
            except Exception as e:  # noqa: BLE001 — must still emit transmit_complete
                _log.exception("bidir TX: uplink send failed")
                detail = repr(e)
            finally:
                self.tx_active.clear()
                await send_event(
                    sockets.status_writer,
                    {
                        "event": "transmit_complete",
                        "frame_id": frame_id,
                        "samples": accepted,
                        "outcome": outcome,
                        "detail": detail,
                    },
                )
            return accepted

    @property
    def has_staged_burst(self) -> bool:
        return self._staged is not None


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

    # Backpressure put, interruptible on stop — shared X-02 primitive (the
    # cubesat dsp engine uses the same one; one implementation, one test).
    _put = make_backpressure_put(queue, loop, stop_requested)

    # Audit round 2 (silent-success class): the reader used to swallow its own death —
    # the except logged, and the finally pushed the SAME `None` terminator a clean EOF
    # uses, so run_rx returned NORMALLY after the SDR failed. The pass then completed
    # with zero frames and no error event: a dead radio was indistinguishable from a
    # quiet sky. The failure is now captured and re-raised once the queue drains.
    reader_error: list[BaseException] = []

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
        except Exception as e:
            _log.exception("bidir RX: IQ source error")
            reader_error.append(e)
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
            # R-13: phase ADVANCES across chunks (the old ph[-1] carry
            # repeated the last sample's phase at every boundary).
            chunk, nco_phase = apply_nco_chunk(chunk, doppler["hz"], sample_rate, nco_phase)
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
    # The reader's death must NOT look like a clean EOF. It used to: the except logged and
    # the finally pushed the SAME `None` terminator an exhausted stream uses, so run_rx
    # returned normally after the SDR failed and the pass completed with zero frames — a
    # dead radio was indistinguishable from a quiet sky. Capturing the exception was not
    # enough (audit): it has to be RAISED, after the queue has drained so nothing already
    # received is lost.
    if reader_error:
        raise EngineFailure(
            f"bidir RX: the IQ source DIED mid-pass — {reader_error[0]!r}. Frames received "
            f"before the failure were emitted; the rest of the pass captured NOTHING."
        ) from reader_error[0]


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
        from _soapy import (
            apply_corrections,
            configure_soapy_source,
            merge_sdr_params,
            merge_sdr_params_tx,
            sdr_env,
        )
        from _soapy_tx import named_tx_gains
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_CS16, SOAPY_SDR_RX, SOAPY_SDR_TX

        # (3c) TX drive is a named per-element gain (PAD) ONLY — the overall setGain overload
        # aborts SoapyXTRX. Resolve + REQUIRE it BEFORE opening the device, so a misconfigured
        # (deaf / would-crash) TX fails closed COLD, not at key time. Raises TxGainConfigError.
        self._tx_settings = merge_sdr_params_tx(params)
        self._tx_named = named_tx_gains(self._tx_settings)
        self._sd = SoapySDR
        self._RX, self._TX, self._CF32 = SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CF32
        # TX streams in the bench-proven CS16 wire format (the XTRX probe shape); RX
        # stays CF32. The uplink burst is packed to flat CS16 before keying.
        self._CS16 = SOAPY_SDR_CS16
        self._lock = threading.Lock()
        self.tx_active = threading.Event()
        rate = resolve_sample_rate(args, params)  # integer sps — must match the modulator's IQ rate
        self._rate = rate
        dev = SoapySDR.Device(args.sdr_args)
        self._dev = dev
        env = sdr_env()
        merged = merge_sdr_params(params)
        # R-14: the DEVICE runs at a supported rate — the smallest INTEGER
        # multiple of the modem rate above the hardware floor (XTRX-class
        # can't stream ~96 kHz directly). RX decimates by the exact factor
        # (stateful); TX bursts are upsampled by it before the write. The
        # readback is validated: a silently-clamped rate desynchronizes the
        # modem, so a mismatch fails the spawn closed (R-11).
        hw_rate, factor = hardware_rate_for(rate, env["capture_rate_hz"])
        self._hw_rate, self._factor = hw_rate, factor
        # ROUND 12: exposed so validate_uplink (pre-key) can bound the HARDWARE-rate allocation.
        self.tx_upsample_factor = factor
        self._decim = StreamingDecimator(factor)
        # RX on the downlink.
        dev.setSampleRate(SOAPY_SDR_RX, 0, hw_rate)
        require_sample_rate(dev, SOAPY_SDR_RX, 0, hw_rate)
        dev.setFrequency(SOAPY_SDR_RX, 0, float(args.center_freq_hz))
        with contextlib.suppress(Exception):
            dev.setBandwidth(SOAPY_SDR_RX, 0, hw_rate)
        # TX on the uplink freq (split-freq). params['uplink_hz'] falls back to the RX centre.
        uplink_hz = float(params.get("uplink_hz", args.center_freq_hz) or args.center_freq_hz)
        self._uplink_hz = uplink_hz
        dev.setSampleRate(SOAPY_SDR_TX, 0, hw_rate)
        require_sample_rate(dev, SOAPY_SDR_TX, 0, hw_rate)
        dev.setFrequency(SOAPY_SDR_TX, 0, uplink_hz)
        with contextlib.suppress(Exception):
            dev.setBandwidth(SOAPY_SDR_TX, 0, hw_rate)

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
        # The RX-oriented ``merged`` params (GS_SDR_ANTENNA/GS_SDR_GAINS — e.g. antenna "LNAW",
        # gain elements LNA/TIA/PGA) are NEVER pushed onto the TX endpoint: those names are
        # RX-only on LMS7/XTRX-class devices and setAntenna/setGain would RAISE, aborting the
        # shared device init and losing the DOWNLINK too (not just the uplink). R-22 + (3c): TX
        # gets its OWN explicit NAMED gains (resolved + required above); the overall setGain
        # overload is never used, so only the named PAD element drives the PA.
        tx_only: dict[str, object] = {"sdr_gains": self._tx_named}
        if isinstance(self._tx_settings.get("sdr_antenna"), str):
            tx_only["sdr_antenna"] = self._tx_settings["sdr_antenna"]
        configure_soapy_source(_EP(dev, SOAPY_SDR_TX), tx_only, default_gain_db=None)
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
        _log.info(
            "bidir SoapySDR: rx=%.0f tx=%.0f modem=%.0f hw=%.0f (x%d)",
            args.center_freq_hz, uplink_hz, rate, hw_rate, factor,
        )

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
                # R-14: hardware-rate capture → modem-rate chunks (stateful,
                # exact integer factor).
                out = self._decim.process(buff[: sr.ret].copy())
                if len(out):
                    yield out
            elif sr.ret in (SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW):
                continue
            else:
                _log.warning("bidir RX readStream ret=%d", sr.ret)

    def transmit_burst(self, cs16: np.ndarray, *, on_first_accept=None, should_abort=None):
        """Write ONE already-final flat CS16 uplink burst. (3a) NO DSP here — the resample,
        Doppler and CS16 pack all ran PRE-KEY in the staging, so the keyed window only opens
        the TX stream and writes the cached buffer (no modulation/rotation/resample/allocation).

        Conforms to the XTRX bench-probe shape (tools/probe_soapy_tx_write.py):
        ``setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16)`` with NO explicit channel list; (3d) the
        stream MTU is queried BEFORE activateStream; there is NO sleep between activate and the
        first write (untimed XTRX buffers go stale); write_burst uses the 3-arg call + one bounded
        readStreamStatus check (no flags/END_BURST/timed writes).

        (3h) Neither the (absent) write timeout nor the per-burst deadline can interrupt a HUNG
        native writeStream — the deadline only bounds time BETWEEN writes; a genuinely hung driver
        call is bounded only by the orchestrator's keyed-window backstop (gs-client PA-off).

        R-15 half-duplex: RX is DEACTIVATED before TX activates. (3e) If RX cannot be cleanly
        deactivated, TX is REFUSED — we do not key a stream we could not break from RX. (3f) On
        every exit each cleanup step (TX deactivate, TX close, RX restore) is attempted
        INDEPENDENTLY so one failing does not skip the others. RX is ALWAYS restored after a burst
        that opened the TX stream, even on a write error.

        Returns a ``_soapy_tx.BurstResult`` (R-16)."""
        from _soapy_tx import query_tx_mtu, write_burst

        buf = np.ascontiguousarray(np.asarray(cs16, dtype=np.int16))
        n_complex = int(buf.size // 2)
        if should_abort is not None and should_abort():
            return BurstResult(accepted=0, total=n_complex, outcome="cancelled",
                               detail="aborted before key")
        if buf.size == 0:  # (3g) empty output is an ERROR, refused before touching T/R
            return BurstResult(accepted=0, total=0, outcome="error",
                               detail="empty burst buffer — refusing to key")
        self.tx_active.set()
        try:
            with self._lock:
                # (3e) break-before-make: if RX cannot be cleanly deactivated, REFUSE TX — do not
                # key a TX stream we could not break from RX. RX is left as-is (unknown state).
                try:
                    self._dev.deactivateStream(self._rx_stream)
                except Exception as e:  # noqa: BLE001
                    _log.error("bidir TX: RX deactivate FAILED; refusing to key: %r", e)
                    return BurstResult(accepted=0, total=n_complex, outcome="error",
                                       detail=f"RX break failed; refusing TX: {e!r}")
                tx = None
                try:
                    # CS16 wire format, NO explicit [0] channel list — the probe-verified shape.
                    tx = self._dev.setupStream(self._TX, self._CS16)
                    # (3d) query MTU BEFORE activateStream (positive-MTU-required stays).
                    mtu = query_tx_mtu(self._dev, tx)
                    self._dev.activateStream(tx)  # no post-activate sleep (probe rule)
                    # Deadline bounds time BETWEEN writes only (see 3h above), not a hung call.
                    deadline_s = max(5.0, 3.0 * n_complex / max(1.0, self._hw_rate))
                    result = write_burst(
                        self._dev,
                        tx,
                        buf,
                        mtu=mtu,
                        deadline_s=deadline_s,
                        on_first_accept=on_first_accept,
                        should_abort=should_abort,
                    )
                    if not result.complete:
                        _log.warning(
                            "bidir TX burst incomplete: %d/%d (%s: %s)",
                            result.accepted, result.total, result.outcome, result.detail,
                        )
                except Exception as e:  # noqa: BLE001 — a truthful error result, still clean up
                    _log.exception("bidir TX: stream/write failed")
                    result = BurstResult(
                        accepted=0, total=n_complex, outcome="error", detail=repr(e)
                    )
                finally:
                    # (3f) each cleanup step independent — one failing must not skip the others.
                    if tx is not None:
                        with contextlib.suppress(Exception):
                            self._dev.deactivateStream(tx)
                        with contextlib.suppress(Exception):
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
    tx = _TxController(
        io, sample_rate=sample_rate, doppler=doppler, downlink_hz=downlink_hz,
        uplink_hz=uplink_hz, should_abort=stop_requested.is_set,
    )
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
            # ROUND 10: `ready` means the DOWNLINK is live. It does NOT mean any uplink is flyable —
            # nothing has been framed or modulated at this point, and it cannot be, because the
            # payload arrives per-burst. This flag is how the app says so out loud: the orchestrator
            # must `prepare_transmit` and receive `tx_prepared` BEFORE it keys the PA. An
            # orchestrator that ignores it and keys anyway gets a refusal, not a burst.
            "tx_prepare_required": True,
        },
    )

    async def _on_start(_cmd: dict[str, object]) -> None:
        await send_event(sockets.status_writer, {"event": "started"})

    stop_reason = {"value": "command"}

    async def _on_stop(cmd: dict[str, object]) -> None:
        stop_requested.set()
        stop_reason["value"] = str(cmd.get("reason", "command"))

    async def _on_set_doppler(cmd: dict[str, object]) -> None:
        off = cmd.get("offset_hz", 0)
        if isinstance(off, (int, float)) and not isinstance(off, bool):
            doppler["hz"] = float(off)

    async def _on_prepare_transmit(cmd: dict[str, object]) -> None:
        """PRE-KEY. Resolve the payload, validate it, build the IQ, cache it, acknowledge.

        This is the only place the uplink is read from disk, framed and modulated — and it runs with
        the antenna on the LNA and the PA cold. A rejection here costs a `tx_prepare_failed` event
        and nothing else: no relay moves, no PA is energized, no forced disarm.
        """
        frame_id = str(cmd.get("frame_id", "") or "")
        try:
            payload = _uplink_payload_from_cmd(cmd, args, params)
        except Exception as e:  # noqa: BLE001 — a bad path/blob is a rejection, not a crash
            _log.exception("bidir TX: could not resolve the uplink payload")
            await send_event(
                sockets.status_writer,
                {
                    "event": "tx_prepare_failed",
                    "frame_id": frame_id,
                    "code": "payload-unresolvable",
                    "detail": repr(e),
                },
            )
            return
        await tx.prepare(sockets, frame_id, payload, sample_rate, params)

    tx_tasks: list[asyncio.Task[None]] = []

    async def _on_transmit(cmd: dict[str, object]) -> None:
        # POST-KEY. Handles both "transmit_frame" (inline bytes_b64) and "transmit_payload_file".
        # The orchestrator sends "transmit_frame" for BOTH (ControlWriter.transmit_payload_file
        # writes cmd="transmit_frame" with a payload_file field); the 2nd key is future-proofing.
        #
        # ROUND 10: this NO LONGER resolves or builds anything. The PA is keyed when it runs, and
        # the payload was staged and proven flyable before the key. An unstaged frame is refused.
        #
        # ROUND 11 (P0-4): the burst runs as a BACKGROUND TASK, not awaited inline. run_command_loop
        # dispatches handlers serially — if this awaited the whole burst (seconds to minutes of RF),
        # the loop could not read the NEXT command, so a `stop` on the control socket would not be
        # dequeued and `stop_requested` would not be set until the burst had already finished. The
        # abort is fully wired (write_burst polls should_abort between chunks); it was simply
        # unreachable. Spawning frees the loop, so `stop` lands and aborts the
        # burst mid-flight. `_TxController._lock` still serializes overlapping bursts; `tx.transmit`
        # swallows its own errors and always emits transmit_complete, so the task never escapes an
        # exception (a done-callback logs the impossible case).
        frame_id = str(cmd.get("frame_id", "") or "")
        task = asyncio.create_task(tx.transmit(sockets, frame_id), name=f"bidir-tx-{frame_id}")
        tx_tasks.append(task)
        task.add_done_callback(_on_tx_task_done)

    def _on_tx_task_done(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(ValueError):
            tx_tasks.remove(task)
        if not task.cancelled() and task.exception() is not None:
            _log.error("bidir TX: burst task raised (unexpected): %r", task.exception())

    rx_task = asyncio.create_task(
        run_rx(args, sockets, params, io, stop_requested=stop_requested, doppler=doppler, tx=tx),
        name="bidir-rx",
    )
    # R-11 / audit: a dead RX engine must FAIL the pass, not linger behind a live command
    # loop that keeps cheerfully answering the orchestrator. This app was the ONE engine
    # that never wired the watcher, which is why a dead SDR still looked like a clean stop.
    watch_engine_death(rx_task, sockets.status_writer, sockets.control_reader, stop_requested)
    handlers = {
        "start": _on_start,
        "stop": _on_stop,
        "set_doppler": _on_set_doppler,
        "prepare_transmit": _on_prepare_transmit,
        "transmit_frame": _on_transmit,
        "transmit_payload_file": _on_transmit,
    }
    async def _shutdown_engine() -> None:
        """Idempotent engine teardown: settle the RX task and any in-flight TX burst, close the
        device. ROUND 11 (P0-4): a stop that arrives mid-burst sets stop_requested, which the burst
        polls and aborts on; we then await the burst task here so its transmit_complete is emitted
        and the device is not closed out from under it."""
        stop_requested.set()
        await asyncio.gather(rx_task, *tx_tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            io.close()

    try:
        reason = await run_command_loop(sockets.control_reader, handlers, sockets.status_writer)
        # P0-08: engine teardown BEFORE the explicit stopped ack; then exit 0.
        # EOF is transport loss — no ack, exit nonzero.
        await _shutdown_engine()
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
            await send_event(
                sockets.status_writer,
                {"event": "stopped", "reason": stop_reason["value"]},
            )
            return 0
        _log.warning("control EOF without stop — transport loss; exiting nonzero (P0-08)")
        return 1
    finally:
        await _shutdown_engine()
        await sockets.aclose()


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

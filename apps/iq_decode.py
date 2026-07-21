#!/usr/bin/env python3
"""Post-pass decode of a recorded ``.cf32`` capture with the NON-live (heavier) framings.

Runs AFTER the pass, decoupled from the flowgraph — free CPU, no 30 s stop budget (the same model
as ``iq_views``). The live RX engines already decode the LIGHT framings (ax25 + endurosat) in real
time; this sweeps the recorded IQ for the framings they do NOT run live
(``framings.POST_PASS_FRAMINGS`` — the other CRC-gated local link layers, currently ``ccsds_tm``),
so a pass that carried one of those still yields frames. ``kiss`` is NOT swept by default (it has
no integrity check, so a blind whole-pass sweep would emit noise "frames") — request it explicitly
if a pass is known KISS. Labels with an available native profile are decoded through the shared
streaming registry; labels that remain planned (USP, AX100, …) still require the GNU Radio engine.

**Doppler.** The recorded ``.cf32`` is RAW — captured BEFORE the live Doppler NCO — so it carries
the full pass Doppler swing (±~9 kHz at 400 MHz LEO), which is larger than a burst CFO estimate can
pull back. We do NOT try to. Doppler is DETERMINISTIC: gs-orbitd re-propagates the pass's TLE over
its time window to the exact same track it drove live. So gs-client hands us that track
(``doppler_track`` — ``[(t_seconds_from_capture_start, offset_hz), …]``, sampled from gs-orbitd),
and we de-rotate the raw IQ with it — reproducing precisely what the live decoder saw — before
demod. Nothing about the track is persisted anywhere; it is regenerated on demand. With NO track
(a lab/file capture with negligible Doppler) we fall back to per-window CFO, which is best-effort.
The track assumes a RAW capture — as the dsp / bidir RX engines record it (pre-NCO). The gnuradio
engine retunes the SDR source in HARDWARE, so its ``.cf32`` is already Doppler-corrected; feeding a
track there would double-correct, so post-pass decode targets the raw-recording engines (gs-client
only pairs the track with those). The de-rotation is per-window, so memory stays bounded to one
window (a whole-capture de-rotation would materialise multi-GB temporaries on a long pass).

Pure numpy — no GNU Radio, no gs-orbitd dependency (gs-client owns the gs-orbitd query and passes
the sampled track) — so it runs anywhere and is unit-testable on the dev box.

License: GPLv3 (see ../COPYING).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import framings
import modem
import numpy as np
from native_framing import (
    AfskConfig,
    BpskConfig,
    FrameResult,
    IntegrityStatus,
    SampleClock,
    SymbolInput,
    build_decoder,
    demodulate_afsk,
    demodulate_bpsk,
    resolve_profile,
)
from native_framing.modem_matrix import RxExecution, plan_native_rx_pairing
from native_framing.output import utc_from_sample_offset
from native_framing.runtime_queue import BoundedQueue, require_lossless

from gfsk_ax25 import gfsk

log = logging.getLogger("iq_decode")
_DEFAULT_SYMBOL_RATE_HZ = 9600.0
_DEFAULT_WINDOW_S = 1.0  # short enough that any residual offset is ~constant across the window
_GNU_RADIO_DRAIN_PERIOD_S = 0.02


def _optional_bool(
    name: str, explicit: bool | None, parameters: Mapping[str, object]
) -> bool | None:
    if explicit is not None or name not in parameters:
        return explicit
    value = parameters[name]
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _optional_float(
    name: str,
    explicit: float | None,
    parameters: Mapping[str, object],
    default: float,
) -> float:
    value = explicit if explicit is not None else parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _decoder_parameters(profile, supplied: Mapping[str, object]) -> dict[str, object]:
    """Select this profile's parameters from the directive-wide waveform map.

    Protobuf ``Struct`` represents every JSON number as ``double``.  Preserve strict
    profile validation while restoring integral values for parameters declared as
    integers; otherwise a valid backend ``frame_size: 256`` arrives here as ``256.0``
    and is rejected before the decoder can be constructed.
    """

    selected: dict[str, object] = {}
    for key, spec in profile.parameters.items():
        if key not in supplied:
            continue
        value = supplied[key]
        if spec.value_type is int and isinstance(value, float) and value.is_integer():
            value = int(value)
        selected[key] = value
    return selected


def _symbols_for_profile(bits: np.ndarray, symbol_input: SymbolInput) -> np.ndarray:
    """Adapt hard GFSK decisions to the native profile's declared symbol convention.

    The offline demodulator currently exposes hard decisions.  Soft-input deframers can
    still consume those decisions at unit confidence, with the package-wide convention
    ``positive => bit 1``.  This is less informative than true discriminator values but
    is type-correct and deterministic; it must never pass 0/1 values as soft amplitudes.
    """

    hard = np.asarray(bits, dtype=np.uint8)
    if symbol_input is SymbolInput.HARD_BITS:
        return hard
    return hard.astype(np.float64) * 2.0 - 1.0


def _afsk_tones(parameters: Mapping[str, object]) -> tuple[float, float]:
    """Return mark/space audio tones from the JSON-safe waveform map."""

    value = parameters.get("tones_hz", (1_200.0, 2_200.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("tones_hz must contain exactly two numeric frequencies")
    if any(isinstance(item, bool) for item in value):
        raise ValueError("tones_hz must contain exactly two numeric frequencies")
    try:
        one_hz, zero_hz = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tones_hz must contain exactly two numeric frequencies"
        ) from exc
    return one_hz, zero_hz


def _fm_discriminator(iq: np.ndarray) -> np.ndarray:
    """Recover bounded real FM audio while retaining the IQ sample clock."""

    samples = np.asarray(iq)
    if samples.ndim != 1 or not np.iscomplexobj(samples):
        raise ValueError("AFSK replay requires one-dimensional complex IQ")
    audio = np.zeros(samples.size, dtype=np.float64)
    if samples.size > 1:
        audio[1:] = np.angle(samples[1:] * np.conjugate(samples[:-1]))
        audio[0] = audio[1]
    return audio


def _derotate_doppler(
    iq: np.ndarray, sample_rate_hz: float, track: list[tuple[float, float]]
) -> np.ndarray:
    """De-rotate ``iq`` by a Doppler ``track`` (``[(t_s, offset_hz), …]`` from gs-orbitd),
    reproducing the live NCO. The offset at each sample is linearly interpolated over the track and
    the phase accumulated continuously, so the whole pass is brought near DC exactly as it was live
    (the live NCO applies ``exp(-j·2π·offset·n/fs)`` per chunk; this is its continuous form)."""
    if not track or len(iq) == 0:
        return np.asarray(iq, dtype=np.complex64)
    ts = np.asarray([float(t) for t, _ in track], dtype=np.float64)
    offs = np.asarray([float(o) for _, o in track], dtype=np.float64)
    t = np.arange(len(iq), dtype=np.float64) / float(sample_rate_hz)
    off_per_sample = np.interp(t, ts, offs)  # Hz at each sample (flat outside the track ends)
    phase = -2.0 * np.pi * np.cumsum(off_per_sample) / float(sample_rate_hz)
    return (np.asarray(iq, dtype=np.complex64) * np.exp(1j * phase)).astype(np.complex64)


def decode_capture(
    cf32: str | Path,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float = _DEFAULT_SYMBOL_RATE_HZ,
    framings_to_try: tuple[str, ...] = framings.POST_PASS_FRAMINGS,
    doppler_track: list[tuple[float, float]] | None = None,
    capture_start_unix_s: float = 0.0,
    window_s: float = _DEFAULT_WINDOW_S,
    framing_parameters: Mapping[str, object] | None = None,
    modulation: str | None = None,
    differential: bool | None = None,
    manchester: bool | None = None,
    mod_index: float | None = None,
    bt: float | None = None,
    native_evaluation: bool = False,
) -> list[dict]:
    """Doppler-de-rotate (from ``doppler_track``) + windowed demod + deframe of ``cf32`` with each
    framing in ``framings_to_try``. Returns the decoded frame records (also appended to
    ``<pass>/frames.jsonl`` when any are found)."""
    path = Path(cf32)
    supplied_parameters = dict(framing_parameters or {})
    modulation_value = modulation if modulation is not None else supplied_parameters.get(
        "modulation", "gfsk"
    )
    if not isinstance(modulation_value, str) or not modulation_value.strip():
        raise ValueError("modulation must be a non-empty string")
    differential = _optional_bool("differential", differential, supplied_parameters)
    manchester = _optional_bool("manchester", manchester, supplied_parameters)
    mod_index_value = _optional_float("mod_index", mod_index, supplied_parameters, 0.5)
    bt_value = _optional_float("bt", bt, supplied_parameters, 0.5)
    if not framings_to_try:
        return []
    if not path.exists():
        log.warning("iq_decode: %s not found", path)
        return []
    # Prefer the recording's own sidecar (the TRUE rate the engine used — it may have widened the
    # channel for a high-baud bird) over the passed-in rate. Mirrors iq_views.
    meta = path.with_name(path.name + ".json")
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            sample_rate_hz = float(d.get("sample_rate_hz", sample_rate_hz))
        except (OSError, ValueError, TypeError):
            log.warning("iq_decode: ignoring unreadable sidecar %s", meta.name)
    n_samp = path.stat().st_size // 8  # 8 B/complex64; floor a torn write to whole samples
    if n_samp < 1:
        log.warning("iq_decode: %s has no samples", path)
        return []
    iq = np.memmap(path, dtype=np.complex64, mode="r", shape=(n_samp,))
    sample_clock = SampleClock(sample_rate_hz, symbol_rate_hz)
    modulation_spec = modem.modulation_spec(modulation_value)
    if modulation_spec is None:
        raise ValueError(f"unknown post-pass modulation: {modulation_value!r}")
    binary_fsk = modulation_spec.family == "fsk" and modulation_spec.order == 2
    binary_afsk = modulation_spec.family == "afsk"
    binary_psk = (
        modulation_spec.family == "psk"
        and modulation_spec.order == 2
        and not modulation_spec.offset
    )
    if not binary_fsk and not binary_afsk and not binary_psk:
        raise ValueError(
            f"no native post-pass IQ replay for modulation {modulation_spec.kind!r}"
        )
    psk_differential = (
        modulation_spec.differential if differential is None else bool(differential)
    )
    psk_manchester = (
        modulation_spec.manchester if manchester is None else bool(manchester)
    )
    # We do NOT de-rotate the whole capture up front: that materialises a complex128 phase array +
    # exp over the ENTIRE pass (multi-GB on a long capture → OOM on a constrained station). Instead
    # each window is sliced from the memmap and de-rotated locally, bounding memory to one window.
    have_track = bool(doppler_track)
    if not have_track:
        log.warning("iq_decode: no Doppler track — falling back to per-window CFO (best-effort)")
    win = max(1, int(sample_rate_hz * window_s))
    # Overlap windows by ~half so a frame straddling a window boundary is fully contained in the
    # next window (a non-overlapping sweep silently drops boundary frames). A seen-set dedups the
    # re-decoded overlap frames by (framing, payload) — a genuine exact-duplicate payload is also
    # collapsed, acceptable for a post-pass completeness sweep (CCSDS frame counters differ anyway).
    step = max(1, win - win // 2)
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for off in range(0, n_samp, step):
        seg = np.asarray(iq[off : off + win])  # bounded window materialised from the memmap here
        if have_track:
            # Shift the track to window-local time (_derotate_doppler times from 0), so only THIS
            # window's samples are rotated — no whole-capture temporaries.
            t_off = off / sample_rate_hz
            wtrack = [(t - t_off, o) for t, o in doppler_track or []]
            seg = _derotate_doppler(seg, sample_rate_hz, wtrack)
        try:
            if binary_fsk:
                hard_symbols = gfsk.demodulate_capture(
                    seg,
                    sample_rate_hz,
                    symbol_rate_hz=symbol_rate_hz,
                    mod_index=mod_index_value,
                    bt=bt_value,
                    # With the track applied the window is already near DC → correct only the
                    # small residual. Without a track, this per-window CFO is the (weaker) sole
                    # correction.
                    correct_cfo=True,
                    # Max-eye sampling (NOT Gardner): demodulate_capture is tuned for it —
                    # Gardner's timing recovery diverges on a capture, so recover_timing=True
                    # yields no frames.
                    recover_timing=False,
                )
                soft_symbols = hard_symbols.astype(np.float64) * 2.0 - 1.0
                sample_for_symbol = sample_clock.sample_offset_for_symbol
            elif binary_afsk:
                one_hz, zero_hz = _afsk_tones(supplied_parameters)
                replay = demodulate_afsk(
                    _fm_discriminator(seg),
                    AfskConfig(
                        sample_rate_hz,
                        symbol_rate_hz,
                        one_hz=one_hz,
                        zero_hz=zero_hz,
                    ),
                )
                hard_symbols = replay.hard_bits
                soft_symbols = replay.soft_symbols
                sample_for_symbol = replay.sample_offset
            else:
                replay = demodulate_bpsk(
                    seg,
                    BpskConfig(
                        sample_rate_hz,
                        symbol_rate_hz,
                        differential=psk_differential,
                        manchester=psk_manchester,
                    ),
                )
                hard_symbols = replay.hard_bits
                soft_symbols = replay.soft_symbols
                sample_for_symbol = replay.sample_offset
        except Exception:  # noqa: BLE001 — one bad window must not abort the whole sweep
            log.exception("iq_decode: demod failed on window @%d", off)
            continue
        if not len(hard_symbols):
            continue
        for name in framings_to_try:
            profile = resolve_profile(name)
            candidates: list[tuple[bytes, str, int | None, FrameResult | None]]
            try:
                if profile is not None and profile.decoder_available:
                    pairing = plan_native_rx_pairing(
                        name,
                        modulation_spec.kind,
                        sample_rate_hz=sample_rate_hz,
                        symbol_rate_hz=symbol_rate_hz,
                        capture_rate_hz=sample_rate_hz,
                        execution=RxExecution.POST_PASS,
                        evaluation=native_evaluation,
                    )
                    if not pairing.accepted:
                        log.warning(
                            "iq_decode: rejected native pairing for %s: %s",
                            name,
                            pairing.reason,
                        )
                        continue
                    decoder = build_decoder(
                        name, _decoder_parameters(profile, supplied_parameters)
                    )
                    symbols = (
                        hard_symbols
                        if profile.symbol_input is SymbolInput.HARD_BITS
                        else soft_symbols
                    )
                    native = decoder.push(symbols) + decoder.flush()
                    candidates = [
                        (
                            result.payload,
                            result.canonical_framing,
                            result.source_start,
                            result,
                        )
                        for result in native
                    ]
                else:
                    frames, matched = framings.deframe(hard_symbols, name)
                    candidates = [(body, matched or name, None, None) for body in frames]
            except Exception:  # noqa: BLE001 — a deframer bug must not abort the sweep
                log.exception("iq_decode: %s deframe failed @%d", name, off)
                continue
            for body, matched_name, symbol_offset, native_result in candidates:
                key = (matched_name, body.hex())
                if key in seen:  # a boundary frame re-decoded in the overlapping next window
                    continue
                seen.add(key)
                if symbol_offset is None:
                    source_sample_offset = off
                    offset_kind = "window_start"
                else:
                    source_sample_offset = off + sample_for_symbol(symbol_offset)
                    offset_kind = "demodulated_symbol_estimate"
                record = {
                    "framing": matched_name,
                    "len": len(body),
                    "crc_ok": (
                        native_result.integrity.value == "passed"
                        if native_result is not None
                        else True
                    ),
                    "payload_hex": body.hex(),
                    "post_pass": True,
                    "source_sample_offset": source_sample_offset,
                    "source_offset_kind": offset_kind,
                }
                if native_result is not None:
                    record.update(
                        {
                            "source_start": native_result.source_start,
                            "source_end": native_result.source_end,
                            "source_sample_end_offset": off
                            + sample_for_symbol(native_result.source_end),
                            "integrity": native_result.integrity.value,
                            "polarity": native_result.polarity.value,
                            "sync_distance": native_result.sync_distance,
                            "corrected_symbols": native_result.corrected_symbols,
                            "metadata": dict(native_result.metadata),
                        }
                    )
                if capture_start_unix_s > 0:
                    pass_start = datetime.fromtimestamp(capture_start_unix_s, tz=timezone.utc)
                    frame_time = utc_from_sample_offset(
                        pass_start, source_sample_offset, sample_rate_hz
                    )
                    record["ts"] = round(frame_time.timestamp(), 3)
                    record["timestamp"] = frame_time.isoformat().replace("+00:00", "Z")
                records.append(record)
    if records:
        _append_frames(path, records)
    log.info(
        "iq_decode: %d post-pass frame(s) from %s (framings=%s, doppler=%s)",
        len(records),
        path.name,
        ",".join(framings_to_try),
        "track" if have_track else "cfo-only",
    )
    return records


def _record_gnuradio_native(
    result: FrameResult,
    *,
    decoder: str,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    capture_start_unix_s: float,
) -> dict:
    """Convert one streaming decoder result into the ordinary post-pass record schema."""

    offsets_available = bool(result.metadata.get("source_offsets_available", True))
    record: dict[str, object] = {
        "framing": result.canonical_framing,
        "len": len(result.payload),
        "crc_ok": result.integrity is IntegrityStatus.PASSED,
        "payload_hex": result.payload.hex(),
        "post_pass": True,
        "decoder": decoder,
        "integrity": result.integrity.value,
        "polarity": result.polarity.value,
        "sync_distance": result.sync_distance,
        "corrected_symbols": result.corrected_symbols,
        "metadata": dict(result.metadata),
    }
    if offsets_available:
        start = int(round(result.source_start * sample_rate_hz / symbol_rate_hz))
        end = int(round(result.source_end * sample_rate_hz / symbol_rate_hz))
        record.update(
            {
                "source_start": result.source_start,
                "source_end": result.source_end,
                "source_sample_offset": start,
                "source_sample_end_offset": end,
                "source_offset_kind": "demodulated_symbol_estimate",
            }
        )
        if capture_start_unix_s > 0:
            pass_start = datetime.fromtimestamp(capture_start_unix_s, tz=timezone.utc)
            frame_time = utc_from_sample_offset(pass_start, start, sample_rate_hz)
            record["ts"] = round(frame_time.timestamp(), 3)
            record["timestamp"] = frame_time.isoformat().replace("+00:00", "Z")
    else:
        record["source_offset_kind"] = "unavailable"
    return record


def _record_gnuradio_upstream(frame) -> dict:
    """Convert one gr-satellites component PDU into the post-pass record schema."""

    return {
        "framing": frame.framing or "gr-satellites",
        "len": len(frame.payload),
        # Component deframers emit only protocol-valid frames. They do not expose source offsets.
        "crc_ok": True,
        "payload_hex": frame.payload.hex(),
        "post_pass": True,
        "decoder": "gr-satellites",
        "integrity": "passed",
        "source_offset_kind": "unavailable",
    }


class _ReplayDecoderFanout:
    """Synchronous symbol-to-frame fan-out for faster-than-real-time GNU Radio replay.

    Symbols never wait in a Python queue: the scheduler callback pushes each chunk directly into
    every bounded streaming decoder. Only complete frames cross the control-thread boundary.
    """

    def __init__(self, decoders) -> None:
        self._decoders = tuple(decoders)
        self._q = BoundedQueue[tuple[str, FrameResult]](
            capacity_items=1_024,
            capacity_units=16 * 1024 * 1024,
        )
        self._error: BaseException | None = None

    def push(self, symbols: np.ndarray) -> None:
        if self._error is not None:
            return
        try:
            for label, decoder in self._decoders:
                for result in decoder.push(symbols.copy()):
                    self._q.offer((label, result), units=len(result.payload))
        except BaseException as exc:  # noqa: BLE001 - surface scheduler callback failure
            self._error = exc

    def drain_results(self) -> list[tuple[str, FrameResult]]:
        if self._error is not None:
            raise RuntimeError("native replay decoder failed") from self._error
        stats = self._q.stats()
        require_lossless(stats, label="native replay frame", unit_name="bytes")
        return self._q.drain()

    def flush_results(self) -> list[tuple[str, FrameResult]]:
        if self._error is not None:
            raise RuntimeError("native replay decoder failed") from self._error
        for label, decoder in self._decoders:
            for result in decoder.flush():
                self._q.offer((label, result), units=len(result.payload))
        return self.drain_results()


def decode_capture_gnuradio(
    cf32: str | Path,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    framings_to_try: tuple[str, ...],
    framing_parameters: Mapping[str, object] | None = None,
    modulation: str | None = None,
    capture_start_unix_s: float = 0.0,
    native_evaluation: bool = False,
    use_grsatellites: bool = True,
    replay_speed: float = 8.0,
    append_frames: bool = False,
) -> list[dict]:
    """Replay recorded channel IQ through the exact live GNU Radio modem.

    This is deliberately a second engine, not another numpy approximation. It instantiates the
    same :func:`gnuradio_satellites._build_fallbacks` path used by ``satellite_rx.py``; that path
    uses gr-satellites' own FSK demodulator. Native streaming deframers and, by default,
    gr-satellites component deframers consume the one recovered symbol stream in parallel.

    Station GNU Radio recordings are already downstream of the live LO/Doppler rotator. Therefore
    this engine consumes the stored channel IQ verbatim and must not apply a Doppler track again.
    A throttle bounds scheduler handoff queues while allowing faster-than-real-time replay.
    """

    path = Path(cf32)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sample_rate_hz <= 0 or symbol_rate_hz <= 0:
        raise ValueError("sample and symbol rates must be positive")
    if not 0.1 <= replay_speed <= 32.0:
        raise ValueError("replay_speed must be between 0.1 and 32")
    labels = tuple(label.strip() for label in framings_to_try if label.strip())
    if not labels:
        return []

    meta = path.with_name(path.name + ".json")
    if meta.exists():
        try:
            sample_rate_hz = float(
                json.loads(meta.read_text(encoding="utf-8")).get(
                    "sample_rate_hz", sample_rate_hz
                )
            )
        except (OSError, ValueError, TypeError):
            log.warning("iq_decode: ignoring unreadable sidecar %s", meta.name)

    parameters = dict(framing_parameters or {})
    modulation_value = modulation if modulation is not None else parameters.get(
        "modulation", "gfsk"
    )
    if not isinstance(modulation_value, str) or not modulation_value.strip():
        raise ValueError("modulation must be a non-empty string")

    # Imports stay inside the explicit engine so normal source-tree tests and numpy replay remain
    # safe on machines without GNU Radio, PMT, or gr-satellites.
    from gnuradio import blocks, digital, gr  # noqa: PLC0415
    from gnuradio_satellites import (  # noqa: PLC0415
        _FrameSink,
        make_grsat_deframers,
    )

    tb = gr.top_block("gs_iq_live_chain_replay")
    source = blocks.file_source(gr.sizeof_gr_complex, str(path), False)
    throttle = blocks.throttle(
        gr.sizeof_gr_complex, float(sample_rate_hz) * replay_speed, True
    )
    tb.connect(source, throttle)
    demod_source = throttle
    if parameters.get("invert") is True:
        conjugate = blocks.conjugate_cc()
        tb.connect(demod_source, conjugate)
        demod_source = conjugate

    modulation_spec = modem.modulation_spec(modulation_value)
    if modulation_spec is None or modulation_spec.family != "fsk":
        raise ValueError("gnuradio-live replay currently requires an FSK/GFSK/GMSK modulation")
    _unused_hard, soft = modem.build_demod(
        modulation_value,
        tb,
        demod_source,
        float(sample_rate_hz),
        float(symbol_rate_hz),
        differential=parameters.get("differential"),
        mod_index=parameters.get("mod_index"),
        channel_bw_hz=parameters.get("bandwidth_hz"),
        collect_hard=False,
    )
    if soft is None:
        raise RuntimeError("the exact live FSK demodulator failed to construct")

    # Offline replay must not hand every recovered symbol through the live scheduler queue. At
    # faster-than-real-time rates a Python producer can fill that queue before the control thread
    # gets the GIL to drain it (cmd_148 replay dropped 43,131 symbols at 8x). Decode synchronously
    # in the GNU Radio work callback instead: retained protocol state stays bounded, complete
    # frames cross a separate bounded queue, and the live station queue limits remain unchanged.
    class _NativeReplaySink(gr.sync_block):
        def __init__(self, name: str, input_type, decoders) -> None:
            gr.sync_block.__init__(self, name=name, in_sig=[input_type], out_sig=None)
            self._fanout = _ReplayDecoderFanout(decoders)

        def work(self, input_items, output_items):  # type: ignore[no-untyped-def]
            chunk = np.array(input_items[0], copy=True)
            self._fanout.push(chunk)
            return len(input_items[0])

        def drain_results(self) -> list[tuple[str, FrameResult]]:
            return self._fanout.drain_results()

        def flush_results(self) -> list[tuple[str, FrameResult]]:
            return self._fanout.flush_results()

    native_soft = []
    native_hard = []
    if native_evaluation:
        for label in labels:
            profile = resolve_profile(label)
            if profile is None or not profile.decoder_available:
                continue
            pairing = plan_native_rx_pairing(
                label,
                modulation_spec.kind,
                sample_rate_hz=sample_rate_hz,
                symbol_rate_hz=symbol_rate_hz,
                capture_rate_hz=sample_rate_hz,
                execution=RxExecution.LIVE,
                evaluation=True,
            )
            if not pairing.accepted:
                log.warning(
                    "iq_decode: native live pairing rejected for %s: %s",
                    label,
                    pairing.reason,
                )
                continue
            decoder = build_decoder(label, _decoder_parameters(profile, parameters))
            target = (
                native_soft
                if profile.symbol_input is SymbolInput.SOFT_SYMBOLS
                else native_hard
            )
            target.append((label, decoder))

    native_sinks = []
    if native_soft:
        native_soft_sink = _NativeReplaySink("native_soft_replay", np.float32, native_soft)
        native_sinks.append(("soft", native_soft_sink))
        tb.connect(soft, native_soft_sink)
    if native_hard:
        slicer = digital.binary_slicer_fb()
        native_hard_sink = _NativeReplaySink("native_hard_replay", np.uint8, native_hard)
        native_sinks.append(("hard", native_hard_sink))
        tb.connect(soft, slicer, native_hard_sink)
    upstream_sink = _FrameSink()
    upstream_deframers = make_grsat_deframers(labels) if use_grsatellites and soft else []
    upstream_sinks = []
    for label, decoder in upstream_deframers:
        tagged_sink = _FrameSink(label, upstream_sink._q)
        upstream_sinks.append(tagged_sink)  # retain Python ownership for the graph lifetime
        tb.connect(soft, blocks.copy(gr.sizeof_float), decoder)
        tb.msg_connect(decoder, "out", tagged_sink, "in")
    if not native_sinks and not upstream_deframers:
        raise RuntimeError(
            "the exact live chain built no deframer; enable --native-evaluation and/or "
            "leave --grsatellites enabled for the requested framing"
        )

    records: list[dict] = []

    def drain_once() -> None:
        # Deduplicate only within one drain. Repeated identical beacons in later drains remain
        # distinct, while a native/upstream parity hit from the same burst becomes one record.
        batch: dict[tuple[str, str], dict] = {}
        for symbol_kind, native_sink in native_sinks:
            for label, result in native_sink.drain_results():
                record = _record_gnuradio_native(
                    result,
                    decoder=f"native:{modulation_spec.kind}{int(symbol_rate_hz)}-{symbol_kind}:{label}",
                    sample_rate_hz=sample_rate_hz,
                    symbol_rate_hz=symbol_rate_hz,
                    capture_start_unix_s=capture_start_unix_s,
                )
                key = (str(record["framing"]).lower(), str(record["payload_hex"]))
                batch[key] = record
        for frame in upstream_sink.drain():
            record = _record_gnuradio_upstream(frame)
            key = (str(record["framing"]).lower(), str(record["payload_hex"]))
            prior = batch.get(key)
            if prior is None:
                batch[key] = record
            else:
                prior["decoder"] = f"{prior['decoder']}+gr-satellites"
        records.extend(batch.values())

    completed = threading.Event()
    failure: list[BaseException] = []

    def run_graph() -> None:
        try:
            tb.run()
        except BaseException as exc:  # noqa: BLE001 - propagate scheduler failure to caller
            failure.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(target=run_graph, name="iq-live-chain-replay", daemon=True)
    worker.start()
    try:
        while not completed.wait(_GNU_RADIO_DRAIN_PERIOD_S):
            drain_once()
        worker.join()
        drain_once()
        for symbol_kind, native_sink in native_sinks:
            for label, result in native_sink.flush_results():
                records.append(
                    _record_gnuradio_native(
                        result,
                        decoder=(
                            f"native:{modulation_spec.kind}{int(symbol_rate_hz)}-"
                            f"{symbol_kind}:{label}"
                        ),
                        sample_rate_hz=sample_rate_hz,
                        symbol_rate_hz=symbol_rate_hz,
                        capture_start_unix_s=capture_start_unix_s,
                    )
                )
    except BaseException:
        tb.stop()
        worker.join(timeout=5.0)
        raise
    if failure:
        raise RuntimeError("GNU Radio replay failed") from failure[0]
    if records and append_frames:
        _append_frames(path, records)
    log.info(
        "iq_decode: %d frame(s) through exact live GNU Radio chain from %s "
        "(framings=%s, native=%s, gr-satellites=%s)",
        len(records),
        path.name,
        ",".join(labels),
        native_evaluation,
        bool(upstream_deframers),
    )
    return records


def _append_frames(cf32: Path, records: list[dict]) -> None:
    """Append post-pass frames to the pass's ``frames.jsonl`` (the decoded-frames product), each
    tagged ``post_pass=True``. Runs after the flowgraph exited, so there is no race with the live
    writer.

    CA-FLOW-006: an append OSError (ENOSPC is reachable even as root) used to be
    swallowed with a warning — decoded frames were LOST while main still exited 0
    and the pass looked successfully post-processed. Persistence failure now
    propagates so the run exits nonzero and the loss is on the record."""
    out = cf32.parent / "frames.jsonl"
    try:
        with out.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    except OSError as e:
        raise RuntimeError(
            f"iq_decode: frames.jsonl append failed with {len(records)} decoded "
            f"record(s) unpersisted: {e}"
        ) from e


def _load_track(path_str: str) -> list[tuple[float, float]]:
    """Load a Doppler track JSON (``[[t_s, offset_hz], …]`` from gs-client / gs-orbitd)."""
    if not path_str:
        return []
    try:
        raw = json.loads(Path(path_str).read_text())
        return [(float(t), float(o)) for t, o in raw]
    except (OSError, ValueError, TypeError):
        log.warning("iq_decode: unreadable doppler track %s; ignoring", path_str)
        return []


def _load_framing_parameters(raw_json: str) -> dict[str, object]:
    if not raw_json:
        return {}
    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise ValueError("--framing-parameters-json must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="iq_decode",
        description="Post-pass decode of a recorded .cf32 with the non-live (CRC-gated) framings.",
    )
    p.add_argument("--input", required=True, help="path to the .cf32 capture")
    p.add_argument("--sample-rate", type=float, required=True, help="capture sample rate, Hz")
    p.add_argument(
        "--symbol-rate", type=float, default=_DEFAULT_SYMBOL_RATE_HZ, help="link symbol rate, Hz"
    )
    p.add_argument(
        "--framings",
        default=",".join(framings.POST_PASS_FRAMINGS),
        help="comma list; default = the non-live CRC-gated local framings",
    )
    p.add_argument(
        "--doppler-track",
        default="",
        help="JSON [[t_s, offset_hz], …] Doppler track from gs-orbitd (else per-window CFO)",
    )
    p.add_argument(
        "--window-s", type=float, default=_DEFAULT_WINDOW_S, help="demod window (s); keep short"
    )
    p.add_argument(
        "--modulation",
        default=None,
        help="native post-pass modem; default follows framing parameters, then GFSK",
    )
    p.add_argument(
        "--differential",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override differential BPSK decisions; default follows the modulation label",
    )
    p.add_argument(
        "--manchester",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override Manchester BPSK recovery; default follows the modulation label",
    )
    p.add_argument("--mod-index", type=float, default=None)
    p.add_argument("--bt", type=float, default=None)
    p.add_argument(
        "--framing-parameters-json",
        default="{}",
        help="directive waveform/framing parameters as one JSON object",
    )
    p.add_argument(
        "--capture-start-unix-s",
        type=float,
        default=0.0,
        help="UTC Unix seconds of source sample zero; zero leaves frame UTC unavailable",
    )
    p.add_argument(
        "--native-evaluation",
        action="store_true",
        help=(
            "allow evaluation-only native profiles; default permits only profiles whose "
            "post-pass production gate is open"
        ),
    )
    p.add_argument(
        "--engine",
        choices=("numpy", "gnuradio-live"),
        default="numpy",
        help=(
            "numpy for the bounded portable replay; gnuradio-live for the exact station modem "
            "using gr-satellites' FSK demodulator"
        ),
    )
    p.add_argument(
        "--grsatellites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with gnuradio-live, also run gr-satellites component deframers (default: enabled)",
    )
    p.add_argument(
        "--replay-speed",
        type=float,
        default=8.0,
        help="gnuradio-live throttle multiplier, 0.1..32 (default: 8)",
    )
    p.add_argument(
        "--append-frames",
        action="store_true",
        help=(
            "persist gnuradio-live replay hits to the pass frames.jsonl; default prints them "
            "without modifying pass evidence"
        ),
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    fmts = tuple(f.strip().lower() for f in args.framings.split(",") if f.strip())
    try:
        framing_parameters = _load_framing_parameters(args.framing_parameters_json)
        if args.engine == "gnuradio-live":
            if args.doppler_track:
                raise ValueError(
                    "--doppler-track cannot be combined with --engine gnuradio-live; station "
                    "GNU Radio recordings are already downstream of the live Doppler rotator"
                )
            parameters = dict(framing_parameters)
            if args.differential is not None:
                parameters["differential"] = args.differential
            if args.mod_index is not None:
                parameters["mod_index"] = args.mod_index
            if args.bt is not None:
                parameters["bt"] = args.bt
            records = decode_capture_gnuradio(
                args.input,
                sample_rate_hz=args.sample_rate,
                symbol_rate_hz=args.symbol_rate,
                framings_to_try=fmts,
                framing_parameters=parameters,
                modulation=args.modulation,
                capture_start_unix_s=args.capture_start_unix_s,
                native_evaluation=args.native_evaluation,
                use_grsatellites=args.grsatellites,
                replay_speed=args.replay_speed,
                append_frames=args.append_frames,
            )
            for record in records:
                print(json.dumps(record, separators=(",", ":"), sort_keys=True))
        else:
            decode_capture(
                args.input,
                sample_rate_hz=args.sample_rate,
                symbol_rate_hz=args.symbol_rate,
                framings_to_try=fmts,
                doppler_track=_load_track(args.doppler_track) or None,
                capture_start_unix_s=args.capture_start_unix_s,
                window_s=args.window_s,
                framing_parameters=framing_parameters,
                modulation=args.modulation,
                differential=args.differential,
                manchester=args.manchester,
                mod_index=args.mod_index,
                bt=args.bt,
                native_evaluation=args.native_evaluation,
            )
    except Exception:
        log.exception("iq_decode: failed on %s", args.input)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

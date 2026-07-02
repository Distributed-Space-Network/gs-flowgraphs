"""EnduroSat-class UHF link profile + end-to-end transmit/receive helpers.

Pins the physical/link parameters worked out from the spec sheet (we cannot
query the radio, so these are the documented values + the de-facto 9k6 coding
assumptions, all in one place):

    center frequency      401.5 MHz   (mission-set; passed in at runtime)
    channel symbol rate   12 480 sym/s
    user data rate         9 600 bps  (net, after framing; informational)
    modulation            2-GFSK, h ~= 0.5, BT ~= 0.5
    occupied bandwidth    ~18.7 kHz   (Carson, cross-checks the symbol rate)
    FEC                   none
    bit coding            NRZI + G3RUH scrambler (assumed; toggle in framing)
    framing               AX.25 UI over HDLC, EnduroSat 128-byte packet

License: GPLv3 (see ``../../COPYING``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import framing
from .gfsk import GfskParams, demodulate, modulate

CENTER_FREQUENCY_HZ = 401_500_000
SYMBOL_RATE_HZ = 12_480.0
USER_DATA_RATE_BPS = 9_600
OCCUPIED_BANDWIDTH_HZ = 18_700
MOD_INDEX = 0.5
BT = 0.5

# EnduroSat UHF radio packet payload ceiling, and the AX.25 user-info ceiling
# before bit-stuffing (per the spec sheet). Used to sanity-cap info fields.
RADIO_PACKET_MAX_BYTES = 128
AX25_INFO_MAX_BYTES = 77


@dataclass(frozen=True)
class LinkProfile:
    scramble: bool = True
    nrzi: bool = True
    mod_index: float = MOD_INDEX
    bt: float = BT
    symbol_rate_hz: float = SYMBOL_RATE_HZ


def gfsk_params(sample_rate_hz: float, profile: LinkProfile | None = None) -> GfskParams:
    p = profile or LinkProfile()
    return GfskParams(
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=p.symbol_rate_hz,
        mod_index=p.mod_index,
        bt=p.bt,
    )


def transmit(
    body: bytes,
    sample_rate_hz: float,
    *,
    profile: LinkProfile | None = None,
    preamble_flags: int = 16,
    postamble_flags: int = 2,
) -> np.ndarray:
    """AX.25 frame body -> baseband IQ ready for the SDR sink (uplink)."""
    p = profile or LinkProfile()
    bits = framing.encode(
        body,
        preamble_flags=preamble_flags,
        postamble_flags=postamble_flags,
        scramble=p.scramble,
        nrzi=p.nrzi,
    )
    return modulate(bits, gfsk_params(sample_rate_hz, p))


def receive(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    profile: LinkProfile | None = None,
    recover_timing: bool = True,
) -> list[bytes]:
    """Baseband IQ -> list of valid AX.25 frame bodies (downlink/beacon)."""
    return [
        body
        for body, _ in receive_with_offsets(
            iq, sample_rate_hz, profile=profile, recover_timing=recover_timing
        )
    ]


def receive_with_offsets(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    profile: LinkProfile | None = None,
    recover_timing: bool = True,
) -> list[tuple[bytes, int]]:
    """:func:`receive`, with each frame's opening-flag BIT index in the
    demodulated stream. Bit index * samples-per-symbol approximates the frame's
    sample position in ``iq`` to within a few symbols (symbol-timing recovery
    jitter) — the positional identity :class:`StreamDecoder` dedups on."""
    p = profile or LinkProfile()
    bits = demodulate(iq, gfsk_params(sample_rate_hz, p), recover_timing=recover_timing)
    return framing.decode_with_offsets(bits, scramble=p.scramble, nrzi=p.nrzi)


# Drain-boundary carry, in bits: ~2 max-length AX.25 frames, mirroring the GR
# engine's bit-level tail (cubesat_gfsk_ax25_rx). Converted to IQ samples via the
# profile's samples/symbol, plus demod filter/timing history (the discriminator's
# 64-symbol moving-mean, the matched-filter span, and Gardner lock-in).
_TAIL_BITS = 4096
_TAIL_SETTLE_SYMBOLS = 128
# Positional-dedup tolerance, in symbols: two decodes of the SAME frame from
# different windows agree on its absolute position to within symbol-timing
# jitter (Gardner phase walk over the carried tail — a few symbols). Distinct
# back-to-back identical beacons are >= a whole frame apart (>=160 symbols for
# the smallest AX.25 UI frame), so 32 separates the two cases with wide margin.
_POS_TOL_SYMBOLS = 32


class StreamDecoder:
    """Incremental downlink decoder for the RX app.

    The SDR delivers IQ in chunks across a pass; each :meth:`decode_new` call
    demodulates only the NEW samples plus a bounded carry tail — the same
    drain-boundary tail-carry + positional-dedup pattern the GR engine uses at
    bit level — so per-call cost is O(new samples) and the retained IQ is
    bounded (docs/10 MED-3; the old whole-capture redecode was quadratic in pass
    length and held every chunk of the pass in RAM). It also removes the HIGH-1
    exposure the old count-based dedup had: the frame list handed to the dedup
    is now per-window, so a longer capture can never re-baseline it under the
    emitted count and silently drop a frame.

    Dedup is POSITIONAL (docs/10 section 7): every emitted frame is remembered
    with its ABSOLUTE sample position (opening-flag bit index * sps), and a
    re-decoded frame is dropped only when both its payload matches AND its
    position falls within ``_POS_TOL_SYMBOLS`` of an already-emitted one. A
    payload-set dedup would permanently suppress genuine repeat beacons; the
    earlier tail-SUBTRACT dedup (decode the carried tail alone, subtract its
    frames from the window's) assumed the demod is suffix-local — it is not
    (capture-global RMS normalization + centered moving mean), so tail-alone
    decode could miss a frame the window decode found (subtracting a NEW frame
    instead: loss) or find one the window decode missed (subtracting nothing:
    ~2-3 % duplicates under mixed-amplitude chunking, docs/J MED-1). Position
    records are pruned once they fall out of the carried tail, so memory stays
    bounded. The tail spans ~2 max-length AX.25 frames plus the demod's
    filter/timing history, so a frame straddling a drain boundary is decoded
    whole from the carry on the next call.

    Call ``decode_new`` on a timer (e.g. every few seconds) rather than per
    chunk; the app drives the cadence. ``push`` may run on a different thread
    than ``decode_new`` (the app decodes off its event loop); a lock guards the
    chunk hand-off.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        *,
        profile: LinkProfile | None = None,
        recover_timing: bool = True,
    ) -> None:
        self._sr = sample_rate_hz
        self._profile = profile or LinkProfile()
        self._recover_timing = recover_timing
        self._sps = sample_rate_hz / self._profile.symbol_rate_hz
        self._tail_max = int((_TAIL_BITS + _TAIL_SETTLE_SYMBOLS) * self._sps)
        self._pos_tol = int(_POS_TOL_SYMBOLS * self._sps)
        self._consumed = 0  # absolute sample index just past the previous window
        self._emitted: list[tuple[bytes, int]] = []  # (payload, absolute sample pos)
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._tail = np.empty(0, dtype=np.complex64)

    def push(self, iq_chunk: np.ndarray) -> None:
        chunk = np.asarray(iq_chunk, dtype=np.complex64)
        with self._lock:
            self._chunks.append(chunk)

    def decode_new(self) -> list[bytes]:
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return []  # nothing new — everything already decoded and emitted
        fresh = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        tail = self._tail
        window = np.concatenate([tail, fresh]) if len(tail) else fresh
        window_start = self._consumed - len(tail)  # absolute pos of window[0]
        out: list[bytes] = []
        for body, bit_off in receive_with_offsets(
            window, self._sr, profile=self._profile, recover_timing=self._recover_timing
        ):
            pos = window_start + int(round(bit_off * self._sps))
            if any(
                body == prev_body and abs(pos - prev_pos) <= self._pos_tol
                for prev_body, prev_pos in self._emitted
            ):
                continue  # the same frame, re-decoded out of the carried tail
            self._emitted.append((body, pos))
            out.append(body)
        self._consumed += len(fresh)
        self._tail = window[-self._tail_max :].copy()
        # A frame whose samples have left the carried tail can never re-decode:
        # prune its record (bounded memory across a pass).
        horizon = self._consumed - len(self._tail) - self._pos_tol
        self._emitted = [(b, p) for b, p in self._emitted if p >= horizon]
        return out

    def flush(self) -> list[bytes]:
        return self.decode_new()


__all__ = [
    "AX25_INFO_MAX_BYTES",
    "BT",
    "CENTER_FREQUENCY_HZ",
    "MOD_INDEX",
    "OCCUPIED_BANDWIDTH_HZ",
    "RADIO_PACKET_MAX_BYTES",
    "SYMBOL_RATE_HZ",
    "USER_DATA_RATE_BPS",
    "LinkProfile",
    "StreamDecoder",
    "gfsk_params",
    "receive",
    "receive_with_offsets",
    "transmit",
]

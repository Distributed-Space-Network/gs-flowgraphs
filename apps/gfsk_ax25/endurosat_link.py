"""EnduroSat UHF chip-packet link layer (SX12xx/CC11xx-style) — RX + TX.

This is the framing the EnduroSat UHF Transceiver Gen 2 uses on the wire. It is
PUBLIC: the same framing is published in GPLv3 gr-satellites'
``endurosat_deframer`` (and the open SmallSatGasTeam/GASPACS-Comms-Info repo), so
this module — like gr-satellites/gr-satnogs, which ship deframers for many
proprietary cubesat protocols — keeps the deframer in the GPL flowgraph layer.
The (non-public, encrypted) AirMAC session/transport layer that rides in the
payload lives in the closed orchestrator, not here. Recipe cross-checked against
``endurosat_deframer`` and CRC-proven on 12 lab frames:

    [0xAA x preamble_len][0x7E sync][length(1)][payload(0-128)][CRC-16 (2)]

  * CRC = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, xorout 0), big-endian,
    computed over (length byte + payload).
  * No G3RUH / NRZI / whitening at this layer (unlike the AX.25 path in
    :mod:`framing`, which we keep for other satellites).
  * The payload is the (AES-encrypted) AirMAC frame — opaque here; the closed
    orchestrator parses/decrypts it (non-public protocol, kept off the public
    repo). This module owns RX/TX of the *link*, so the station can receive and
    transmit EnduroSat UHF packets even before the crypto key is available (the
    payload is carried verbatim).

Bit order: bytes are packed MSB-first from the demodulated bitstream, which is
what decodes our captures with :mod:`gfsk`. gr-satellites documents LSB-first;
the two are internally consistent given our demod's bit phase. For TX *interop
with the real module*, confirm the on-air bit order on the bench.

License: GPLv3 (see ../../COPYING).
"""

from __future__ import annotations

import re
import struct
import threading

import numpy as np

from . import crc as _crc
from . import gfsk

PREAMBLE_BYTE = 0xAA
SYNC_BYTE = 0x7E
DEFAULT_PREAMBLE_LEN = 5
MAX_PAYLOAD = 128

DEFAULT_SYMBOL_RATE_HZ = 9600.0
DEFAULT_MOD_INDEX = 0.5
DEFAULT_BT = 0.5

# preamble (0xAA = 10101010, either phase after demod) followed by the 0x7E flag.
_SYNC_RE = re.compile(r"(?:10101010|01010101){1,}01111110")


def crc16(body: bytes) -> int:
    """CRC-16/CCITT-FALSE over the (length byte + payload), as on the wire."""
    return _crc.crc16_ccitt_false(body)


def frame_bytes(payload: bytes, *, preamble_len: int = DEFAULT_PREAMBLE_LEN) -> bytes:
    """Build a full on-wire EnduroSat packet around ``payload``."""
    if not 0 <= len(payload) <= MAX_PAYLOAD:
        msg = f"payload must be 0..{MAX_PAYLOAD} bytes, got {len(payload)}"
        raise ValueError(msg)
    body = bytes([len(payload)]) + payload
    return (
        bytes([PREAMBLE_BYTE]) * preamble_len
        + bytes([SYNC_BYTE])
        + body
        + struct.pack(">H", crc16(body))
    )


def frame_bits(payload: bytes, *, preamble_len: int = DEFAULT_PREAMBLE_LEN) -> np.ndarray:
    """On-wire packet as a uint8 bit array (MSB-first) for the modulator."""
    raw = frame_bytes(payload, preamble_len=preamble_len)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def deframe(bits: np.ndarray) -> list[bytes]:
    """Recover CRC-valid payloads from a demodulated bitstream.

    Scans for the preamble+0x7E sync, reads the length byte, and keeps payloads
    whose CRC-16 verifies. Spurious 0x7E matches inside data are rejected by the
    CRC, so this is safe to run over a whole burst.
    """
    arr = np.asarray(bits, dtype=np.uint8)
    s = "".join(map(str, arr.tolist()))
    out: list[bytes] = []
    for m in _SYNC_RE.finditer(s):
        by = _pack_msb(arr[m.end() :])
        if len(by) < 3:
            continue
        length = by[0]
        if not 1 <= length <= MAX_PAYLOAD or len(by) < 3 + length:
            continue
        body = by[: 1 + length]
        if by[1 + length : 3 + length] == struct.pack(">H", crc16(body)):
            out.append(body[1:])
    return out


def _pack_msb(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n], bitorder="big").tobytes() if n else b""


def transmit(
    payload: bytes,
    sample_rate_hz: float,
    *,
    symbol_rate_hz: float = DEFAULT_SYMBOL_RATE_HZ,
    mod_index: float = DEFAULT_MOD_INDEX,
    bt: float = DEFAULT_BT,
    preamble_len: int = DEFAULT_PREAMBLE_LEN,
) -> np.ndarray:
    """Payload bytes -> baseband 2-GFSK IQ for one EnduroSat uplink packet.

    ``sample_rate_hz`` must be an integer multiple of ``symbol_rate_hz`` (the
    modulator needs integer samples/symbol), e.g. 153600 = 16 sps at 9600.
    """
    params = gfsk.GfskParams(
        sample_rate_hz=sample_rate_hz, symbol_rate_hz=symbol_rate_hz, mod_index=mod_index, bt=bt
    )
    return gfsk.modulate(frame_bits(payload, preamble_len=preamble_len), params)


# Ordered demod ensemble: (correct_cfo, target_sps, recover_timing). Cheapest /
# most-likely first; receive() stops at the first config that yields a CRC-valid
# frame. This recovers 21/21 EnduroSat lab bursts vs 20/21 for the first config
# alone — short packets fail on different settings, so a few tries close the gap.
_RX_ENSEMBLE: tuple[tuple[bool, int, bool], ...] = (
    (True, 16, False),
    (True, 20, False),
    (True, 16, True),
    (True, 12, False),
    (False, 16, False),
    (True, 8, False),
)


def receive(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    symbol_rate_hz: float = DEFAULT_SYMBOL_RATE_HZ,
    mod_index: float = DEFAULT_MOD_INDEX,
    bt: float = DEFAULT_BT,
) -> list[bytes]:
    """Baseband IQ of one burst -> list of CRC-valid EnduroSat payloads.

    Runs the capture-robust demod ensemble (CFO correction + integer-sps
    resample + max-eye/Gardner, both bit polarities) and returns the payloads
    from the first configuration that decodes a valid frame. For a multi-burst
    recording, segment bursts first and call this per burst.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    for correct_cfo, target_sps, recover_timing in _RX_ENSEMBLE:
        bits = gfsk.demodulate_capture(
            iq,
            sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            mod_index=mod_index,
            bt=bt,
            target_sps=target_sps,
            correct_cfo=correct_cfo,
            recover_timing=recover_timing,
        )
        for candidate in (bits, 1 - bits):  # both bit polarities
            frames = deframe(candidate)
            if frames:
                return frames
    return []


def find_bursts(
    iq: np.ndarray, sample_rate_hz: float, *, min_ms: float = 2.0, threshold_mult: float = 4.0
) -> list[tuple[int, int]]:
    """(start, end) sample indices of on-air bursts via a magnitude gate."""
    mag = np.abs(np.asarray(iq))
    if len(mag) == 0:
        return []
    # Noise floor from a low percentile (robust whether bursts are sparse, as in
    # a 10 s capture, or dense); median would overshoot when signal isn't sparse.
    floor = float(np.percentile(mag, 10))
    thr = max(floor * threshold_mult, float(mag.max()) * 0.08)
    on = (mag > thr).astype(np.int8)
    d = np.diff(on, prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    min_samp = sample_rate_hz * min_ms / 1000.0
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False) if (e - s) > min_samp]


# The shortest CRC-valid on-wire frame the deframer can accept: one preamble byte
# (``_SYNC_RE`` needs at least one 0xAA repetition) + sync + length + 1-byte
# payload + CRC-16 = 6 bytes. Used to cap the re-scanned carry below one frame.
_MIN_FRAME_BITS = 6 * 8
# Ceiling on how long an unfinished (still-above-gate) burst may be deferred
# across ``decode_new`` calls. Generous vs. the longest legal packet (137 bytes
# ~= 114 ms at 9k6) so real packet trains are never force-cut, yet it bounds the
# retained IQ if a continuous carrier/interferer holds the gate open.
_MAX_DEFER_S = 5.0


class StreamDecoder:
    """Incremental burst-based EnduroSat RX for the flowgraph app.

    Buffers IQ chunks across a pass; each ``decode_new`` call segments and
    decodes the bursts in the samples accumulated since the previous call and
    returns their payloads. Drive ``decode_new`` on a timer; the app handles the
    cadence. ``push`` may run on a different thread than ``decode_new`` (the app
    decodes off its event loop); a lock guards the chunk hand-off.

    No-loss / no-duplicate argument (docs/10 review, HIGH-1): every sample is
    burst-gated and decoded EXACTLY ONCE — each call decodes a window of new
    samples and then discards it. The gate threshold (``find_bursts``: noise
    floor vs. ``mag.max()*0.08``) therefore adapts only WITHIN one window and can
    never re-baseline an earlier, already-emitted decode. The old whole-capture
    redecode + count-based dedup was NOT prefix-stable: a strong culmination
    burst raised ``mag.max()`` over the whole growing capture, pushed an earlier
    weak (already-emitted) burst below the gate, and the ``frames[emitted:]``
    slice then silently dropped one new frame forever. The only samples seen by
    two windows are the ``_carry`` samples kept for gate/settling context, and
    ``_carry`` is strictly shorter than the shortest CRC-valid frame, so the
    overlap can never re-yield a frame => no double emission. Identical payloads
    in different bursts are distinct frames and each is emitted (positional
    dedup semantics, docs/10 section 7).

    A burst still above the gate at the window edge is deferred — carried whole
    into the next window so it is decoded once, complete — with the deferral
    capped at ``_MAX_DEFER_S`` (a longer ON region is force-decoded so a
    continuous carrier cannot grow the buffer without bound; only a frame
    straddling that pathological forced cut can be missed, and it was never
    decoded before, so nothing once-emitted is ever lost). Per-call cost is
    O(new samples) and retained IQ is bounded (docs/10 MED-3).
    """

    def __init__(
        self,
        sample_rate_hz: float,
        *,
        symbol_rate_hz: float = DEFAULT_SYMBOL_RATE_HZ,
        mod_index: float = DEFAULT_MOD_INDEX,
        bt: float = DEFAULT_BT,
        guard_ms: float = 3.0,
    ) -> None:
        self._sr = sample_rate_hz
        self._symbol_rate_hz = symbol_rate_hz
        self._mod_index = mod_index
        self._bt = bt
        self._guard = int(sample_rate_hz * guard_ms / 1000.0)
        # Re-scanned overlap: capped BELOW the shortest CRC-valid frame so the
        # only twice-seen samples can never re-emit a frame (see class docstring).
        min_frame = int(_MIN_FRAME_BITS * sample_rate_hz / symbol_rate_hz)
        self._carry = min(self._guard, max(min_frame - 1, 0) // 2)
        self._max_defer = int(sample_rate_hz * _MAX_DEFER_S)
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._pending = np.empty(0, dtype=np.complex64)

    def push(self, iq_chunk: np.ndarray) -> None:
        chunk = np.asarray(iq_chunk, dtype=np.complex64)
        with self._lock:
            self._chunks.append(chunk)

    def decode_new(self) -> list[bytes]:
        return self._decode(final=False)

    def flush(self) -> list[bytes]:
        return self._decode(final=True)

    def _decode(self, *, final: bool) -> list[bytes]:
        with self._lock:
            chunks, self._chunks = self._chunks, []
            pending, self._pending = self._pending, np.empty(0, dtype=np.complex64)
        parts = ([pending] if len(pending) else []) + chunks
        if not parts:
            return []
        window = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if not len(window):  # e.g. only empty chunks pushed
            return []
        cut = len(window) if final else self._cut_point(window)
        out: list[bytes] = []
        for s, e in find_bursts(window[:cut], self._sr):
            seg = window[max(0, s - self._guard) : e + self._guard]
            out.extend(
                receive(
                    seg,
                    self._sr,
                    symbol_rate_hz=self._symbol_rate_hz,
                    mod_index=self._mod_index,
                    bt=self._bt,
                )
            )
        # Keep the deferred (un-decoded) region plus a sub-frame carry for gate
        # context; anything pushed while decoding is in ``_chunks`` and follows.
        keep = window[max(0, cut - self._carry) :].copy()
        with self._lock:
            self._pending = keep
        return out

    def _cut_point(self, window: np.ndarray) -> int:
        """End of the fully-decodable prefix: defer an ON region touching the
        window edge (a burst likely still arriving), capped at ``_max_defer``."""
        mag = np.abs(window)
        floor = float(np.percentile(mag, 10))
        thr = max(floor * 4.0, float(mag.max()) * 0.08)  # same gate as find_bursts
        if mag[-1] <= thr:
            return len(window)
        off = np.flatnonzero(mag <= thr)
        start = int(off[-1]) + 1 if len(off) else 0
        if len(window) - start > self._max_defer:
            return len(window)  # continuous carrier: force-decode, bound the buffer
        return max(0, start - self._guard)


__all__ = [
    "DEFAULT_PREAMBLE_LEN",
    "MAX_PAYLOAD",
    "PREAMBLE_BYTE",
    "SYNC_BYTE",
    "StreamDecoder",
    "crc16",
    "deframe",
    "find_bursts",
    "frame_bits",
    "frame_bytes",
    "receive",
    "transmit",
]

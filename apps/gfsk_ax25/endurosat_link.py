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


class StreamDecoder:
    """Incremental burst-based EnduroSat RX for the flowgraph app.

    Buffers IQ chunks across a pass; ``decode_new`` segments bursts and returns
    only payloads not previously returned (frame order is prefix-stable as the
    buffer grows, so slicing past the emitted count never re-emits). Drive
    ``decode_new`` on a timer; the app handles the cadence.
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
        self._chunks: list[np.ndarray] = []
        self._emitted = 0

    def push(self, iq_chunk: np.ndarray) -> None:
        self._chunks.append(np.asarray(iq_chunk, dtype=np.complex64))

    def decode_new(self) -> list[bytes]:
        if not self._chunks:
            return []
        iq = np.concatenate(self._chunks)
        frames: list[bytes] = []
        for s, e in find_bursts(iq, self._sr):
            seg = iq[max(0, s - self._guard) : e + self._guard]
            frames.extend(
                receive(
                    seg,
                    self._sr,
                    symbol_rate_hz=self._symbol_rate_hz,
                    mod_index=self._mod_index,
                    bt=self._bt,
                )
            )
        new = frames[self._emitted :]
        self._emitted = len(frames)
        return new

    def flush(self) -> list[bytes]:
        return self.decode_new()


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

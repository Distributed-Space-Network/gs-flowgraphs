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
    p = profile or LinkProfile()
    bits = demodulate(iq, gfsk_params(sample_rate_hz, p), recover_timing=recover_timing)
    return framing.decode(bits, scramble=p.scramble, nrzi=p.nrzi)


class StreamDecoder:
    """Incremental downlink decoder for the RX app.

    The SDR delivers IQ in chunks across a pass; this buffers them and, when
    :meth:`decode_new` (or :meth:`flush`) is called, demodulates the whole
    capture and returns only frames not previously returned. Frame discovery is
    prefix-stable (a longer capture finds a superset, in order), so slicing past
    the already-emitted count is sufficient and never re-emits a frame.

    Call ``decode_new`` on a timer (e.g. every few seconds) rather than per
    chunk to keep cost bounded; the app drives the cadence.
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
        self._chunks: list[np.ndarray] = []
        self._emitted = 0

    def push(self, iq_chunk: np.ndarray) -> None:
        self._chunks.append(np.asarray(iq_chunk, dtype=np.complex64))

    def decode_new(self) -> list[bytes]:
        if not self._chunks:
            return []
        iq = np.concatenate(self._chunks)
        frames = receive(
            iq, self._sr, profile=self._profile, recover_timing=self._recover_timing
        )
        new = frames[self._emitted :]
        self._emitted = len(frames)
        return new

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
    "transmit",
]

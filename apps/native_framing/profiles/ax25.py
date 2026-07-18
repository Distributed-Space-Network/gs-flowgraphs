"""Bounded hard-bit AX.25 streaming adapters.

The underlying HDLC, FCS, NRZI, and G3RUH algorithms are the established
repository-owned implementations in :mod:`gfsk_ax25`.  This module adds the
engine-independent streaming and metadata contract required by the native
profile registry.

License: GPLv3 (see ``../../../COPYING``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gfsk_ax25 import g3ruh, hdlc
from native_framing.policies import valid_ax25_address
from native_framing.types import FrameResult, IntegrityStatus, Polarity

_TRANSFORM_HISTORY = 17
_DEFAULT_MAX_FRAME_BYTES = 4096
_HDLC_EXPANSION_NUMERATOR = 10


def _flag_starts(bits: np.ndarray) -> list[int]:
    source = bits.tolist()
    flag = list(hdlc.FLAG_BITS)
    starts: list[int] = []
    index = 0
    while index <= len(source) - len(flag):
        if source[index : index + len(flag)] == flag:
            starts.append(index)
            index += len(flag)
        else:
            index += 1
    return starts


@dataclass
class _HdlcWindow:
    max_symbols: int
    bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    source_base: int = 0

    def push(self, decoded: np.ndarray) -> list[tuple[bytes, int, int]]:
        if decoded.size:
            self.bits = np.concatenate((self.bits, decoded))
        starts = _flag_starts(self.bits)
        closing_by_opening = dict(zip(starts, starts[1:], strict=False))
        frames = []
        for payload, opening in hdlc.deframe_with_offsets(self.bits):
            closing = closing_by_opening.get(opening)
            if closing is not None:
                frames.append(
                    (payload, self.source_base + opening, self.source_base + closing + 8)
                )

        # The last flag can be both the previous frame's closing delimiter and
        # the next frame's opening delimiter.  Retaining from there is enough
        # for arbitrary chunk boundaries and avoids any emitted-frame dedup set.
        if starts:
            keep_from = starts[-1]
            if self.bits.size - keep_from <= self.max_symbols:
                self.bits = self.bits[keep_from:].copy()
                self.source_base += keep_from
            else:
                self._retain_partial_flag()
        elif self.bits.size > 7:
            # No opening flag exists, so only a partial flag can matter later.
            self._retain_partial_flag()
        return frames

    def _retain_partial_flag(self) -> None:
        keep = min(7, self.bits.size)
        dropped = self.bits.size - keep
        self.bits = self.bits[-keep:].copy() if keep else np.empty(0, dtype=np.uint8)
        self.source_base += dropped

    def clear(self) -> None:
        self.source_base += self.bits.size
        self.bits = np.empty(0, dtype=np.uint8)


@dataclass
class _Hypothesis:
    scrambled: bool
    polarity: Polarity
    window: _HdlcWindow
    raw_history: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    previous_level: int = 1

    def transform(self, bits: np.ndarray) -> np.ndarray:
        source = 1 - bits if self.polarity is Polarity.INVERTED else bits
        if self.scrambled:
            combined = np.concatenate((self.raw_history, source))
            decoded = g3ruh.descramble(combined)[self.raw_history.size :]
            self.raw_history = combined[-_TRANSFORM_HISTORY:].copy()
        else:
            decoded = source
        nrzi = g3ruh.nrzi_decode(decoded, initial=self.previous_level)
        if decoded.size:
            self.previous_level = int(decoded[-1])
        return nrzi

    @property
    def retained_symbols(self) -> int:
        return int(self.window.bits.size + self.raw_history.size)

    def clear(self) -> None:
        self.window.clear()
        self.raw_history = np.empty(0, dtype=np.uint8)


class Ax25StreamingDecoder:
    """Incremental AX.25 decoder with explicit coding/polarity hypotheses."""

    def __init__(
        self,
        *,
        canonical: str,
        scramble_hypotheses: tuple[bool, ...],
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if max_frame_bytes < 18:
            raise ValueError("max_frame_bytes must be at least 18")
        self._canonical = canonical
        self._max_frame_bytes = int(max_frame_bytes)
        # Worst-case HDLC stuffing plus delimiters and some sync runway.
        window_symbols = max_frame_bytes * _HDLC_EXPANSION_NUMERATOR + 64
        self._hypotheses = tuple(
            _Hypothesis(scrambled, polarity, _HdlcWindow(window_symbols))
            for scrambled in scramble_hypotheses
            for polarity in (Polarity.NORMAL, Polarity.INVERTED)
        )
        self._max_retained = len(self._hypotheses) * (window_symbols + _TRANSFORM_HISTORY)

    @property
    def retained_symbols(self) -> int:
        return sum(hypothesis.retained_symbols for hypothesis in self._hypotheses)

    @property
    def max_retained_symbols(self) -> int:
        return self._max_retained

    def push(self, symbols: np.ndarray | Sequence[float]) -> list[FrameResult]:
        bits = np.asarray(symbols)
        if bits.ndim != 1:
            raise ValueError("hard-bit chunks must be one-dimensional")
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValueError("hard-bit chunks may contain only 0 and 1")
        hard = bits.astype(np.uint8, copy=False)
        candidates: dict[
            tuple[int, int, bytes], list[tuple[Polarity, bool]]
        ] = {}
        for hypothesis in self._hypotheses:
            transformed = hypothesis.transform(hard)
            for payload, source_start, source_end in hypothesis.window.push(transformed):
                # The configured limit includes the two FCS octets stripped by
                # the HDLC layer. Enforce it even when a whole oversized frame
                # arrives in one chunk before retained-state trimming runs.
                if len(payload) + 2 > self._max_frame_bytes:
                    continue
                identity = (source_start, source_end, payload)
                candidates.setdefault(identity, []).append(
                    (hypothesis.polarity, hypothesis.scrambled)
                )
        results: list[FrameResult] = []
        for (source_start, source_end, payload), hypotheses in candidates.items():
            polarities = tuple(dict.fromkeys(polarity for polarity, _ in hypotheses))
            scramblings = tuple(dict.fromkeys(scrambled for _, scrambled in hypotheses))
            # NRZI represents data as transitions, so complementing every line
            # level leaves all bits after the initial state unchanged.  When
            # both hypotheses validate, absolute RF polarity is unknowable.
            polarity = polarities[0] if len(polarities) == 1 else Polarity.AMBIGUOUS
            results.append(
                FrameResult(
                    canonical_framing=self._canonical,
                    payload=payload,
                    integrity=IntegrityStatus.PASSED,
                    source_start=source_start,
                    source_end=source_end,
                    polarity=polarity,
                    sync_distance=0.0,
                    metadata={
                        "address_policy_ok": valid_ax25_address(payload),
                        "g3ruh": scramblings[0] if len(scramblings) == 1 else "ambiguous",
                        "nrzi": True,
                        "polarity_hypotheses": tuple(value.value for value in polarities),
                    },
                )
            )
        results.sort(key=lambda result: (result.source_start, result.source_end))
        if self.retained_symbols > self.max_retained_symbols:
            raise RuntimeError("decoder retained-symbol bound violated")
        return results

    def flush(self) -> list[FrameResult]:
        # A valid HDLC frame already has a closing flag and is emitted by push.
        # Anything still buffered is truncated by definition.
        for hypothesis in self._hypotheses:
            hypothesis.clear()
        return []


def build_ax25(parameters: Mapping[str, object]) -> Ax25StreamingDecoder:
    return Ax25StreamingDecoder(
        canonical="ax25",
        scramble_hypotheses=(False, True),
        max_frame_bytes=int(parameters.get("max_frame_bytes", _DEFAULT_MAX_FRAME_BYTES)),
    )


def build_ax25_g3ruh(parameters: Mapping[str, object]) -> Ax25StreamingDecoder:
    return Ax25StreamingDecoder(
        canonical="ax25_g3ruh",
        scramble_hypotheses=(True,),
        max_frame_bytes=int(parameters.get("max_frame_bytes", _DEFAULT_MAX_FRAME_BYTES)),
    )


__all__ = ["Ax25StreamingDecoder", "build_ax25", "build_ax25_g3ruh"]

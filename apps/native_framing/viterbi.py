"""Engine-independent rate-1/2, K=7 convolutional coding and Viterbi decode.

Signed polynomials follow the convention used by GNU Radio and the pinned
gr-satellites ``ccsds_viterbi`` hierarchy: a negative polynomial complements
that encoder output.  Soft symbols use the repository convention
``positive -> one``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

CONVENTIONS: dict[str, tuple[int, int]] = {
    "CCSDS": (0x4F, -0x6D),
    "NASA-DSN": (-0x6D, 0x4F),
    "CCSDS uninverted": (0x4F, 0x6D),
    "NASA-DSN uninverted": (0x6D, 0x4F),
}

_CONSTRAINT_LENGTH = 7
_STATE_BITS = _CONSTRAINT_LENGTH - 1
_STATE_COUNT = 1 << _STATE_BITS
_STATE_MASK = _STATE_COUNT - 1
_REGISTER_MASK = (1 << _CONSTRAINT_LENGTH) - 1
_NEXT_STATES = np.arange(_STATE_COUNT, dtype=np.uint8)
_NEXT_BITS = _NEXT_STATES & 1
_PREVIOUS_0 = _NEXT_STATES >> 1
_PREVIOUS_1 = _PREVIOUS_0 | (_STATE_COUNT >> 1)


@dataclass(frozen=True)
class ViterbiResult:
    bits: tuple[int, ...]
    metric: float
    convention: str | None
    phase: int
    start_state: int
    final_state: int
    mode: str


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _validate_polynomials(polynomials: Sequence[int]) -> tuple[int, int]:
    if len(polynomials) != 2:
        raise ValueError("rate-1/2 coding requires exactly two polynomials")
    result = tuple(int(poly) for poly in polynomials)
    if any(poly == 0 or abs(poly) > _REGISTER_MASK for poly in result):
        raise ValueError("polynomials must be signed non-zero K=7 masks")
    return result  # type: ignore[return-value]


def _resolve_convention(
    convention: str | Sequence[int],
) -> tuple[str | None, tuple[int, int]]:
    if isinstance(convention, str):
        try:
            return convention, CONVENTIONS[convention]
        except KeyError as exc:
            raise ValueError(f"unknown convolutional convention: {convention}") from exc
    return None, _validate_polynomials(convention)


def _validate_bits(bits: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(bit) for bit in bits)
    if not result:
        raise ValueError("input bits must be non-empty")
    if any(bit not in (0, 1) for bit in result):
        raise ValueError("input bits must contain only zero and one")
    return result


def _validate_state(state: int, name: str) -> int:
    state = int(state)
    if not 0 <= state < _STATE_COUNT:
        raise ValueError(f"{name} must be in range 0..{_STATE_COUNT - 1}")
    return state


class ConvolutionalCode:
    """Rate-1/2, K=7 encoder/maximum-likelihood decoder.

    The finite-block modes are deliberately explicit:

    * ``terminated`` appends/removes six zero tail bits and forces state zero;
    * ``truncated`` starts at the supplied state and chooses the best end state;
    * ``tail_biting`` requires the encoder start and end states to match and
      searches all 64 states when decoding.

    Continuous GNU Radio streaming state is intentionally not emulated by this
    finite-block class; an adapter must define its reset and flush boundaries.
    """

    def __init__(self, convention: str | Sequence[int] = "CCSDS") -> None:
        self.convention, self.polynomials = _resolve_convention(convention)

    def encode(
        self,
        bits: Iterable[int],
        *,
        mode: str = "terminated",
        start_state: int = 0,
    ) -> tuple[int, ...]:
        data = _validate_bits(bits)
        start_state = _validate_state(start_state, "start_state")
        if mode == "terminated":
            if start_state != 0:
                raise ValueError("terminated encoding requires start_state zero")
            encoded_bits = data + (0,) * _STATE_BITS
        elif mode == "truncated":
            encoded_bits = data
        elif mode == "tail_biting":
            expected = self._tail_biting_start(data)
            if start_state not in (0, expected):
                raise ValueError(
                    f"tail_biting start_state must be zero/automatic or {expected}"
                )
            start_state = expected
            encoded_bits = data
        else:
            raise ValueError("mode must be 'terminated', 'truncated', or 'tail_biting'")

        state = start_state
        output: list[int] = []
        for bit in encoded_bits:
            register = ((state << 1) | bit) & _REGISTER_MASK
            for polynomial in self.polynomials:
                output.append(_parity(register & abs(polynomial)) ^ (polynomial < 0))
            state = register & _STATE_MASK
        if mode in ("terminated", "tail_biting") and state != start_state:
            raise AssertionError("convolutional encoder termination invariant failed")
        return tuple(output)

    def decode_hard(
        self,
        symbols: Iterable[int],
        *,
        mode: str = "terminated",
        start_state: int = 0,
    ) -> ViterbiResult:
        hard = np.asarray(tuple(symbols))
        if hard.ndim != 1 or hard.size == 0 or np.any((hard != 0) & (hard != 1)):
            raise ValueError("hard symbols must be a non-empty one-dimensional 0/1 sequence")
        return self.decode_soft(
            hard.astype(np.float64) * 2.0 - 1.0,
            mode=mode,
            start_state=start_state,
        )

    def decode_soft(
        self,
        symbols: Iterable[float],
        *,
        mode: str = "terminated",
        start_state: int = 0,
    ) -> ViterbiResult:
        soft = np.asarray(tuple(symbols), dtype=np.float64)
        if soft.ndim != 1 or soft.size == 0:
            raise ValueError("soft symbols must be a non-empty one-dimensional sequence")
        if soft.size % 2:
            raise ValueError("rate-1/2 soft-symbol input must contain complete pairs")
        if not np.all(np.isfinite(soft)):
            raise ValueError("soft symbols must be finite")
        start_state = _validate_state(start_state, "start_state")

        if mode == "terminated":
            if start_state != 0:
                raise ValueError("terminated decoding requires start_state zero")
            if soft.size // 2 <= _STATE_BITS:
                raise ValueError("terminated input must include payload and six tail pairs")
            result = self._decode_path(soft, start_state=0, final_state=0)
            return self._make_result(result, mode, strip_tail=True, start_state=0)
        if mode == "truncated":
            result = self._decode_path(soft, start_state=start_state, final_state=None)
            return self._make_result(
                result, mode, strip_tail=False, start_state=start_state
            )
        if mode == "tail_biting":
            candidates = (start_state,) if start_state != 0 else range(_STATE_COUNT)
            best: tuple[tuple[int, ...], float, int] | None = None
            best_start = 0
            for candidate in candidates:
                try:
                    decoded = self._decode_path(
                        soft, start_state=candidate, final_state=candidate
                    )
                except ValueError:
                    continue
                if best is None or decoded[1] < best[1]:
                    best = decoded
                    best_start = candidate
            if best is None:
                raise ValueError("no valid tail-biting path")
            return self._make_result(
                best, mode, strip_tail=False, start_state=best_start
            )
        raise ValueError("mode must be 'terminated', 'truncated', or 'tail_biting'")

    def _decode_path(
        self,
        soft: np.ndarray,
        *,
        start_state: int,
        final_state: int | None,
    ) -> tuple[tuple[int, ...], float, int]:
        pairs = soft.reshape((-1, 2))
        metrics = np.full(_STATE_COUNT, np.inf, dtype=np.float64)
        metrics[start_state] = 0.0
        predecessors = np.empty((len(pairs), _STATE_COUNT), dtype=np.uint8)

        expected = np.empty((_STATE_COUNT, 2, 2), dtype=np.float64)
        for previous in range(_STATE_COUNT):
            for bit in (0, 1):
                register = ((previous << 1) | bit) & _REGISTER_MASK
                expected[previous, bit] = [
                    1.0
                    if _parity(register & abs(poly)) ^ (poly < 0)
                    else -1.0
                    for poly in self.polynomials
                ]

        for step, observed in enumerate(pairs):
            branch_0 = np.square(
                observed - expected[_PREVIOUS_0, _NEXT_BITS]
            ).sum(axis=1)
            branch_1 = np.square(
                observed - expected[_PREVIOUS_1, _NEXT_BITS]
            ).sum(axis=1)
            metric_0 = metrics[_PREVIOUS_0] + branch_0
            metric_1 = metrics[_PREVIOUS_1] + branch_1
            select_0 = metric_0 <= metric_1
            next_metrics = np.where(select_0, metric_0, metric_1)
            predecessors[step] = np.where(
                select_0, _PREVIOUS_0, _PREVIOUS_1
            ).astype(np.uint8)
            metrics = next_metrics

        selected_state = int(np.argmin(metrics)) if final_state is None else final_state
        metric = float(metrics[selected_state])
        if not np.isfinite(metric):
            raise ValueError("no valid convolutional path for the requested states")
        state = selected_state
        bits = [0] * len(pairs)
        for step in range(len(pairs) - 1, -1, -1):
            bits[step] = state & 1
            state = int(predecessors[step, state])
        if state != start_state:
            raise AssertionError("Viterbi traceback start-state invariant failed")
        return tuple(bits), metric, selected_state

    def _make_result(
        self,
        decoded: tuple[tuple[int, ...], float, int],
        mode: str,
        *,
        strip_tail: bool,
        start_state: int,
    ) -> ViterbiResult:
        bits, metric, final_state = decoded
        if strip_tail:
            bits = bits[:-_STATE_BITS]
        return ViterbiResult(
            bits=bits,
            metric=metric,
            convention=self.convention,
            phase=0,
            start_state=start_state,
            final_state=final_state,
            mode=mode,
        )

    @staticmethod
    def _tail_biting_start(bits: tuple[int, ...]) -> int:
        for candidate in range(_STATE_COUNT):
            state = candidate
            for bit in bits:
                state = ((state << 1) | bit) & _STATE_MASK
            if state == candidate:
                return candidate
        raise ValueError("bits do not admit a tail-biting state")


class StreamingViterbiDecoder:
    """Bounded continuous Viterbi decoder with explicit traceback latency.

    One decoded bit is emitted per complete input pair once ``traceback_depth``
    decisions have accumulated. ``flush()`` resolves the remaining survivor
    path and then resets the decoder. An odd final symbol is rejected rather
    than silently discarded.
    """

    def __init__(
        self,
        convention: str | Sequence[int] = "CCSDS",
        *,
        traceback_depth: int = 80,
        start_state: int | None = None,
    ) -> None:
        self._codec = ConvolutionalCode(convention)
        if traceback_depth < _STATE_BITS:
            raise ValueError("traceback_depth must be at least six bits")
        self.traceback_depth = int(traceback_depth)
        self.start_state = (
            None if start_state is None else _validate_state(start_state, "start_state")
        )
        self._expected = self._build_expected()
        self.reset()

    @property
    def retained_symbol_count(self) -> int:
        return 2 * len(self._decisions) + len(self._pending)

    def reset(self) -> None:
        self._metrics = np.zeros(_STATE_COUNT, dtype=np.float64)
        if self.start_state is not None:
            self._metrics.fill(np.inf)
            self._metrics[self.start_state] = 0.0
        self._decisions: deque[np.ndarray] = deque()
        self._pending: tuple[float, ...] = ()

    def push(self, symbols: Iterable[float]) -> tuple[int, ...]:
        incoming = tuple(float(symbol) for symbol in symbols)
        if not all(np.isfinite(symbol) for symbol in incoming):
            raise ValueError("soft symbols must be finite")
        combined = self._pending + incoming
        complete = len(combined) - len(combined) % 2
        self._pending = combined[complete:]
        output: list[int] = []
        for offset in range(0, complete, 2):
            self._advance(np.asarray(combined[offset : offset + 2]))
            if len(self._decisions) > self.traceback_depth:
                output.append(self._oldest_survivor_bit())
                self._decisions.popleft()
        return tuple(output)

    def flush(self) -> tuple[int, ...]:
        if self._pending:
            raise ValueError("cannot flush an incomplete rate-1/2 symbol pair")
        if not self._decisions:
            self.reset()
            return ()
        state = int(np.argmin(self._metrics))
        reversed_bits: list[int] = []
        for decisions in reversed(self._decisions):
            reversed_bits.append(state & 1)
            state = int(decisions[state])
        output = tuple(reversed(reversed_bits))
        self.reset()
        return output

    def _advance(self, observed: np.ndarray) -> None:
        branch_0 = np.square(
            observed - self._expected[_PREVIOUS_0, _NEXT_BITS]
        ).sum(axis=1)
        branch_1 = np.square(
            observed - self._expected[_PREVIOUS_1, _NEXT_BITS]
        ).sum(axis=1)
        metric_0 = self._metrics[_PREVIOUS_0] + branch_0
        metric_1 = self._metrics[_PREVIOUS_1] + branch_1
        select_0 = metric_0 <= metric_1
        next_metrics = np.where(select_0, metric_0, metric_1)
        predecessors = np.where(select_0, _PREVIOUS_0, _PREVIOUS_1).astype(np.uint8)
        minimum = float(np.min(next_metrics))
        if not np.isfinite(minimum):
            raise ValueError("no valid streaming convolutional path")
        self._metrics = next_metrics - minimum
        self._decisions.append(predecessors)

    def _oldest_survivor_bit(self) -> int:
        state = int(np.argmin(self._metrics))
        decisions = tuple(self._decisions)
        for row in reversed(decisions[1:]):
            state = int(row[state])
        return state & 1

    def _build_expected(self) -> np.ndarray:
        expected = np.empty((_STATE_COUNT, 2, 2), dtype=np.float64)
        for previous in range(_STATE_COUNT):
            for bit in (0, 1):
                register = ((previous << 1) | bit) & _REGISTER_MASK
                expected[previous, bit] = [
                    1.0
                    if _parity(register & abs(poly)) ^ (poly < 0)
                    else -1.0
                    for poly in self._codec.polynomials
                ]
        return expected


def decode_hypotheses(
    symbols: Iterable[float],
    *,
    conventions: Sequence[str] = ("CCSDS", "NASA-DSN"),
    phases: Sequence[int] = (0, 1),
    mode: str = "truncated",
) -> ViterbiResult:
    """Decode explicit convention/pair-phase hypotheses and return best metric."""

    soft = tuple(float(symbol) for symbol in symbols)
    candidates: list[ViterbiResult] = []
    for phase in phases:
        if phase not in (0, 1):
            raise ValueError("phase must be zero or one soft symbol")
        phased = soft[phase:]
        if not phased or len(phased) % 2:
            continue
        for convention in conventions:
            result = ConvolutionalCode(convention).decode_soft(phased, mode=mode)
            candidates.append(
                ViterbiResult(
                    bits=result.bits,
                    metric=result.metric,
                    convention=result.convention,
                    phase=phase,
                    start_state=result.start_state,
                    final_state=result.final_state,
                    mode=result.mode,
                )
            )
    if not candidates:
        raise ValueError("no convention/phase hypothesis has complete symbol pairs")
    return min(candidates, key=lambda result: (result.metric, result.phase))


__all__ = [
    "CONVENTIONS",
    "ConvolutionalCode",
    "StreamingViterbiDecoder",
    "ViterbiResult",
    "decode_hypotheses",
]

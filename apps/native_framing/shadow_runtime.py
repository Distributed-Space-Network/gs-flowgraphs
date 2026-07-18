"""Bounded, deterministic reconciliation for decoder shadow operation.

The live GNU Radio bridge polls independent decoder outputs. The same radio
frame can therefore arrive from the two engines in adjacent polls. This
module pairs those observations without treating a repeated payload from the
same engine as a duplicate.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class ShadowCapacityError(RuntimeError):
    """Raised rather than silently losing comparison state."""


@dataclass(frozen=True)
class ShadowStats:
    polls: int
    primary_frames: int
    reference_frames: int
    matched_pairs: int
    primary_only: int
    reference_only: int
    emitted_frames: int
    duplicates_suppressed: int
    pending_frames: int
    finalized: bool

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class _Pending(Generic[T]):
    source: str
    key: Hashable
    generation: int
    item: T


class ShadowReconciler(Generic[T]):
    """Pair primary/reference results over a small, explicit poll horizon.

    Items are emitted immediately. A later equal item from the other engine
    is suppressed and counted as the second observation of the same frame.
    Equal items from one engine are always preserved. Pairing is by
    multiplicity and insertion order, which makes replay deterministic.
    """

    def __init__(
        self,
        *,
        key: Callable[[T], Hashable],
        max_lag_polls: int = 1,
        max_pending_items: int = 4096,
    ) -> None:
        if isinstance(max_lag_polls, bool) or max_lag_polls < 0:
            raise ValueError("max_lag_polls must be a non-negative integer")
        if isinstance(max_pending_items, bool) or max_pending_items <= 0:
            raise ValueError("max_pending_items must be a positive integer")
        self._key = key
        self._max_lag = max_lag_polls
        self._max_pending = max_pending_items
        self._generation = 0
        self._pending: list[_Pending[T]] = []
        self._primary_frames = 0
        self._reference_frames = 0
        self._matched_pairs = 0
        self._primary_only = 0
        self._reference_only = 0
        self._emitted_frames = 0
        self._duplicates_suppressed = 0
        self._finalized = False

    def reconcile(self, primary: Iterable[T], reference: Iterable[T]) -> list[T]:
        if self._finalized:
            raise RuntimeError("shadow reconciler is finalized")
        self._generation += 1
        self._expire_before(self._generation - self._max_lag)

        primary_items = list(primary)
        reference_items = list(reference)
        self._primary_frames += len(primary_items)
        self._reference_frames += len(reference_items)

        emitted: list[T] = []
        for source, items in (("primary", primary_items), ("reference", reference_items)):
            opposite = "reference" if source == "primary" else "primary"
            for item in items:
                item_key = self._key(item)
                match_index = next(
                    (
                        index
                        for index, pending in enumerate(self._pending)
                        if pending.source == opposite and pending.key == item_key
                    ),
                    None,
                )
                if match_index is not None:
                    self._pending.pop(match_index)
                    self._matched_pairs += 1
                    self._duplicates_suppressed += 1
                    continue
                if len(self._pending) >= self._max_pending:
                    raise ShadowCapacityError(
                        "native shadow comparison capacity exceeded: "
                        f"{self._max_pending} unmatched frames"
                    )
                self._pending.append(_Pending(source, item_key, self._generation, item))
                emitted.append(item)
                self._emitted_frames += 1
        return emitted

    def finalize(self) -> ShadowStats:
        if not self._finalized:
            self._expire_before(self._generation + self._max_lag + 1)
            self._finalized = True
        return self.stats()

    def stats(self) -> ShadowStats:
        return ShadowStats(
            polls=self._generation,
            primary_frames=self._primary_frames,
            reference_frames=self._reference_frames,
            matched_pairs=self._matched_pairs,
            primary_only=self._primary_only,
            reference_only=self._reference_only,
            emitted_frames=self._emitted_frames,
            duplicates_suppressed=self._duplicates_suppressed,
            pending_frames=len(self._pending),
            finalized=self._finalized,
        )

    def _expire_before(self, minimum_generation: int) -> None:
        retained: list[_Pending[T]] = []
        for pending in self._pending:
            if pending.generation < minimum_generation:
                if pending.source == "primary":
                    self._primary_only += 1
                else:
                    self._reference_only += 1
            else:
                retained.append(pending)
        self._pending = retained

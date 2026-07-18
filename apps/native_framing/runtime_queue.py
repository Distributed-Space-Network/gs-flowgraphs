"""Bounded, non-blocking queues for live native-framing scheduler handoff."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True)
class QueueStats:
    """Immutable cumulative accounting plus the current queue occupancy."""

    capacity_items: int
    capacity_units: int
    queued_items: int
    queued_units: int
    accepted_items: int
    accepted_units: int
    drained_items: int
    drained_units: int
    dropped_items: int
    dropped_units: int


class QueueOverflowError(RuntimeError):
    """A live handoff lost data, so downstream decoding cannot remain valid."""

    def __init__(self, label: str, unit_name: str, stats: QueueStats) -> None:
        self.stats = stats
        super().__init__(
            f"{label} queue overflow: dropped {stats.dropped_units} {unit_name} "
            f"in {stats.dropped_items} items"
        )


def require_lossless(stats: QueueStats, *, label: str, unit_name: str) -> None:
    """Fail closed when a scheduler handoff reports any cumulative loss."""

    if stats.dropped_items:
        raise QueueOverflowError(label, unit_name, stats)


class BoundedQueue(Generic[_T]):
    """A small FIFO that never blocks a GNU Radio scheduler thread.

    ``units`` is chosen by the caller: symbols for demodulator queues and bytes
    for frame queues. An item that would exceed either bound is rejected in its
    entirety and accounted. Keeping accepted chunks intact makes a drop an
    explicit stream discontinuity that the live bridge can fail closed on.
    """

    def __init__(self, *, capacity_items: int, capacity_units: int) -> None:
        if not isinstance(capacity_items, int) or isinstance(capacity_items, bool):
            raise ValueError("capacity_items must be a positive integer")
        if not isinstance(capacity_units, int) or isinstance(capacity_units, bool):
            raise ValueError("capacity_units must be a positive integer")
        if capacity_items <= 0 or capacity_units <= 0:
            raise ValueError("queue capacities must be positive")

        self._capacity_items = int(capacity_items)
        self._capacity_units = int(capacity_units)
        self._items: deque[tuple[_T, int]] = deque()
        self._queued_units = 0
        self._accepted_items = 0
        self._accepted_units = 0
        self._drained_items = 0
        self._drained_units = 0
        self._dropped_items = 0
        self._dropped_units = 0
        self._lock = Lock()

    def offer(self, item: _T, *, units: int) -> bool:
        """Enqueue ``item`` without blocking, or account and reject it."""

        if not isinstance(units, int) or isinstance(units, bool) or units < 0:
            raise ValueError("units must be a non-negative integer")
        item_units = int(units)
        with self._lock:
            if (
                len(self._items) >= self._capacity_items
                or self._queued_units + item_units > self._capacity_units
            ):
                self._dropped_items += 1
                self._dropped_units += item_units
                return False
            self._items.append((item, item_units))
            self._queued_units += item_units
            self._accepted_items += 1
            self._accepted_units += item_units
            return True

    def drain(self) -> list[_T]:
        """Atomically remove and return all queued items in FIFO order."""

        with self._lock:
            if not self._items:
                return []
            entries = list(self._items)
            self._items.clear()
            drained_units = self._queued_units
            self._queued_units = 0
            self._drained_items += len(entries)
            self._drained_units += drained_units
        return [item for item, _units in entries]

    def stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                capacity_items=self._capacity_items,
                capacity_units=self._capacity_units,
                queued_items=len(self._items),
                queued_units=self._queued_units,
                accepted_items=self._accepted_items,
                accepted_units=self._accepted_units,
                drained_items=self._drained_items,
                drained_units=self._drained_units,
                dropped_items=self._dropped_items,
                dropped_units=self._dropped_units,
            )

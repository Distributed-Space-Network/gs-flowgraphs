"""Out-of-bench tests for bounded live scheduler handoff queues."""

from __future__ import annotations

import pytest
from _fallback_select import (
    LIVE_SYMBOL_QUEUE_CAPACITY_ITEMS,
    LIVE_SYMBOL_QUEUE_CAPACITY_SYMBOLS,
)
from native_framing.runtime_queue import (
    BoundedQueue,
    QueueOverflowError,
    require_lossless,
)


def test_live_symbol_queue_tolerates_scheduler_fragmentation_without_hiding_loss() -> None:
    """Regression for cmd_176_176's tiny GNU Radio scheduler work items."""

    queue = BoundedQueue[bytes](
        capacity_items=LIVE_SYMBOL_QUEUE_CAPACITY_ITEMS,
        capacity_units=LIVE_SYMBOL_QUEUE_CAPACITY_SYMBOLS,
    )
    for _ in range(290):
        assert queue.offer(b"0123456789", units=10)

    stats = queue.stats()
    assert stats.queued_items == 290
    assert stats.queued_units == 2900
    assert stats.dropped_items == 0
    require_lossless(stats, label="soft-symbol", unit_name="symbols")


def test_bounded_queue_is_fifo_and_accounts_accepted_and_drained_units() -> None:
    queue = BoundedQueue[str](capacity_items=3, capacity_units=10)

    assert queue.offer("first", units=4)
    assert queue.offer("second", units=6)
    assert queue.drain() == ["first", "second"]

    stats = queue.stats()
    assert stats.queued_items == 0
    assert stats.queued_units == 0
    assert stats.accepted_items == 2
    assert stats.accepted_units == 10
    assert stats.drained_items == 2
    assert stats.drained_units == 10
    assert stats.dropped_items == 0
    assert stats.dropped_units == 0


def test_bounded_queue_rejects_whole_items_and_counts_every_drop() -> None:
    queue = BoundedQueue[str](capacity_items=2, capacity_units=5)

    assert queue.offer("kept", units=4)
    assert not queue.offer("unit-overflow", units=2)
    assert queue.offer("zero-size-marker", units=0)
    assert not queue.offer("item-overflow", units=0)
    assert not queue.offer("oversized", units=99)
    assert queue.drain() == ["kept", "zero-size-marker"]

    stats = queue.stats()
    assert stats.accepted_items == 2
    assert stats.accepted_units == 4
    assert stats.dropped_items == 3
    assert stats.dropped_units == 101

    with pytest.raises(
        QueueOverflowError,
        match=r"native symbols queue overflow: dropped 101 symbols in 3 items",
    ) as caught:
        require_lossless(stats, label="native symbols", unit_name="symbols")
    assert caught.value.stats == stats


@pytest.mark.parametrize(
    ("kwargs", "units"),
    [
        ({"capacity_items": 0, "capacity_units": 1}, None),
        ({"capacity_items": 1, "capacity_units": 0}, None),
        ({"capacity_items": True, "capacity_units": 1}, None),
        ({"capacity_items": 1.0, "capacity_units": 1}, None),
        ({"capacity_items": 1, "capacity_units": 1}, -1),
        ({"capacity_items": 1, "capacity_units": 1}, True),
        ({"capacity_items": 1, "capacity_units": 1}, 1.0),
    ],
)
def test_bounded_queue_validation_is_fail_closed(
    kwargs: dict[str, int | float], units: int | float | None
) -> None:
    if units is None:
        with pytest.raises(ValueError):
            BoundedQueue[str](**kwargs)
        return

    queue = BoundedQueue[str](**kwargs)
    with pytest.raises(ValueError):
        queue.offer("invalid", units=units)

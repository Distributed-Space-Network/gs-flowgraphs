"""Out-of-bench tests for bounded native/reference shadow reconciliation."""

from __future__ import annotations

import pytest
from native_framing.shadow_runtime import ShadowCapacityError, ShadowReconciler


def _reconciler(**kwargs: int) -> ShadowReconciler[bytes]:
    return ShadowReconciler(key=lambda payload: payload, **kwargs)


def test_same_poll_comparison_prefers_primary_and_pairs_by_multiplicity() -> None:
    shadow = _reconciler()

    assert shadow.reconcile([b"same", b"same"], [b"same", b"same"]) == [b"same", b"same"]
    stats = shadow.finalize()

    assert stats.matched_pairs == 2
    assert stats.duplicates_suppressed == 2
    assert stats.primary_only == stats.reference_only == 0


def test_adjacent_poll_duplicate_is_suppressed_but_same_engine_repeat_is_preserved() -> None:
    shadow = _reconciler(max_lag_polls=1)

    assert shadow.reconcile([b"beacon"], []) == [b"beacon"]
    assert shadow.reconcile([b"beacon"], [b"beacon"]) == [b"beacon"]
    stats = shadow.finalize()

    assert stats.primary_frames == 2
    assert stats.reference_frames == 1
    assert stats.matched_pairs == 1
    assert stats.primary_only == 1


def test_results_outside_the_explicit_poll_horizon_are_not_suppressed() -> None:
    shadow = _reconciler(max_lag_polls=1)

    assert shadow.reconcile([b"late"], []) == [b"late"]
    assert shadow.reconcile([], []) == []
    assert shadow.reconcile([], [b"late"]) == [b"late"]
    stats = shadow.finalize()

    assert stats.matched_pairs == 0
    assert stats.primary_only == stats.reference_only == 1


def test_pending_comparison_state_is_bounded_and_overflow_fails_closed() -> None:
    shadow = _reconciler(max_pending_items=2)

    with pytest.raises(ShadowCapacityError, match="capacity exceeded"):
        shadow.reconcile([b"one", b"two", b"three"], [])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_lag_polls": -1}, "max_lag_polls"),
        ({"max_lag_polls": True}, "max_lag_polls"),
        ({"max_pending_items": 0}, "max_pending_items"),
        ({"max_pending_items": True}, "max_pending_items"),
    ],
)
def test_configuration_is_validated(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _reconciler(**kwargs)


def test_finalize_is_idempotent_and_rejects_later_input() -> None:
    shadow = _reconciler()
    shadow.reconcile([], [b"reference-only"])

    first = shadow.finalize()
    assert shadow.finalize() == first
    assert first.reference_only == 1
    assert first.pending_frames == 0
    assert first.finalized is True
    with pytest.raises(RuntimeError, match="finalized"):
        shadow.reconcile([], [])

"""Tests for Phase 5: Conversation Leases and Concurrency Control (Section 5 & 18)."""
import pytest
import time
import asyncio
import threading

from agy_bridge.domain import Lease
from agy_bridge.errors import LeaseConflictError, QueueFullError
from agy_bridge.engine.leases import LeaseCoordinator, BoundedQueueError


def test_acquire_and_release_conversation_lease():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    lease = coordinator.acquire_conversation_lease(conversation_id="conv-1", holder_id="req-1", ttl_seconds=10.0)
    assert lease.resource_id == "conv-1"
    assert lease.holder_id == "req-1"
    assert not lease.is_expired

    # Release
    coordinator.release_conversation_lease(lease)
    assert coordinator.get_active_lease("conv-1") is None


def test_concurrent_conversation_calls_cannot_both_submit():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=0)
    lease1 = coordinator.acquire_conversation_lease(conversation_id="conv-1", holder_id="req-1", ttl_seconds=10.0)
    assert lease1 is not None

    # Immediate second call without queue capacity must raise LeaseConflictError (HTTP 409)
    with pytest.raises(LeaseConflictError):
        coordinator.acquire_conversation_lease(conversation_id="conv-1", holder_id="req-2", ttl_seconds=10.0)


def test_bounded_conversation_queue_capacity():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    lease1 = coordinator.acquire_conversation_lease("conv-1", "req-1", ttl_seconds=10.0)

    # Queue up to 2 items
    waiter1 = coordinator.queue_conversation_turn("conv-1", "req-2")
    waiter2 = coordinator.queue_conversation_turn("conv-1", "req-3")
    assert waiter1 is not None
    assert waiter2 is not None

    # 3rd item exceeds queue capacity -> QueueFullError (HTTP 503)
    with pytest.raises(QueueFullError):
        coordinator.queue_conversation_turn("conv-1", "req-4")


def test_engine_concurrency_limit_enforced():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    engine_lease1 = coordinator.acquire_engine_lease("engine-1", "req-1", ttl_seconds=5.0)
    assert engine_lease1 is not None

    # Exceeding engine concurrency
    with pytest.raises(LeaseConflictError):
        coordinator.acquire_engine_lease("engine-1", "req-2", ttl_seconds=5.0)

    # Releasing allows next
    coordinator.release_engine_lease(engine_lease1)
    engine_lease2 = coordinator.acquire_engine_lease("engine-1", "req-2", ttl_seconds=5.0)
    assert engine_lease2.holder_id == "req-2"


def _wait_for_engine_waiters(coordinator, engine_id, count):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with coordinator._condition:
            if len(coordinator._engine_waiters.get(engine_id, ())) == count:
                return
        time.sleep(0.001)
    pytest.fail(f"expected {count} queued engine waiters")


def test_engine_wait_admits_second_holder_after_release():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    first = coordinator.acquire_engine_lease("engine", "first")
    acquired = threading.Event()
    result = []

    def wait_for_lease():
        result.append(coordinator.acquire_engine_lease_wait(
            "engine", "second", ttl_seconds=10, wait_timeout_s=2
        ))
        acquired.set()

    thread = threading.Thread(target=wait_for_lease)
    thread.start()
    _wait_for_engine_waiters(coordinator, "engine", 1)
    assert not acquired.is_set()
    coordinator.release_engine_lease(first)
    assert acquired.wait(1)
    thread.join()
    assert result[0].holder_id == "second"


def test_engine_wait_preserves_fifo_and_never_overlaps():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    first = coordinator.acquire_engine_lease("engine", "first")
    order = []
    active = 0
    overlap = threading.Event()
    guard = threading.Lock()

    def waiter(holder):
        nonlocal active
        lease = coordinator.acquire_engine_lease_wait("engine", holder, ttl_seconds=10, wait_timeout_s=3)
        with guard:
            active += 1
            if active > 1:
                overlap.set()
            order.append(holder)
        time.sleep(0.01)
        with guard:
            active -= 1
        coordinator.release_engine_lease(lease)

    threads = [threading.Thread(target=waiter, args=(name,)) for name in ("second", "third")]
    threads[0].start()
    _wait_for_engine_waiters(coordinator, "engine", 1)
    threads[1].start()
    _wait_for_engine_waiters(coordinator, "engine", 2)
    coordinator.release_engine_lease(first)
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert order == ["second", "third"]
    assert not overlap.is_set()


def test_engine_wait_timeout_removes_wait_slot():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=1)
    coordinator.acquire_engine_lease("engine", "first")
    with pytest.raises(LeaseConflictError, match="(?i)timed out"):
        coordinator.acquire_engine_lease_wait("engine", "waiter", ttl_seconds=10, wait_timeout_s=0.02)
    with coordinator._condition:
        assert coordinator._engine_waiters.get("engine", []) == []


def test_engine_wait_full_queue_rejects_without_leaking():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=1)
    coordinator.acquire_engine_lease("engine", "first")
    result = []
    # Record the worker exception without sharing pytest assertion context.
    def wait_second():
        try:
            coordinator.acquire_engine_lease_wait("engine", "second", ttl_seconds=10, wait_timeout_s=2)
        except Exception as exc:
            result.append(exc)
    thread = threading.Thread(target=wait_second)
    thread.start()
    _wait_for_engine_waiters(coordinator, "engine", 1)
    with pytest.raises(QueueFullError):
        coordinator.acquire_engine_lease_wait("engine", "third", ttl_seconds=10, wait_timeout_s=1)
    coordinator.shutdown()
    thread.join(1)
    assert len(result) == 1
    assert isinstance(result[0], LeaseConflictError)
    with coordinator._condition:
        assert coordinator._engine_waiters == {}


def test_engine_wait_shutdown_wakes_waiters():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=1)
    coordinator.acquire_engine_lease("engine", "first")
    result = []
    def wait_for_shutdown():
        try:
            coordinator.acquire_engine_lease_wait("engine", "second", ttl_seconds=10, wait_timeout_s=5)
        except Exception as exc:
            result.append(exc)
    thread = threading.Thread(target=wait_for_shutdown)
    thread.start()
    _wait_for_engine_waiters(coordinator, "engine", 1)
    coordinator.shutdown()
    thread.join(1)
    assert not thread.is_alive()
    assert len(result) == 1 and isinstance(result[0], LeaseConflictError)


def test_engine_wait_queue_size_zero_allows_immediate_admission():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=0)
    lease = coordinator.acquire_engine_lease_wait(
        "engine", "immediate", ttl_seconds=10, wait_timeout_s=0
    )
    assert lease.holder_id == "immediate"


def test_expired_lease_is_evicted_on_admission():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=0)
    # Acquired with already expired TTL
    expired_lease = coordinator.acquire_conversation_lease("conv-1", "req-1", ttl_seconds=-1.0)
    assert expired_lease.is_expired

    # New request should succeed by evicting expired lease
    new_lease = coordinator.acquire_conversation_lease("conv-1", "req-2", ttl_seconds=10.0)
    assert new_lease.holder_id == "req-2"


def test_shutdown_rejects_new_admission_and_clears_queue():
    coordinator = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=2)
    coordinator.acquire_conversation_lease("conv-1", "req-1", ttl_seconds=10.0)
    coordinator.queue_conversation_turn("conv-1", "req-2")

    coordinator.shutdown()

    # New acquisition rejected
    with pytest.raises(LeaseConflictError):
        coordinator.acquire_conversation_lease("conv-2", "req-3", ttl_seconds=10.0)

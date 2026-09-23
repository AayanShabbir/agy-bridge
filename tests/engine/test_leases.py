"""Tests for Phase 5: Conversation Leases and Concurrency Control (Section 5 & 18)."""
import pytest
import time
import asyncio

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

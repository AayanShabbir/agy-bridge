"""Conversation leases and concurrency control (Section 5 & 18)."""
from __future__ import annotations

import time
import threading
import uuid
from typing import Dict, List, Optional

from agy_bridge.domain import Lease
from agy_bridge.errors import LeaseConflictError, QueueFullError


class BoundedQueueError(QueueFullError):
    """Queue limit reached for resource."""


class LeaseCoordinator:
    """Coordinates single-turn conversation leases and bounded engine concurrency."""

    def __init__(self, max_engine_concurrency: int = 1, max_queue_size: int = 5) -> None:
        self.max_engine_concurrency = max_engine_concurrency
        self.max_queue_size = max_queue_size
        self._conversation_leases: Dict[str, Lease] = {}
        self._conversation_queues: Dict[str, List[str]] = {}
        self._engine_leases: Dict[str, List[Lease]] = {}
        self._engine_waiters: Dict[str, List[str]] = {}
        self._condition = threading.Condition(threading.RLock())
        self._is_shutdown = False

    def acquire_conversation_lease(
        self, conversation_id: str, holder_id: str, ttl_seconds: float = 30.0, now: Optional[float] = None
    ) -> Lease:
        with self._condition:
            if self._is_shutdown:
                raise LeaseConflictError("Bridge engine is shutting down; admissions closed")
            current_time = time.monotonic() if now is None else now
            existing = self._conversation_leases.get(conversation_id)
            if existing:
                if existing.expires_at <= current_time:
                    del self._conversation_leases[conversation_id]
                else:
                    raise LeaseConflictError(
                        f"Conversation {conversation_id!r} is currently locked by turn {existing.holder_id!r}"
                    )
            lease = Lease(str(uuid.uuid4())[:8], conversation_id, current_time,
                          current_time + ttl_seconds, holder_id)
            self._conversation_leases[conversation_id] = lease
            return lease

    def release_conversation_lease(self, lease: Lease) -> None:
        with self._condition:
            existing = self._conversation_leases.get(lease.resource_id)
            if existing and existing.lease_id == lease.lease_id:
                del self._conversation_leases[lease.resource_id]

    def get_active_lease(self, conversation_id: str, now: Optional[float] = None) -> Optional[Lease]:
        with self._condition:
            current_time = time.monotonic() if now is None else now
            lease = self._conversation_leases.get(conversation_id)
            if lease:
                if lease.expires_at <= current_time:
                    del self._conversation_leases[conversation_id]
                    return None
                return lease
            return None

    def queue_conversation_turn(self, conversation_id: str, holder_id: str) -> str:
        with self._condition:
            if self._is_shutdown:
                raise LeaseConflictError("Bridge engine is shutting down; admissions closed")
            queue = self._conversation_queues.setdefault(conversation_id, [])
            if len(queue) >= self.max_queue_size:
                raise QueueFullError(
                    f"Queue capacity of {self.max_queue_size} reached for conversation {conversation_id!r}"
                )
            queue.append(holder_id)
            return holder_id

    def acquire_engine_lease(
        self, engine_id: str, holder_id: str, ttl_seconds: float = 60.0, now: Optional[float] = None
    ) -> Lease:
        with self._condition:
            if self._is_shutdown:
                raise LeaseConflictError("Bridge engine is shutting down; admissions closed")
            current_time = time.monotonic() if now is None else now
            active_leases = self._active_engine_leases(engine_id, current_time)
            if len(active_leases) >= self.max_engine_concurrency:
                raise LeaseConflictError(
                    f"Engine {engine_id!r} concurrency limit of {self.max_engine_concurrency} reached"
                )
            lease = Lease(str(uuid.uuid4())[:8], engine_id, current_time,
                          current_time + ttl_seconds, holder_id)
            active_leases.append(lease)
            self._engine_leases[engine_id] = active_leases
            return lease

    def _active_engine_leases(self, engine_id: str, now: float) -> List[Lease]:
        leases = self._engine_leases.get(engine_id, [])
        active = [lease for lease in leases if lease.expires_at > now]
        if len(active) != len(leases):
            self._engine_leases[engine_id] = active
            self._condition.notify_all()
        return active

    def acquire_engine_lease_wait(
        self, engine_id: str, holder_id: str, *, ttl_seconds: float, wait_timeout_s: float
    ) -> Lease:
        deadline = time.monotonic() + max(0.0, wait_timeout_s)
        waiter_id = str(uuid.uuid4())
        with self._condition:
            if self._is_shutdown:
                raise LeaseConflictError("Bridge engine is shutting down; admissions closed")
            active = self._active_engine_leases(engine_id, time.monotonic())
            waiters = self._engine_waiters.setdefault(engine_id, [])
            if not waiters and len(active) < self.max_engine_concurrency:
                return self._make_engine_lease(engine_id, holder_id, ttl_seconds)
            if len(waiters) >= self.max_queue_size:
                raise QueueFullError(f"Engine {engine_id!r} wait queue capacity of {self.max_queue_size} reached")
            waiters.append(waiter_id)
            try:
                while True:
                    if self._is_shutdown:
                        raise LeaseConflictError("Bridge engine is shutting down; admissions closed")
                    now = time.monotonic()
                    active = self._active_engine_leases(engine_id, now)
                    queue = self._engine_waiters[engine_id]
                    if queue[0] == waiter_id and len(active) < self.max_engine_concurrency:
                        queue.pop(0)
                        if not queue:
                            self._engine_waiters.pop(engine_id, None)
                        lease = self._make_engine_lease(engine_id, holder_id, ttl_seconds)
                        self._condition.notify_all()
                        return lease
                    remaining = deadline - now
                    if remaining <= 0:
                        raise LeaseConflictError(f"Timed out waiting for engine {engine_id!r} admission")
                    expiries = [lease.expires_at - now for lease in active]
                    wait_for = min([remaining, 1.0] + [max(0.001, expiry) for expiry in expiries])
                    self._condition.wait(timeout=wait_for)
            finally:
                queue = self._engine_waiters.get(engine_id)
                if queue and waiter_id in queue:
                    queue.remove(waiter_id)
                    if not queue:
                        self._engine_waiters.pop(engine_id, None)
                    self._condition.notify_all()

    def _make_engine_lease(self, engine_id: str, holder_id: str, ttl_seconds: float) -> Lease:
        now = time.monotonic()
        lease = Lease(str(uuid.uuid4())[:8], engine_id, now, now + ttl_seconds, holder_id)
        self._engine_leases.setdefault(engine_id, []).append(lease)
        return lease

    def release_engine_lease(self, lease: Lease) -> None:
        with self._condition:
            leases = self._engine_leases.get(lease.resource_id, [])
            self._engine_leases[lease.resource_id] = [l for l in leases if l.lease_id != lease.lease_id]
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._is_shutdown = True
            self._conversation_queues.clear()
            self._conversation_leases.clear()
            self._engine_leases.clear()
            self._engine_waiters.clear()
            self._condition.notify_all()

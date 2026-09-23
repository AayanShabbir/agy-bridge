"""Conversation leases and concurrency control (Section 5 & 18)."""
from __future__ import annotations

import time
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
        self._is_shutdown = False

    def acquire_conversation_lease(
        self, conversation_id: str, holder_id: str, ttl_seconds: float = 30.0, now: Optional[float] = None
    ) -> Lease:
        if self._is_shutdown:
            raise LeaseConflictError("Bridge engine is shutting down; admissions closed")

        current_time = time.monotonic() if now is None else now

        existing = self._conversation_leases.get(conversation_id)
        if existing:
            if existing.expires_at <= current_time:
                # Evict expired lease
                del self._conversation_leases[conversation_id]
            else:
                raise LeaseConflictError(
                    f"Conversation {conversation_id!r} is currently locked by turn {existing.holder_id!r}"
                )

        lease = Lease(
            lease_id=str(uuid.uuid4())[:8],
            resource_id=conversation_id,
            acquired_at=current_time,
            expires_at=current_time + ttl_seconds,
            holder_id=holder_id,
        )
        self._conversation_leases[conversation_id] = lease
        return lease

    def release_conversation_lease(self, lease: Lease) -> None:
        existing = self._conversation_leases.get(lease.resource_id)
        if existing and existing.lease_id == lease.lease_id:
            del self._conversation_leases[lease.resource_id]

    def get_active_lease(self, conversation_id: str, now: Optional[float] = None) -> Optional[Lease]:
        current_time = time.monotonic() if now is None else now
        lease = self._conversation_leases.get(conversation_id)
        if lease:
            if lease.expires_at <= current_time:
                del self._conversation_leases[conversation_id]
                return None
            return lease
        return None

    def queue_conversation_turn(self, conversation_id: str, holder_id: str) -> str:
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
        if self._is_shutdown:
            raise LeaseConflictError("Bridge engine is shutting down; admissions closed")

        current_time = time.monotonic() if now is None else now
        leases = self._engine_leases.setdefault(engine_id, [])

        # Prune expired
        active_leases = [l for l in leases if l.expires_at > current_time]
        self._engine_leases[engine_id] = active_leases

        if len(active_leases) >= self.max_engine_concurrency:
            raise LeaseConflictError(
                f"Engine {engine_id!r} concurrency limit of {self.max_engine_concurrency} reached"
            )

        lease = Lease(
            lease_id=str(uuid.uuid4())[:8],
            resource_id=engine_id,
            acquired_at=current_time,
            expires_at=current_time + ttl_seconds,
            holder_id=holder_id,
        )
        active_leases.append(lease)
        return lease

    def release_engine_lease(self, lease: Lease) -> None:
        leases = self._engine_leases.get(lease.resource_id, [])
        self._engine_leases[lease.resource_id] = [l for l in leases if l.lease_id != lease.lease_id]

    def shutdown(self) -> None:
        self._is_shutdown = True
        self._conversation_queues.clear()
        self._conversation_leases.clear()
        self._engine_leases.clear()

"""Core domain entities, enums, and invariants for agy-bridge."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class TurnState(str, Enum):
    """Lifecycle state machine for a single turn."""
    QUEUED = "QUEUED"
    LEASED = "LEASED"
    SUBSCRIBING = "SUBSCRIBING"
    BASELINED = "BASELINED"
    SENDING = "SENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"
    RECOVERY = "RECOVERY"


class SubmissionCertainty(str, Enum):
    """Tracked submission certainty to prohibit automatic unsafe duplicate execution."""
    NOT_SENT = "NOT_SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED_BEFORE_EXECUTION = "REJECTED_BEFORE_EXECUTION"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class AccountState(str, Enum):
    """Health and admission state of an upstream account identity."""
    READY = "READY"
    BUSY = "BUSY"
    COOLDOWN = "COOLDOWN"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"
    QUARANTINED = "QUARANTINED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class EngineGeneration:
    """Represents a specific process generation of the language server."""
    generation_id: str
    observed_at: float
    endpoint: str
    csrf_token: str = field(repr=False, default="")


@dataclass
class Lease:
    """Exclusive lease acquired for a conversation or engine slot."""
    lease_id: str
    resource_id: str
    acquired_at: float
    expires_at: float
    holder_id: str

    @property
    def is_expired(self) -> bool:
        import time
        return time.monotonic() >= self.expires_at


@dataclass
class TurnObservation:
    """Attributable state accumulator for an in-flight turn."""
    request_id: str
    turn_id: str
    conversation_id: Optional[str]
    generation: str
    state: TurnState = TurnState.QUEUED
    certainty: SubmissionCertainty = SubmissionCertainty.NOT_SENT
    emitted_text_chunks: List[str] = field(default_factory=list)
    emitted_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    baseline_step_count: int = 0
    last_accumulated_text: str = ""
    error: Optional[str] = None
    # Exact-duplicate guard for the reducer: replayed steps (re-attach streams,
    # repeated planner frames) must never re-emit already-attributed text.
    _dedupe_seen: List[str] = field(default_factory=list, repr=False)
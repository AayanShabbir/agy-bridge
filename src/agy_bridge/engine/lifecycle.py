"""Turn lifecycle coordinator and attribution state machine (Section 8)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agy_bridge.domain import (
    SubmissionCertainty,
    TurnObservation,
    TurnState,
)
from agy_bridge.errors import (
    AgyBridgeError,
    AttributionError,
    OutcomeUnknownError,
    UpstreamEmptyResponse,
    UpstreamTimeout,
)


class LifecycleError(AgyBridgeError):
    """Base exception for lifecycle and state machine violations."""


class AttachmentTimeoutError(LifecycleError):
    """Raised when stream subscription failed to attach before submission."""
    http_status = 504
    error_code = "subscription_attachment_timeout"


class DuplicateSubmissionForbiddenError(LifecycleError):
    """Raised when submission is attempted multiple times for the same turn."""
    http_status = 409
    error_code = "duplicate_submission_forbidden"


class TurnLifecycleCoordinator:
    """Coordinates turn transitions, baseline capture, attribution, and completion."""

    def __init__(
        self,
        request_id: str,
        turn_id: str,
        conversation_id: Optional[str],
        generation: str,
    ) -> None:
        self.request_id = request_id
        self.turn_id = turn_id
        self.conversation_id = conversation_id
        self.generation = generation
        self.state = TurnState.LEASED
        self.certainty = SubmissionCertainty.NOT_SENT
        self.observation = TurnObservation(
            request_id=request_id,
            turn_id=turn_id,
            conversation_id=conversation_id,
            generation=generation,
            state=TurnState.LEASED,
            certainty=SubmissionCertainty.NOT_SENT,
            baseline_step_count=0,
        )
        self._has_sent = False
        self._has_attached = False
        self._transitions: List[Dict[str, Any]] = []
        self._record_transition(TurnState.LEASED, "initial_lease")

    def _record_transition(self, new_state: TurnState, reason: str) -> None:
        self.state = new_state
        self.observation.state = new_state
        self._transitions.append({
            "state": new_state.value,
            "reason": reason,
            "timestamp": time.monotonic(),
        })

    def begin_subscription(self) -> None:
        self._record_transition(TurnState.SUBSCRIBING, "begin_subscription")

    def record_attachment(self, initial_step_count: int = 0) -> None:
        self._has_attached = True
        self.observation.baseline_step_count = initial_step_count
        self._record_transition(TurnState.BASELINED, f"attached_baseline_{initial_step_count}")

    def mark_sending(self) -> None:
        if not self._has_attached:
            raise AttachmentTimeoutError("Cannot submit message: subscription not attached")
        if self._has_sent:
            raise DuplicateSubmissionForbiddenError("Message has already been submitted for this turn")
        self._has_sent = True
        self.certainty = SubmissionCertainty.ACKNOWLEDGED
        self.observation.certainty = SubmissionCertainty.ACKNOWLEDGED
        self._record_transition(TurnState.SENDING, "message_submitted")

    def mark_running(self) -> None:
        self._record_transition(TurnState.RUNNING, "agent_running")

    def record_output(self, chunk: str) -> None:
        if chunk:
            self.observation.emitted_text_chunks.append(chunk)
            self.observation.last_accumulated_text += chunk

    def record_step_update(self, step_index: int, text: str) -> None:
        if step_index < self.observation.baseline_step_count:
            raise AttributionError(
                f"Step {step_index} is prior to turn baseline ({self.observation.baseline_step_count}); "
                "cannot attribute stale step to current turn"
            )
        self.record_output(text)

    def mark_completed(self) -> None:
        # Verify attributable output exists
        total_text = "".join(self.observation.emitted_text_chunks).strip()
        has_tools = bool(self.observation.emitted_tool_calls)
        if not total_text and not has_tools:
            self._record_transition(TurnState.FAILED, "empty_response")
            raise UpstreamEmptyResponse("Upstream completed without emitting attributable text or tools")
        self._record_transition(TurnState.COMPLETED, "completed_with_evidence")

    def handle_timeout(self, reason: str = "timeout") -> None:
        if self._has_sent:
            self.certainty = SubmissionCertainty.OUTCOME_UNKNOWN
            self.observation.certainty = SubmissionCertainty.OUTCOME_UNKNOWN
        self.observation.error = reason
        self._record_transition(TurnState.FAILED, f"timeout: {reason}")

    def handle_failure_before_send(self, reason: str) -> None:
        self.certainty = SubmissionCertainty.NOT_SENT
        self.observation.certainty = SubmissionCertainty.NOT_SENT
        self.observation.error = reason
        self._record_transition(TurnState.FAILED, f"pre_send_failure: {reason}")

    def handle_disconnect(self, reason: str) -> None:
        if self._has_sent:
            self.certainty = SubmissionCertainty.OUTCOME_UNKNOWN
            self.observation.certainty = SubmissionCertainty.OUTCOME_UNKNOWN
        self.observation.error = reason
        self._record_transition(TurnState.FAILED, f"disconnect: {reason}")

    def is_retry_allowed(self) -> bool:
        if bool(self.observation.emitted_text_chunks):
            return False
        return self.certainty in (
            SubmissionCertainty.NOT_SENT,
            SubmissionCertainty.REJECTED_BEFORE_EXECUTION,
        )

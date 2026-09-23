"""Tests for Phase 4: Turn Lifecycle & Attribution State Machine (Section 8)."""
import pytest
import time
from typing import List

from agy_bridge.domain import (
    TurnState,
    SubmissionCertainty,
    TurnObservation,
)
from agy_bridge.errors import (
    AttributionError,
    OutcomeUnknownError,
    UpstreamEmptyResponse,
    UpstreamTimeout,
)
from agy_bridge.engine.lifecycle import (
    TurnLifecycleCoordinator,
    AttachmentTimeoutError,
    DuplicateSubmissionForbiddenError,
)


def test_subscribe_before_send_workflow():
    """Turn must transition LEASED -> SUBSCRIBING -> BASELINED -> SENDING -> RUNNING -> COMPLETED."""
    coord = TurnLifecycleCoordinator(
        request_id="req-1",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    assert coord.state == TurnState.LEASED
    assert coord.certainty == SubmissionCertainty.NOT_SENT

    # 1. Begin subscription
    coord.begin_subscription()
    assert coord.state == TurnState.SUBSCRIBING

    # 2. Receive qualifying attachment signal / initial snapshot with baseline steps
    coord.record_attachment(initial_step_count=3)
    assert coord.state == TurnState.BASELINED
    assert coord.observation.baseline_step_count == 3

    # 3. Mark sending
    coord.mark_sending()
    assert coord.state == TurnState.SENDING
    assert coord.certainty == SubmissionCertainty.ACKNOWLEDGED

    # 4. Stream running
    coord.mark_running()
    assert coord.state == TurnState.RUNNING

    # 5. Attributable text received (step 3)
    coord.record_output(chunk="Hello world")
    assert coord.observation.emitted_text_chunks == ["Hello world"]

    # 6. Explicit completion
    coord.mark_completed()
    assert coord.state == TurnState.COMPLETED


def test_send_before_attachment_is_forbidden():
    """Submission cannot occur if subscription attachment timed out or was not established."""
    coord = TurnLifecycleCoordinator(
        request_id="req-2",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()

    # Attempting to send while still SUBSCRIBING (without attachment) must raise
    with pytest.raises(AttachmentTimeoutError):
        coord.mark_sending()


def test_duplicate_submission_forbidden():
    """Cannot call mark_sending() more than once per turn."""
    coord = TurnLifecycleCoordinator(
        request_id="req-3",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()
    coord.record_attachment(initial_step_count=0)
    coord.mark_sending()

    with pytest.raises(DuplicateSubmissionForbiddenError):
        coord.mark_sending()


def test_timeout_after_send_marks_outcome_unknown_forbids_retry():
    """If timeout occurs after bytes may have been sent, outcome is UNKNOWN and auto-retry is forbidden."""
    coord = TurnLifecycleCoordinator(
        request_id="req-4",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()
    coord.record_attachment(initial_step_count=0)
    coord.mark_sending()
    coord.mark_running()

    # Timeout occurs post-send
    coord.handle_timeout(reason="language_server read timeout after 180s")
    assert coord.state == TurnState.FAILED
    assert coord.certainty == SubmissionCertainty.OUTCOME_UNKNOWN
    assert coord.is_retry_allowed() is False


def test_empty_response_on_terminal_completion_raises_upstream_empty():
    """Explicitly completed but unexpectedly empty output produces upstream_empty_response."""
    coord = TurnLifecycleCoordinator(
        request_id="req-5",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()
    coord.record_attachment(initial_step_count=1)
    coord.mark_sending()
    coord.mark_running()

    # Complete without emitting any text or tool calls
    with pytest.raises(UpstreamEmptyResponse):
        coord.mark_completed()

    assert coord.state == TurnState.FAILED


def test_prior_turn_replayed_text_cannot_complete_turn():
    """Steps at or below baseline_step_count cannot satisfy completion or emit text."""
    coord = TurnLifecycleCoordinator(
        request_id="req-6",
        turn_id="turn-2",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()
    # Baseline has 4 steps from turn 1
    coord.record_attachment(initial_step_count=4)
    coord.mark_sending()
    coord.mark_running()

    # Step 3 arrives (stale step)
    with pytest.raises(AttributionError):
        coord.record_step_update(step_index=3, text="Old turn answer")

    assert coord.observation.emitted_text_chunks == []


def test_retry_allowed_only_when_not_sent_or_cleanly_rejected():
    """Retry is allowed if failed before send, but never after send."""
    coord = TurnLifecycleCoordinator(
        request_id="req-7",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord.begin_subscription()
    # Failed before sending (e.g. connection refused to language server)
    coord.handle_failure_before_send("connection refused")
    assert coord.is_retry_allowed() is True

    coord2 = TurnLifecycleCoordinator(
        request_id="req-8",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
    )
    coord2.begin_subscription()
    coord2.record_attachment(0)
    coord2.mark_sending()
    # Partially emitted then disconnected
    coord2.record_output("partial text")
    coord2.handle_disconnect("peer reset")
    assert coord2.is_retry_allowed() is False

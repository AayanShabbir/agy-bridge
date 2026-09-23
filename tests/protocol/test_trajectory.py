"""Tests for Layer 1 Trajectory Reducer and turn attribution."""
import pytest
from agy_bridge.domain import SubmissionCertainty, TurnObservation, TurnState
from agy_bridge.protocol.trajectory import parse_agent_state_update


def test_previous_answer_does_not_complete_new_turn():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    # Turn baseline had 2 steps prior to current turn submission
    obs = TurnObservation(
        request_id="req-1",
        turn_id="turn-2",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        certainty=SubmissionCertainty.ACKNOWLEDGED,
        baseline_step_count=2,
    )

    # An old snapshot from turn 1 arrives (steps 0 and 1)
    old_update = AgentStateUpdate(
        raw={},
        steps=[
            AgentStep(step_index=0, source="CORTEX_TRAJECTORY_SOURCE_CASCADE_CLIENT", step_type="USER_INPUT", content="Old question"),
            AgentStep(step_index=1, source="CORTEX_TRAJECTORY_SOURCE_MODEL", step_type="PLANNER_RESPONSE", planner_response="Old answer from previous turn"),
        ],
        fully_idle=False,
    )

    reduction = reduce_update(obs, old_update)

    assert reduction.new_text_chunks == []
    assert reduction.new_tool_calls == []
    assert reduction.is_completed is False
    assert reduction.is_failed is False
    assert obs.emitted_text_chunks == []


def test_attributable_step_emits_text_chunks():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    obs = TurnObservation(
        request_id="req-2",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        certainty=SubmissionCertainty.ACKNOWLEDGED,
        baseline_step_count=1,
    )

    # Step 1 is after baseline step count (1) -> attributable
    update = AgentStateUpdate(
        raw={},
        steps=[
            AgentStep(step_index=0, step_type="USER_INPUT", content="Hi"),
            AgentStep(step_index=1, step_type="PLANNER_RESPONSE", planner_response="Hello world"),
        ],
        fully_idle=False,
    )

    reduction = reduce_update(obs, update)
    assert reduction.new_text_chunks == ["Hello world"]
    assert obs.emitted_text_chunks == ["Hello world"]


def test_incremental_text_chunks_do_not_duplicate():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    obs = TurnObservation(
        request_id="req-3",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=1,
    )

    # First update: partial text
    up1 = AgentStateUpdate(
        raw={},
        steps=[AgentStep(step_index=1, step_type="PLANNER_RESPONSE", planner_response="Hello")],
        fully_idle=False,
    )
    red1 = reduce_update(obs, up1)
    assert red1.new_text_chunks == ["Hello"]

    # Second update: extended text
    up2 = AgentStateUpdate(
        raw={},
        steps=[AgentStep(step_index=1, step_type="PLANNER_RESPONSE", planner_response="Hello world!")],
        fully_idle=False,
    )
    red2 = reduce_update(obs, up2)
    assert red2.new_text_chunks == [" world!"]
    assert obs.last_accumulated_text == "Hello world!"


def test_duplicate_update_is_noop():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    obs = TurnObservation(
        request_id="req-4",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=0,
    )

    up = AgentStateUpdate(
        raw={},
        steps=[AgentStep(step_index=0, step_type="PLANNER_RESPONSE", planner_response="Constant text")],
        fully_idle=False,
    )
    red1 = reduce_update(obs, up)
    assert red1.new_text_chunks == ["Constant text"]

    # Feed identical update again
    red2 = reduce_update(obs, up)
    assert red2.new_text_chunks == []


def test_completion_with_attributable_text():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    obs = TurnObservation(
        request_id="req-5",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=1,
    )

    up = AgentStateUpdate(
        raw={},
        steps=[AgentStep(step_index=1, step_type="PLANNER_RESPONSE", planner_response="Final answer")],
        fully_idle=True,
    )
    red = reduce_update(obs, up)

    assert red.is_completed is True
    assert red.is_failed is False
    assert obs.state == TurnState.COMPLETED


def test_empty_response_on_fully_idle_fails_with_upstream_empty_response():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, reduce_update

    obs = TurnObservation(
        request_id="req-6",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=1,
    )

    # fullyIdle arrives with NO new steps or empty text
    up = AgentStateUpdate(raw={}, steps=[], fully_idle=True)
    red = reduce_update(obs, up)

    assert red.is_completed is False
    assert red.is_failed is True
    assert red.error == "upstream_empty_response"
    assert obs.state == TurnState.FAILED


def test_error_in_update_marks_turn_failed():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, reduce_update

    obs = TurnObservation(
        request_id="req-7",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=0,
    )

    up = AgentStateUpdate(raw={}, steps=[], error="language_server disconnected")
    red = reduce_update(obs, up)

    assert red.is_failed is True
    assert red.error == "language_server disconnected"
    assert obs.state == TurnState.FAILED


def test_attributable_tool_calls_emitted():
    from agy_bridge.protocol.trajectory import AgentStateUpdate, AgentStep, reduce_update

    obs = TurnObservation(
        request_id="req-8",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-1",
        state=TurnState.RUNNING,
        baseline_step_count=1,
    )

    tc = {"id": "call-1", "name": "read_file", "arguments": {"path": "/tmp/test"}}
    up = AgentStateUpdate(
        raw={},
        steps=[AgentStep(step_index=1, step_type="MODEL", tool_calls=[tc])],
        fully_idle=True,
    )
    red = reduce_update(obs, up)

    assert red.new_tool_calls == [tc]
    assert red.is_completed is True
    assert obs.state == TurnState.COMPLETED


def test_parse_production_stream_frame_shape():
    """The language server's real frame nests steps under update.mainTrajectoryUpdate.stepsUpdate."""
    data = {
        "update": {
            "status": "CASCADE_RUN_STATUS_RUNNING",
            "fullyIdle": False,
            "mainTrajectoryUpdate": {
                "stepsUpdate": {
                    "steps": [
                        {"stepIndex": 9, "plannerResponse": {"modifiedResponse": "M9"}},
                        {"stepIndex": 10, "plannerResponse": {"response": "M10"}},
                    ]
                }
            },
        }
    }
    u = parse_agent_state_update(data)
    assert len(u.steps) == 2
    assert u.steps[0].planner_response == "M9"
    assert u.steps[1].planner_response == "M10"
    assert u.status == "CASCADE_RUN_STATUS_RUNNING"
    assert u.fully_idle is False


def test_parse_idle_from_status_enum():
    """CASCADE_RUN_STATUS_IDLE counts as fully idle even without the flag."""
    u = parse_agent_state_update({"update": {"status": "CASCADE_RUN_STATUS_IDLE"}})
    assert u.fully_idle is True
    assert u.is_terminal is False

"""Layer 1: Pure trajectory reducer and turn attribution engine.

Maintains step identity, enforces baseline step attribution (rejecting replayed
historical steps from previous turns), and detects terminal completion and
upstream empty responses without filesystem or database scraping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agy_bridge.domain import TurnObservation, TurnState


@dataclass
class AgentStep:
    """Normalized step within an agent trajectory update."""

    step_index: int
    source: str = ""
    step_type: str = ""
    status: str = ""
    content: str = ""
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    planner_response: Optional[str] = None
    step_id: Optional[str] = None


@dataclass
class AgentStateUpdate:
    """State update envelope emitted by the language server."""

    raw: Dict[str, Any]
    steps: List[AgentStep] = field(default_factory=list)
    fully_idle: bool = False
    is_terminal: bool = False
    error: Optional[str] = None
    turn_id: Optional[str] = None
    status: str = ""


@dataclass
class Reduction:
    """Deterministic output delta reduced from an AgentStateUpdate."""

    new_text_chunks: List[str] = field(default_factory=list)
    new_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_completed: bool = False
    is_failed: bool = False
    error: Optional[str] = None


def parse_agent_state_update(data: Dict[str, Any]) -> AgentStateUpdate:
    """Parse raw JSON dict from language server into typed AgentStateUpdate.

    Accepts both the production StreamAgentStateUpdates frame shape
    ({"update": {"status", "fullyIdle", "mainTrajectoryUpdate": {"stepsUpdate": {"steps"}}}})
    and the flat shape ({"trajectory": {"steps": [...]}}).
    """
    update = data.get("update") if isinstance(data.get("update"), dict) else data
    traj = update.get("trajectory") if isinstance(update.get("trajectory"), dict) else update
    mtu = traj.get("mainTrajectoryUpdate")
    if isinstance(mtu, dict):
        traj = mtu
    steps_container = traj.get("stepsUpdate") if isinstance(traj.get("stepsUpdate"), dict) else traj
    raw_steps = steps_container.get("steps") if isinstance(steps_container.get("steps"), list) else []

    steps: List[AgentStep] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue

        step_idx = s.get("stepIndex", s.get("step_index", len(steps)))
        source = str(s.get("source", ""))
        step_type = str(s.get("type", s.get("step_type", "")))
        status = str(s.get("status", ""))
        content = str(s.get("content", ""))
        thinking = str(s.get("thinking", ""))

        planner_resp = None
        if isinstance(s.get("plannerResponse"), dict):
            planner_resp = (
                s["plannerResponse"].get("modifiedResponse")
                or s["plannerResponse"].get("response")
            )
        elif "planner_response" in s:
            planner_resp = str(s["planner_response"])

        tool_calls = []
        raw_tc = s.get("toolCalls") or s.get("tool_calls") or []
        if isinstance(raw_tc, list):
            for tc in raw_tc:
                if isinstance(tc, dict):
                    tool_calls.append(tc)

        steps.append(
            AgentStep(
                step_index=step_idx,
                source=source,
                step_type=step_type,
                status=status,
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                planner_response=planner_resp,
                step_id=s.get("stepId") or s.get("step_id"),
            )
        )

    status_raw = str(update.get("status") or data.get("status") or "")
    fully_idle = bool(
        data.get("fullyIdle", data.get("fully_idle", update.get("fullyIdle", False)))
    )
    # The language server also reports idle via the run status enum.
    if status_raw == "CASCADE_RUN_STATUS_IDLE":
        fully_idle = True
    is_terminal = bool(
        data.get("isTerminal", data.get("is_terminal", update.get("isTerminal", False)))
    )
    err = data.get("error") or update.get("error")
    error_str = str(err) if err else None

    return AgentStateUpdate(
        raw=data,
        steps=steps,
        fully_idle=fully_idle,
        is_terminal=is_terminal,
        error=error_str,
        turn_id=data.get("turnId") or data.get("turn_id"),
        status=status_raw,
    )


def reduce_update(
    state: TurnObservation,
    update: AgentStateUpdate,
) -> Reduction:
    """Pure trajectory reducer.

    Attributes updates strictly to the current turn based on state.baseline_step_count.
    Rejects prior-turn outputs, calculates non-duplicative text deltas, and verifies
    terminal completion evidence.
    """
    if update.error:
        state.state = TurnState.FAILED
        state.error = update.error
        return Reduction(is_failed=True, error=update.error)

    new_text_chunks: List[str] = []
    new_tool_calls: List[Dict[str, Any]] = []

    # Filter steps strictly attributable to this turn (step_index >= baseline)
    attributable_steps = [
        s for s in update.steps if s.step_index >= state.baseline_step_count
    ]

    for step in attributable_steps:
        # Determine step text: prefer planner_response, fall back to content for MODEL steps
        candidate_text = step.planner_response
        if candidate_text is None and (
            step.step_type == "PLANNER_RESPONSE" or "MODEL" in step.source
        ):
            candidate_text = step.content

        if candidate_text is not None and candidate_text:
            # Exact replay of an already-attributed step (re-attach streams,
            # repeated planner frames) must never re-emit its text.
            if candidate_text in state._dedupe_seen:
                candidate_text = None
            else:
                state._dedupe_seen.append(candidate_text)

        if candidate_text is not None and candidate_text:
            if candidate_text.startswith(state.last_accumulated_text):
                delta = candidate_text[len(state.last_accumulated_text):]
                if delta:
                    new_text_chunks.append(delta)
                    state.last_accumulated_text = candidate_text
                    state.emitted_text_chunks.append(delta)
            elif candidate_text != state.last_accumulated_text:
                # Text revision or distinct step response
                delta = candidate_text
                new_text_chunks.append(delta)
                state.last_accumulated_text = candidate_text
                state.emitted_text_chunks.append(delta)

        # Attributable tool calls
        for tc in step.tool_calls:
            if tc not in state.emitted_tool_calls:
                new_tool_calls.append(tc)
                state.emitted_tool_calls.append(tc)

    # Terminal completion check
    is_terminal = update.fully_idle or update.is_terminal
    if is_terminal:
        has_content = bool(state.last_accumulated_text or state.emitted_tool_calls)
        if has_content:
            state.state = TurnState.COMPLETED
            return Reduction(
                new_text_chunks=new_text_chunks,
                new_tool_calls=new_tool_calls,
                is_completed=True,
                is_failed=False,
            )
        else:
            # Fully idle with no attributable output -> fail explicitly
            state.state = TurnState.FAILED
            state.error = "upstream_empty_response"
            return Reduction(
                new_text_chunks=new_text_chunks,
                new_tool_calls=new_tool_calls,
                is_completed=False,
                is_failed=True,
                error="upstream_empty_response",
            )

    return Reduction(
        new_text_chunks=new_text_chunks,
        new_tool_calls=new_tool_calls,
        is_completed=False,
        is_failed=False,
    )

"""Hermetic tests for the real AGY completion engine (engine/agent.py).

All turns run against ScriptedTransport: no network, no :8790, no Docker.
Mirrors the production wire ordering (subscribe-first, attach-verified, then
send) and the plan's terminal-evidence rules (silence never proves completion).
"""
from __future__ import annotations

import uuid

import pytest

from agy_bridge.bootstrap import DisabledSupervisor
from agy_bridge.config import BridgeConfig
from agy_bridge.engine.agent import AgyCompletionEngine, render_prompt
from agy_bridge.engine.leases import LeaseCoordinator
from agy_bridge.engine.lifecycle import AttachmentTimeoutError
from agy_bridge.engine.recovery import RecoveryCoordinator
from agy_bridge.engine.registry import RegistryManager
from agy_bridge.engine.scheduler import AccountStatusTracker, AccountScheduler
from agy_bridge.engine.transport import ScriptedTransport
from agy_bridge.errors import (
    RateLimitExceeded,
    UpstreamAuthRequired,
    UpstreamEmptyResponse,
    UpstreamTimeout,
    UpstreamUnavailable,
)
from agy_bridge.domain import SubmissionCertainty


def snap(status: str = "CASCADE_RUN_STATUS_IDLE", steps=None, fully_idle: bool = False) -> dict:
    """Production StreamAgentStateUpdates frame.

    Idle is also inferred from the CASCADE_RUN_STATUS_IDLE enum by the parser,
    so RUNNING frames pass fully_idle=False (default) and IDLE frames leave the
    default status.
    """
    return {
        "update": {
            "status": status,
            "fullyIdle": fully_idle,
            "mainTrajectoryUpdate": {"stepsUpdate": {"steps": steps or []}},
        }
    }


def step(index: int, text: str) -> dict:
    return {"stepIndex": index, "plannerResponse": {"response": text}}


def make_engine(script, tmp_path, *, timeout: float = 240.0, min_deadline_s: float = 10.0) -> tuple:
    reg = tmp_path / "registry.json"
    reg.write_text(
        '{"active": "main", "lanes": {"main": {"http_port": 9, "csrf": "tok", '
        '"model": "gemini-3.8-flash"}}}'
    )
    registry = RegistryManager(registry_path=str(reg))
    leases = LeaseCoordinator(max_engine_concurrency=1, max_queue_size=8)
    tracker = AccountStatusTracker()
    active_ep = registry.get_active_endpoint()
    scheduler = AccountScheduler(
        tracker=tracker, endpoints={active_ep.account_id: active_ep}, preferred_account=active_ep.account_id
    )
    recovery = RecoveryCoordinator(supervisor=DisabledSupervisor(), max_restarts_per_hour=2)
    transport = ScriptedTransport(script=script)
    engine = AgyCompletionEngine(
        registry=registry,
        leases=leases,
        scheduler=scheduler,
        recovery=recovery,
        transport=transport,
        timeout=timeout,
        attach_timeout=2.0,
        delete_after=True,
        min_deadline_s=min_deadline_s,
    )
    return engine, transport, leases, tracker


def request(messages=None, model="gemini-3.8-flash", conversation_id="conv-1"):
    return {
        "model": model,
        "messages": messages or [{"role": "user", "content": "hello"}],
        "conversation_id": conversation_id,
    }


def happy_script() -> dict:
    return {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [
            # pre-run snapshot replays the prior turn (step 7), fullyIdle
            {"frame": snap(steps=[step(7, "OLD ANSWER")])},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(7, "OLD ANSWER")])},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(7, "OLD ANSWER"), step(8, "NEW ANSWER")])},
            {"frame": snap(steps=[step(7, "OLD ANSWER"), step(8, "NEW ANSWER")])},
            {"trailer": {"status": 0}},
        ],
        "DeleteCascadeTrajectory": {},
    }


# ---------------------------------------------------------------------------
# Turn semantics
# ---------------------------------------------------------------------------


def test_full_turn_returns_attributed_content(tmp_path):
    engine, transport, leases, _ = make_engine(happy_script(), tmp_path)
    result = engine.execute_completion(request())
    assert result["content"] == "NEW ANSWER"
    assert result["finish_reason"] == "stop"
    assert result["model"] == "gemini-3.8-flash"
    # deletes the stateless cascade
    assert ("DeleteCascadeTrajectory", {"conversationId": "c-1"}) in transport.calls


def test_upstream_stream_and_cleanup_use_started_cascade_id(tmp_path, monkeypatch):
    import agy_bridge.engine.agent as agent_module

    engine, transport, leases, _ = make_engine(
        {
            "StartCascade": {"cascadeId": "cascade-test-123"},
            "SendUserCascadeMessage": {},
            "StreamAgentStateUpdates": [
                {"frame": snap()},
                {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "ANSWER")])},
                {"frame": snap(steps=[step(0, "ANSWER")])},
                {"trailer": {"status": 0}},
            ],
            "DeleteCascadeTrajectory": {},
        },
        tmp_path,
    )
    original_lifecycle = agent_module.TurnLifecycleCoordinator
    lifecycle_conversation_ids = []
    lease_conversation_ids = []
    original_acquire_lease = leases.acquire_conversation_lease

    def track_conversation_lease(conversation_id, *args, **kwargs):
        lease_conversation_ids.append(conversation_id)
        return original_acquire_lease(conversation_id, *args, **kwargs)

    class TrackingLifecycle(original_lifecycle):
        def __init__(self, *args, **kwargs):
            lifecycle_conversation_ids.append(kwargs["conversation_id"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(agent_module, "TurnLifecycleCoordinator", TrackingLifecycle)
    monkeypatch.setattr(leases, "acquire_conversation_lease", track_conversation_lease)
    engine.execute_completion(request(conversation_id="local-conversation-42"))

    stream_payload = next(payload for method, payload in transport.calls if method == "StreamAgentStateUpdates")
    assert stream_payload["conversationId"] == "cascade-test-123"
    assert ("DeleteCascadeTrajectory", {"conversationId": "cascade-test-123"}) in transport.calls
    assert lifecycle_conversation_ids == ["local-conversation-42"]
    assert leases.get_active_lease("local-conversation-42") is None
    assert leases.get_active_lease("cascade-test-123") is None
    assert lease_conversation_ids == ["local-conversation-42"]


def test_replayed_prior_step_is_not_attributed(tmp_path):
    engine, transport, leases, _ = make_engine(happy_script(), tmp_path)
    result = engine.execute_completion(request())
    assert "OLD ANSWER" not in result["content"]


def test_ordering_subscribe_before_send(tmp_path):
    engine, transport, leases, _ = make_engine(happy_script(), tmp_path)
    engine.execute_completion(request())
    methods = [m for m, _ in transport.calls]
    assert methods.index("StreamAgentStateUpdates") < methods.index("SendUserCascadeMessage")


def test_fresh_cascade_without_replay(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-9"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [
            {"frame": snap()},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING")},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "FRESH")])},
            {"frame": snap(steps=[step(0, "FRESH")])},
            {"trailer": {"status": 0}},
        ],
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    result = engine.execute_completion(request())
    assert result["content"] == "FRESH"


def test_empty_terminal_is_explicit_failure(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [
            {"frame": snap()},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING")},
            {"frame": snap()},
            {"trailer": {"status": 0}},
        ],
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    with pytest.raises(UpstreamEmptyResponse):
        engine.execute_completion(request())


def test_attachment_timeout_blocks_send(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [],  # no frames ever
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    with pytest.raises(AttachmentTimeoutError):
        engine.execute_completion(request())
    methods = [m for m, _ in transport.calls]
    assert "SendUserCascadeMessage" not in methods


def test_send_failure_is_not_sent_and_force_stops(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": RateLimitExceeded("quota out"),
        "StreamAgentStateUpdates": [{"frame": snap()}],
        "ForceStopCascadeTree": {},
        "CancelCascadeInvocation": {},
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    with pytest.raises(RateLimitExceeded):
        engine.execute_completion(request())
    stopped = any(m in ("ForceStopCascadeTree", "CancelCascadeInvocation") for m, _ in transport.calls)
    assert stopped
    # conversation lease released even on failure
    assert leases.get_active_lease("conv-1") is None


def test_start_cascade_failure_propagates_before_subscribe(tmp_path):
    script = {
        "StartCascade": UpstreamUnavailable("brain down"),
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [{"frame": snap()}],
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    with pytest.raises(UpstreamUnavailable):
        engine.execute_completion(request())
    methods = [m for m, _ in transport.calls]
    assert "StreamAgentStateUpdates" not in methods


def test_deadline_timeout_is_outcome_unknown_not_success(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [
            # running with partial output, then stream dies silently
            {"frame": snap()},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "PARTIAL")])},
        ],
    }
    engine, transport, leases, _ = make_engine(script, tmp_path, timeout=1.0, min_deadline_s=0.0)
    with pytest.raises(UpstreamTimeout):
        engine.execute_completion(request())


def test_stream_reattach_when_subscription_ends_early(tmp_path):
    """Stream ending without terminal evidence re-attaches once (production behavior)."""
    streams = [
        [
            {"frame": snap()},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "MORE")])},
        ],
        [
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "MORE"), step(1, "DONE")])},
            {"frame": snap(steps=[step(0, "MORE"), step(1, "DONE")])},
            {"trailer": {"status": 0}},
        ],
    ]

    def stream_handler(payload, on_frame):
        seq = streams.pop(0)
        result_holder = {}

        from agy_bridge.engine.transport import SubscriptionResult

        for item in seq:
            if "trailer" in item:
                result_holder["trailer"] = item["trailer"]
            else:
                on_frame(item["frame"])
        if "trailer" in result_holder:
            from agy_bridge.protocol.trailers import GrpcTrailers

            return SubscriptionResult(
                trailers=GrpcTrailers(status=0, is_success=True),
                outcome=SubscriptionResult.COMPLETED,
            )
        return SubscriptionResult(outcome=SubscriptionResult.TERMINATED, error="ended early")

    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": stream_handler,
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    result = engine.execute_completion(request())
    # the reducer carries prior attributed text forward; the new step appends
    assert result["content"] == "MOREDONE"
    assert len([m for m, _ in transport.calls if m == "StreamAgentStateUpdates"]) == 2


def test_cooldown_blocks_before_any_transport_call(tmp_path):
    engine, transport, leases, tracker = make_engine(happy_script(), tmp_path)
    tracker.record_failure("main", "quota")
    with pytest.raises(RateLimitExceeded):
        engine.execute_completion(request())
    assert transport.calls == []


def test_model_enum_and_thinking_budget_wired(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": {},
        "StreamAgentStateUpdates": [
            {"frame": snap()},
            {"frame": snap("CASCADE_RUN_STATUS_RUNNING", steps=[step(0, "OK")])},
            {"frame": snap(steps=[step(0, "OK")])},
            {"trailer": {"status": 0}},
        ],
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, _ = make_engine(script, tmp_path)
    engine.execute_completion(request(model="claude-opus-4-6"))
    send_payload = dict(transport.calls).get("SendUserCascadeMessage")
    planner = send_payload["cascadeConfig"]["plannerConfig"]
    assert planner["planModel"] == "MODEL_PLACEHOLDER_M26"
    assert planner["thinkingBudget"] == 1024
    assert planner["requestedModel"]["model"] == "MODEL_PLACEHOLDER_M26"


def test_leases_released_on_success(tmp_path):
    engine, transport, leases, _ = make_engine(happy_script(), tmp_path)
    engine.execute_completion(request())
    assert leases.get_active_lease("conv-1") is None
    assert leases._engine_leases.get("language_server") == []


def test_auth_failure_recorded_and_pinned(tmp_path):
    script = {
        "StartCascade": {"cascadeId": "c-1"},
        "SendUserCascadeMessage": UpstreamAuthRequired("not logged in"),
        "StreamAgentStateUpdates": [{"frame": snap()}],
        "ForceStopCascadeTree": {},
        "CancelCascadeInvocation": {},
        "DeleteCascadeTrajectory": {},
    }
    engine, transport, leases, tracker = make_engine(script, tmp_path)
    with pytest.raises(UpstreamAuthRequired):
        engine.execute_completion(request())
    assert tracker.get_state("main") == "AUTH_REQUIRED"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_render_prompt_flattens_roles():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2"},
    ]
    prompt = render_prompt(messages, model="gemini-3.8-flash")
    assert "<system>" in prompt and "be terse" in prompt
    assert "<user>" in prompt and "what is 2+2" in prompt


def test_render_prompt_handles_tool_messages():
    messages = [
        {"role": "user", "content": "run the command"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"ok": true}'},
    ]
    prompt = render_prompt(messages, model="gemini-3.8-flash")
    assert "<assistant_tool_call" in prompt and "run_command" in prompt
    assert "<tool_result" in prompt and '{"ok": true}' in prompt


def test_render_prompt_skips_image_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "describe this"},
            ],
        }
    ]
    prompt = render_prompt(messages, model="gemini-3.8-flash")
    assert "data:image/png" not in prompt
    assert "describe this" in prompt


def test_render_prompt_rejects_empty_messages():
    with pytest.raises(Exception):
        render_prompt([], model="gemini-3.8-flash")

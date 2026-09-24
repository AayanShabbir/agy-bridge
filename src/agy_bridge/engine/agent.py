"""Layer 2: Real AGY completion engine.

Runs one completion turn against the local Antigravity language server over
the EngineTransport, following the wire ordering verified in production
app_lane.py:

  1. StartCascade -> cascadeId
  2. SUBSCRIBE FIRST (the server only pushes updates to attached subscribers)
  3. wait for an accepted/open stream handshake, THEN
  4. SendUserCascadeMessage (bounded, once per turn)
  5. consume state updates; attribute strictly to this turn via the
     trajectory reducer (prior-turn steps below the baseline are rejected);
     terminal evidence = CASCADE_RUN_STATUS_RUNNING observed, then
     fullyIdle / status IDLE with attributable output.
  6. silence never proves completion: a deadline without terminal evidence is
     UpstreamTimeout (outcome unknown, no auto-retry after partial output);
     an explicit idle with no attributable output is UpstreamEmptyResponse.

Failures are lifecycle-recorded (NOT_SENT vs OUTCOME_UNKNOWN) so the
duplicate-execution invariant holds, and every typed error is recorded on
the account tracker for cooldown scheduling. All per-turn state is local to
the call (no shared mutable state): safe under concurrent requests.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agy_bridge.engine.leases import LeaseCoordinator
from agy_bridge.engine.lifecycle import AttachmentTimeoutError, TurnLifecycleCoordinator
from agy_bridge.engine.recovery import RecoveryCoordinator
from agy_bridge.engine.registry import EngineEndpoint, RegistryManager
from agy_bridge.engine.scheduler import AccountScheduler
from agy_bridge.engine.transport import AgyHttpTransport, EngineTransport, error_kind
from agy_bridge.errors import (
    AgyBridgeError,
    UnsupportedInputError,
    UpstreamEmptyResponse,
    UpstreamError,
    UpstreamTimeout,
)
from agy_bridge.protocol.trajectory import parse_agent_state_update, reduce_update

SOURCE_CASCADE_CLIENT = "CORTEX_TRAJECTORY_SOURCE_CASCADE_CLIENT"
VERBOSITY_FULL = "CLIENT_TRAJECTORY_VERBOSITY_FULL"
CASCADE_RUNNING = "CASCADE_RUN_STATUS_RUNNING"
_STREAM_OPENED = object()

DEFAULT_MODEL_ENUM = os.environ.get("AGY_APP_MODEL_ENUM", "MODEL_PLACEHOLDER_M319")

# Verified against the language server's GetAvailableModels (enrolled 2026-09-21).
MODEL_MAP = {
    # GEMINI 3.8 Flash family
    "gemini-3.8-flash-low": "MODEL_PLACEHOLDER_M320",
    "gemini-3.8-flash-medium": "MODEL_PLACEHOLDER_M319",
    "gemini-3.8-flash-high": "MODEL_PLACEHOLDER_M318",
    "gemini-3.8-flash": "MODEL_PLACEHOLDER_M319",
    # GEMINI 3.7 Flash family
    "gemini-3.7-flash-low": "MODEL_PLACEHOLDER_M300",
    "gemini-3.7-flash-medium": "MODEL_PLACEHOLDER_M299",
    "gemini-3.7-flash-high": "MODEL_PLACEHOLDER_M298",
    "gemini-3.7-flash": "MODEL_PLACEHOLDER_M299",
    # GEMINI 3.6 Flash family
    "gemini-3.6-flash-low": "MODEL_PLACEHOLDER_M73",
    "gemini-3.6-flash-medium": "MODEL_PLACEHOLDER_M72",
    "gemini-3.6-flash-high": "MODEL_PLACEHOLDER_M71",
    "gemini-3.6-flash": "MODEL_PLACEHOLDER_M72",
    # GEMINI 3.5 Flash family
    "gemini-3.5-flash-low": "MODEL_PLACEHOLDER_M187",
    "gemini-3.5-flash-medium": "MODEL_PLACEHOLDER_M20",
    "gemini-3.5-flash-high": "MODEL_PLACEHOLDER_M84",
    "gemini-3.5-flash-lite": "MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE",
    "gemini-3.5-flash": "MODEL_PLACEHOLDER_M20",
    # GEMINI 3.1 Pro / Lite
    "gemini-3.1-pro-low": "MODEL_PLACEHOLDER_M36",
    "gemini-3.1-pro-high": "MODEL_PLACEHOLDER_M37",
    "gemini-3.1-pro": "MODEL_PLACEHOLDER_M37",
    "gemini-3.1-flash-lite": "MODEL_PLACEHOLDER_M50",
    # GEMINI 2.5 Pro
    "gemini-2.5-pro": "MODEL_GOOGLE_GEMINI_2_5_PRO",
    # Claude family (Anthropic Vertex)
    "claude-opus-4-6": "MODEL_PLACEHOLDER_M26",
    "claude-opus-4-6-thinking": "MODEL_PLACEHOLDER_M26",
    "claude-sonnet-4-6": "MODEL_PLACEHOLDER_M35",
    "claude-sonnet-4-6-thinking": "MODEL_PLACEHOLDER_M35",
    # Open models
    "gpt-oss-120b-medium": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
}

# Thinking models need a thinkingBudget in cascadeConfig or the lane wedges
# (fleet-verified: M26 needs 1024, M36/M37 need 300).
THINKING_BUDGET = {
    "MODEL_PLACEHOLDER_M26": 1024,  # claude-opus-4-6[-thinking]
    "MODEL_PLACEHOLDER_M16": 300,   # gemini-pro-agent (deprecated alias)
    "MODEL_PLACEHOLDER_M36": 300,   # gemini-3.1-pro-low
    "MODEL_PLACEHOLDER_M37": 300,   # gemini-3.1-pro / -high
}

# Verified production framing for stateless text inference (bridge.py).
TEXT_BACKEND_FRAME = (
    "CRITICAL INSTRUCTION: You are serving as an OpenAI-compatible text inference "
    "backend. You have ZERO local tool execution permissions. Do NOT call run_command, "
    "view_file, write_to_file, or any built-in tools. Output only your direct text response."
)


def resolve_model_enum(model: Optional[str]) -> str:
    if not model:
        return DEFAULT_MODEL_ENUM
    if str(model).startswith("MODEL_"):
        return str(model)
    return MODEL_MAP.get(str(model).lower().strip(), DEFAULT_MODEL_ENUM)


def cascade_config(model_enum: Optional[str]) -> dict:
    """Minimal cascadeConfig the executor needs (planModel + thinking budget)."""
    resolved = resolve_model_enum(model_enum)
    cfg = {
        "plannerConfig": {
            "planModel": resolved,
            "requestedModel": {"model": resolved},
        }
    }
    budget = THINKING_BUDGET.get(resolved)
    if budget:
        cfg["plannerConfig"]["thinkingBudget"] = budget
        cfg["plannerConfig"]["requestedModel"]["thinkingBudget"] = budget
    return cfg


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("text"))
    return str(content)


def render_prompt(messages: list, model: str = "") -> str:
    """Flatten OpenAI message history into the app-lane prompt.

    Mirrors production bridge.py flattening: role tags, tool results,
    assistant tool calls. Image parts are dropped (text inference backend).
    """
    if not messages:
        raise UnsupportedInputError("messages must be a non-empty list")
    parts = [TEXT_BACKEND_FRAME]
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            name = m.get("name") or ""
            tool_id = m.get("tool_call_id", "")
            id_attr = f' id="{tool_id}"' if tool_id else ""
            name_attr = f' name="{name}"' if name else ""
            parts.append(f"<tool_result{id_attr}{name_attr}>\n{_text_of(content)}\n</tool_result>")
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = tc.get("name") or fn.get("name") or "?"
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args, separators=(",", ":"))
                    except Exception:
                        args = "{}"
                parts.append(f'<assistant_tool_call name="{fn_name}" arguments={args}>')
            if content:
                parts.append(f"<assistant>\n{_text_of(content)}\n</assistant>")
            continue
        parts.append(f"<{role}>\n{_text_of(content)}\n</{role}>")
    return "\n\n".join(parts)


class AgyCompletionEngine:
    """Real completion engine: one bounded turn over the AGY transport."""

    def __init__(
        self,
        registry: RegistryManager,
        leases: LeaseCoordinator,
        scheduler: AccountScheduler,
        recovery: RecoveryCoordinator,
        transport: Optional[EngineTransport] = None,
        *,
        timeout: float = 240.0,
        attach_timeout: float = 8.0,
        delete_after: bool = True,
        min_deadline_s: float = 10.0,
    ) -> None:
        self.registry = registry
        self.leases = leases
        self.scheduler = scheduler
        self.recovery = recovery
        self.transport = transport or AgyHttpTransport()
        self.timeout = timeout
        self.attach_timeout = attach_timeout
        self.delete_after = delete_after
        self.min_deadline_s = min_deadline_s

    # -- request entry -------------------------------------------------------

    def execute_completion(self, request_data: dict) -> dict:
        model = str(request_data.get("model") or "gemini-3.8-flash")
        messages = request_data.get("messages") or []
        prompt = render_prompt(messages, model=model)
        conversation_id = (
            request_data.get("conversation_id")
            or request_data.get("conversationId")
            or f"conv-{uuid.uuid4().hex[:12]}"
        )
        request_id = str(request_data.get("request_id") or uuid.uuid4().hex)
        turn_id = uuid.uuid4().hex[:12]
        subscriber_id = f"{request_id}-{turn_id}"

        endpoint = self.scheduler.select_endpoint()  # cooldown-aware admission
        conv_lease = self.leases.acquire_conversation_lease(conversation_id, turn_id)
        engine_lease = self.leases.acquire_engine_lease(endpoint.engine_id, turn_id)
        lifecycle = TurnLifecycleCoordinator(
            request_id=request_id,
            turn_id=turn_id,
            conversation_id=conversation_id,
            generation=endpoint.generation,
        )
        frame_q: "queue.Queue[Any]" = queue.Queue()
        stream_timeout = max(10.0, min(self.timeout, 200.0))
        cascade_id: Optional[str] = None
        upstream_conversation_id = conversation_id
        stream_thread: Optional[threading.Thread] = None

        def _stream_worker() -> None:
            """Subscribe and forward frames into the queue (ended marker = None)."""
            try:
                self.transport.subscribe(
                    endpoint,
                    "StreamAgentStateUpdates",
                    {
                        "conversationId": upstream_conversation_id,
                        "subscriberId": subscriber_id,
                        "initialStepsPageBounds": {"startIndex": -50},
                        "trajectoryVerbosity": VERBOSITY_FULL,
                    },
                    on_frame=lambda d: (frame_q.put(d), False)[1],
                    timeout=stream_timeout,
                    on_open=lambda: frame_q.put(_STREAM_OPENED),
                )
            except Exception as ex:  # typed transport errors
                frame_q.put(ex)
            finally:
                frame_q.put(None)

        def _relaunch_stream() -> None:
            nonlocal stream_thread
            stream_thread = threading.Thread(target=_stream_worker, daemon=True)
            stream_thread.start()

        try:
            # 1. Start the cascade
            start_resp = self.transport.unary(
                endpoint,
                "StartCascade",
                {"source": SOURCE_CASCADE_CLIENT, "prompt": prompt},
                timeout=40,
            )
            if not isinstance(start_resp, dict) or not start_resp.get("cascadeId"):
                self._pre_send_fail(
                    lifecycle, endpoint, UpstreamError("StartCascade returned no cascadeId")
                )
            cascade_id = start_resp["cascadeId"]
            assert cascade_id is not None  # _pre_send_fail raised otherwise
            upstream_conversation_id = cascade_id

            # 2. Subscribe FIRST, then wait for HTTP/gRPC acceptance.
            #    StartCascade made this a fresh upstream cascade, so baseline
            #    zero is safe until pre-run replay frames can refine it.
            lifecycle.begin_subscription()
            _relaunch_stream()
            self._wait_attachment(frame_q, lifecycle)
            baseline = 0
            lifecycle.record_attachment(baseline)

            # 3. Send the turn message exactly once
            try:
                self.transport.unary(
                    endpoint,
                    "SendUserCascadeMessage",
                    {
                        "cascadeId": cascade_id,
                        "items": [{"text": prompt}],
                        "cascadeConfig": cascade_config(model),
                    },
                    timeout=40,
                )
            except Exception as ex:
                self._stop_invocation(endpoint, cascade_id)
                self._pre_send_fail(lifecycle, endpoint, ex)
            lifecycle.mark_sending()

            # 4. Consume until terminal evidence or deadline
            self._consume_turn(lifecycle, endpoint, frame_q, _relaunch_stream)

            content = "".join(lifecycle.observation.emitted_text_chunks).strip()
            self.scheduler.tracker.record_success(endpoint.account_id)
            return {"content": content, "finish_reason": "stop", "model": model}
        except AgyBridgeError as ex:
            self.scheduler.tracker.record_failure(endpoint.account_id, error_kind(ex))
            raise
        finally:
            if stream_thread is not None:
                stream_thread.join(timeout=1.0)
            if self.delete_after and cascade_id is not None:
                try:
                    self.transport.unary(
                        endpoint,
                        "DeleteCascadeTrajectory",
                        {"conversationId": upstream_conversation_id},
                        timeout=10,
                    )
                except Exception:
                    pass  # best-effort stateless cleanup
            self.leases.release_engine_lease(engine_lease)
            self.leases.release_conversation_lease(conv_lease)

    # -- turn mechanics ------------------------------------------------------

    def _wait_attachment(self, frame_q: "queue.Queue[Any]",
                         lifecycle: TurnLifecycleCoordinator) -> None:
        attach_deadline = time.monotonic() + self.attach_timeout
        while time.monotonic() < attach_deadline:
            try:
                data = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if data is None:
                break  # stream ended before the HTTP/gRPC handshake was accepted
            if isinstance(data, Exception):
                lifecycle.handle_disconnect(str(data))
                raise data
            if data is _STREAM_OPENED:
                return
            # A data frame is not proof that the HTTP/gRPC handshake succeeded.
        lifecycle.handle_timeout(f"stream was not accepted within {self.attach_timeout}s")
        raise AttachmentTimeoutError(
            "Cannot submit message: stream handshake was not accepted within "
            f"{self.attach_timeout}s"
        )

    @staticmethod
    def _baseline_step_count(data: dict) -> int:
        """First attributable step index: max replayed step index + 1."""
        update = data.get("update") if isinstance(data.get("update"), dict) else data
        traj = update.get("trajectory") if isinstance(update.get("trajectory"), dict) else update
        mtu = traj.get("mainTrajectoryUpdate")
        if isinstance(mtu, dict):
            traj = mtu
        container = traj.get("stepsUpdate") if isinstance(traj.get("stepsUpdate"), dict) else traj
        steps = container.get("steps") if isinstance(container.get("steps"), list) else []
        indexes = [
            s.get("stepIndex", s.get("idx", 0))
            for s in steps
            if isinstance(s, dict) and isinstance(s.get("stepIndex", s.get("idx", 0)), int)
        ]
        return (max(indexes) + 1) if indexes else 0

    def _consume_turn(
        self,
        lifecycle: TurnLifecycleCoordinator,
        endpoint: EngineEndpoint,
        frame_q: "queue.Queue[Any]",
        relaunch: Callable[[], None],
    ) -> None:
        """Consume frames until terminal evidence, re-attaching once if needed."""
        deadline = time.monotonic() + min(max(self.timeout, self.min_deadline_s), 300.0)
        saw_running = False
        baseline_open = True
        reattached = False

        while time.monotonic() < deadline:
            try:
                data = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(data, Exception):
                lifecycle.handle_disconnect(str(data))
                raise data
            if data is None:
                # Stream ended without terminal evidence: re-attach once while
                # the turn is young (a fresh subscription replays recent steps).
                if not reattached and time.monotonic() < deadline:
                    reattached = True
                    saw_running = False  # re-baseline: each stream's pre-run snapshot is not this turn's terminal
                    relaunch()
                    continue
                break
            if data is _STREAM_OPENED:
                continue  # re-attachment accepted; not a trajectory frame

            update = data.get("update") if isinstance(data.get("update"), dict) else data
            status = str(update.get("status") or "")
            if status == CASCADE_RUNNING:
                saw_running = True
                baseline_open = False
            if not saw_running:
                if baseline_open:
                    current_baseline = lifecycle.observation.baseline_step_count or 0
                    lifecycle.observation.baseline_step_count = max(
                        current_baseline, self._baseline_step_count(data)
                    )
                continue  # pre-run snapshot / replayed prior-turn history

            parsed = parse_agent_state_update(data)
            reduction = reduce_update(lifecycle.observation, parsed)
            if reduction.is_failed:
                if reduction.error == "upstream_empty_response":
                    lifecycle.observation.error = "upstream completed without attributable output"
                    raise UpstreamEmptyResponse(
                        "Upstream completed without emitting attributable text or tools"
                    )
                raise UpstreamError(reduction.error or "upstream trajectory failed")
            if reduction.is_completed:
                return

        # Loop ended without terminal evidence. Silence never proves completion.
        if saw_running or lifecycle.observation.emitted_text_chunks:
            lifecycle.handle_timeout("deadline without terminal evidence")
            raise UpstreamTimeout(
                "Turn deadline reached without terminal evidence from upstream; "
                "outcome unknown, duplicate submission forbidden"
            )
        lifecycle.handle_disconnect("stream ended before run evidence")
        raise UpstreamError(
            "Stream ended before this turn produced run evidence; outcome not attributable"
        )

    def _stop_invocation(self, endpoint: EngineEndpoint, cascade_id: str) -> None:
        """Best-effort upstream stop; never raises (cleanup path)."""
        for method in ("ForceStopCascadeTree", "CancelCascadeInvocation"):
            try:
                self.transport.unary(endpoint, method, {"cascadeId": cascade_id}, timeout=10)
            except Exception:
                continue

    # -- lifecycle helpers ---------------------------------------------------

    def _pre_send_fail(self, lifecycle: TurnLifecycleCoordinator, endpoint: EngineEndpoint,
                       ex: BaseException) -> None:
        lifecycle.handle_failure_before_send(str(ex))
        if isinstance(ex, AgyBridgeError):
            raise ex
        raise UpstreamError(f"Pre-send failure: {ex}") from ex

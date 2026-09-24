"""FastAPI application for agy-bridge (Section 12)."""
from __future__ import annotations

import time
import uuid
import asyncio
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from agy_bridge.engine.agent import MODEL_MAP

from agy_bridge.errors import (
    AgyBridgeError,
    RateLimitExceeded,
    UnsupportedInputError,
    UnsupportedImageError,
)
from agy_bridge.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
)
from agy_bridge.api.sse import (
    format_sse_chunk,
    format_sse_done,
    format_sse_error,
)

KNOWN_MODELS = [
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium", "gemini-3.8-flash-low",
    "gemini-3.7-flash-high", "gemini-3.7-flash-medium", "gemini-3.7-flash-low",
    "gemini-3.6-flash-high", "gemini-3.6-flash-medium", "gemini-3.6-flash-low",
    "gemini-3.1-pro-high", "gemini-3.1-pro-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking", "gpt-oss-120b-medium", "gemini-3.5-flash-lite",
    "gemini-3.5-flash", "gemini-flash", "gemini-pro", "agy", "agy-auto",
]


def resolve_model(model: Optional[str]) -> str:
    default_model = os.environ.get("AGY_BRIDGE_MODEL", "gemini-3.5-flash-lite")
    fast_model = os.environ.get("AGY_FLASH_MODEL", "gemini-3.8-flash")
    requested = default_model if model is None else model
    if not requested or requested in ("agy", "agy-auto", "auto", "default"):
        return fast_model
    if requested in ("gemini-flash", "flash"):
        return "gemini-3.8-flash-medium"
    if requested in ("gemini-pro", "pro"):
        return "gemini-3.8-flash-high"
    return requested


def create_app(engine: Optional[Any] = None) -> FastAPI:
    app = FastAPI(title="AGY Bridge", version="0.2.0")

    # In-memory mock store for conversations (bounded to 128)
    conversations: Dict[str, Dict[str, Any]] = {}

    async def active_lane_ids():
        if engine and hasattr(engine, "lanes"):
            return await run_in_threadpool(engine.lanes)
        return list(conversations)

    @app.exception_handler(AgyBridgeError)
    async def bridge_error_handler(request: Request, exc: AgyBridgeError):
        headers = {}
        if isinstance(exc, RateLimitExceeded) and exc.retry_after_s:
            headers["Retry-After"] = str(exc.retry_after_s)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(), headers=headers)

    @app.get("/")
    @app.get("/health")
    @app.get("/health/live")
    @app.get("/health/ready")
    async def health(request: Request):
        path = request.url.path
        if "live" in path:
            return {"status": "live"}
        result = {"ok": False}
        try:
            if engine and hasattr(engine, "health"):
                result = await run_in_threadpool(engine.health)
        except Exception:
            result = {"ok": False}
        healthy = bool(result.get("ok"))
        if "ready" in path:
            return JSONResponse({"status": "ready" if healthy else "degraded"}, status_code=200 if healthy else 503)
        return {"status": "healthy" if healthy else "degraded", "app_lane": result,
                "active_lanes": len(await active_lane_ids()), "port": int(os.environ.get("AGY_BRIDGE_PORT", "8790"))}

    @app.get("/v1/models")
    async def list_models():
        data = [
            {
                "id": m,
                "object": "model",
                "created": 1726000000,
                "owned_by": "agy",
            }
            for m in KNOWN_MODELS
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/bridge/capabilities")
    async def capabilities():
        return {
            "tools": {
                "supported": True,
                "mode": "verified_envelope",
            },
            "vision": {
                "policy": "legacy_placeholders",
                "supported": True,
            },
            "streaming": {
                "mode": "buffered_sse",
                "heartbeat": True,
            },
            "conversations": {
                "supported": True,
                "ttl_minutes": 30,
                "max_retained": 128,
            },
        }

    @app.post("/v1/conversations")
    async def create_conversation(req: Optional[Dict[str, Any]] = None):
        conv_id = ((req or {}).get("conversation_id") or "").strip() or f"lane_{uuid.uuid4().hex[:8]}"
        now = time.time()
        conversations[conv_id] = {"updated_at": now}
        if engine and hasattr(engine, "acquire_for_lane"):
            engine.acquire_for_lane(conv_id)
        return {"conversation_id": conv_id, "status": "open", "model": "gemini-3.8-flash",
                "active": len(await active_lane_ids()), "pool_size": "app-lane (elastic)"}

    @app.get("/v1/conversations")
    async def list_conversations():
        lanes = [{"id": cid, "active": True, "pinned_pool_slot": None} for cid in conversations]
        if engine and hasattr(engine, "lanes"):
            lanes = [{"id": cid, "active": True, "pinned_pool_slot": None} for cid in await active_lane_ids()]
        return {"object": "list", "data": lanes, "active": len(lanes), "pool_size": "app-lane (elastic)"}

    @app.delete("/v1/conversations/")
    async def delete_missing_conversation():
        return JSONResponse(status_code=400, content={"error": {"message": "conversation_id required",
                                                                 "type": "invalid_request_error"}})

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        conversations.pop(conversation_id, None)
        if engine and hasattr(engine, "release_lane"):
            await run_in_threadpool(engine.release_lane, conversation_id)
        return {"conversation_id": conversation_id, "status": "closed", "active": len(await active_lane_ids())}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Validate model
        requested_model = req.model
        concrete_model = resolve_model(requested_model)
        if (requested_model is not None and requested_model not in KNOWN_MODELS
                and requested_model not in MODEL_MAP and requested_model not in ("", "flash", "pro", "auto", "default")):
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Model {requested_model!r} not found or unsupported",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                        "param": "model",
                    }
                },
            )

        # Validate n=1
        if req.n != 1:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Only n=1 is currently supported by agy-bridge",
                        "type": "invalid_request_error",
                        "code": "unsupported_parameter",
                        "param": "n",
                    }
                },
            )

        if not engine:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "No completion engine configured",
                        "type": "upstream_error",
                        "code": "upstream_unavailable",
                        "param": None,
                    }
                },
            )

        req_data = req.model_dump()
        req_data["model"] = concrete_model
        if req.conversation_id:
            req_data["messages"] = [m.model_dump() for m in req.messages if m.role == "user"][-1:]
        if req.stream:
            async def stream_generator():
                task = asyncio.create_task(run_in_threadpool(engine.execute_completion, req_data))
                yield format_sse_chunk(req_id, requested_model or concrete_model, delta_content="", delta_role="assistant")
                try:
                    while not task.done():
                        done, _ = await asyncio.wait({task}, timeout=15.0)
                        if not done:
                            yield format_sse_chunk(req_id, requested_model or concrete_model, delta_content="")
                    result = await task
                except AgyBridgeError as exc:
                    yield format_sse_error(exc.to_dict())
                    yield format_sse_done()
                    return
                content = result.get("content", "")
                if result.get("tool_calls"):
                    yield format_sse_chunk(req_id, requested_model or concrete_model, delta_tool_calls=result["tool_calls"])
                    yield format_sse_chunk(req_id, requested_model or concrete_model, finish_reason="tool_calls")
                else:
                    if content:
                        yield format_sse_chunk(req_id, requested_model or concrete_model, delta_content=content)
                    yield format_sse_chunk(req_id, requested_model or concrete_model, finish_reason="stop")
                yield format_sse_done()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # Non-streaming
        result = await run_in_threadpool(engine.execute_completion, req_data)
        response = ChatCompletionResponse(
            id=req_id,
            model=requested_model or concrete_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=None if result.get("tool_calls") else result.get("content", ""),
                        tool_calls=result.get("tool_calls") or None,
                    ),
                    finish_reason=result.get("finish_reason", "tool_calls" if result.get("tool_calls") else "stop"),
                )
            ],
            usage=result.get("usage") or {},
        )
        return response.model_dump()

    return app

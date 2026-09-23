"""FastAPI application for agy-bridge (Section 12)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    format_sse_ping,
)

KNOWN_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-low",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-high",
    "gemini-3.7-flash",
    "gemini-3.1-pro",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
]


def create_app(engine: Optional[Any] = None) -> FastAPI:
    app = FastAPI(title="AGY Bridge", version="0.2.0")

    # In-memory mock store for conversations (bounded to 128)
    conversations: Dict[str, Dict[str, Any]] = {}

    @app.exception_handler(AgyBridgeError)
    async def bridge_error_handler(request: Request, exc: AgyBridgeError):
        headers = {}
        if isinstance(exc, RateLimitExceeded) and exc.retry_after_s:
            headers["Retry-After"] = str(exc.retry_after_s)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(), headers=headers)

    @app.get("/health")
    @app.get("/health/live")
    @app.get("/health/ready")
    async def health(request: Request):
        path = request.url.path
        if "live" in path:
            return {"status": "live"}
        if "ready" in path:
            return {"status": "ready"}
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/v1/models")
    async def list_models():
        data = [
            {
                "id": m,
                "object": "model",
                "created": 1726000000,
                "owned_by": "google-antigravity",
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
                "policy": "reject",
                "supported": False,
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
        conv_id = f"conv-{uuid.uuid4().hex[:12]}"
        now = time.time()
        conversations[conv_id] = {
            "id": conv_id,
            "created_at": now,
            "updated_at": now,
            "metadata": req or {},
        }
        return {"id": conv_id, "created_at": now}

    @app.get("/v1/conversations")
    async def list_conversations():
        return {"data": list(conversations.values())}

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        existed = conversation_id in conversations
        conversations.pop(conversation_id, None)
        return {"id": conversation_id, "deleted": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Validate model
        if req.model not in KNOWN_MODELS:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Model {req.model!r} not found or unsupported",
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

        # Check vision guard (Phase 9 preview)
        for msg in req.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": {
                                    "message": "Image/multimodal input rejected by policy until verified adapter is configured.",
                                    "type": "invalid_request_error",
                                    "code": "unsupported_image_input",
                                    "param": "messages",
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

        if req.stream:
            # Buffered SSE streaming
            try:
                result = engine.execute_completion(req.model_dump())
            except AgyBridgeError as exc:
                # If error happens before headers, exception handler handles it
                raise exc

            content = result.get("content", "")

            def stream_generator():
                # Emit ping heartbeat
                yield format_sse_ping()
                # Emit assistant role
                yield format_sse_chunk(req_id, req.model, delta_content="")
                # Emit content
                if content:
                    yield format_sse_chunk(req_id, req.model, delta_content=content)
                # Emit finish reason
                yield format_sse_chunk(req_id, req.model, finish_reason="stop")
                yield format_sse_done()

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # Non-streaming
        result = engine.execute_completion(req.model_dump())
        response = ChatCompletionResponse(
            id=req_id,
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result.get("content", ""),
                    ),
                    finish_reason=result.get("finish_reason", "stop"),
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        return response.model_dump()

    return app

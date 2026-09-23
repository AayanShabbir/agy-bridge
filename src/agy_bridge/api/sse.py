"""Server-Sent Events (SSE) formatting for OpenAI streaming."""
import json
import time
from typing import Any, Dict, Optional


def format_sse_chunk(
    request_id: str,
    model: str,
    delta_content: Optional[str] = None,
    delta_tool_calls: Optional[list] = None,
    finish_reason: Optional[str] = None,
) -> str:
    """Format one SSE data frame matching OpenAI streaming chunk contract."""
    delta: Dict[str, Any] = {}
    if delta_content is not None:
        delta["content"] = delta_content
    if delta_tool_calls is not None:
        delta["tool_calls"] = delta_tool_calls

    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def format_sse_error(error_dict: dict) -> str:
    """Format SSE error payload for post-header stream failures."""
    return f"data: {json.dumps(error_dict)}\n\n"


def format_sse_done() -> str:
    return "data: [DONE]\n\n"


def format_sse_ping() -> str:
    return ": ping\n\n"

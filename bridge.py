#!/usr/bin/env python3
"""gateway.py — High-performance persistent inference gateway for Hermes <-> Antigravity.

Provides an OpenAI-compatible /v1/chat/completions endpoint on 127.0.0.1:8790
backed by a persistent Google Antigravity SDK (google.antigravity) daemon for
flash tiers (sub-second median latency, zero cold-boot overhead) with automatic
smart CLI failover (run_agy_smart) for reasoning tiers and quota handling.

Endpoints:
- GET  /v1/models             -> OpenAI model list
- POST /v1/chat/completions   -> OpenAI chat completion (non-streaming + SSE streaming)
- GET  /health                -> Liveness check and runtime stats
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Ensure we run in Python 3.11 venv with google.antigravity
VENV_PYTHON = "/Users/aayan/.hermes/hermes-agent/venv/bin/python"
if sys.executable != VENV_PYTHON and os.path.isfile(VENV_PYTHON):
    try:
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
    except Exception as _reexec_err:
        print(f"Warning: Failed to auto-reexec into {VENV_PYTHON}: {_reexec_err}", file=sys.stderr)

HOME = os.path.expanduser("~")
HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGY_BRIDGE_PORT", "8790"))
AGY = os.environ.get("AGY_BIN", os.path.join(HOME, ".local", "bin", "agy"))
AGY_LANE = os.environ.get("AGY_LANE", "app")   # app = Antigravity app lane (default); sdk = API-key rollback
_APP_MGR = [None]
_APP_RETRIES = 2


def _get_app_manager():
    if _APP_MGR[0] is None:
        import app_lane
        _APP_MGR[0] = app_lane.AppLaneManager(os.environ.get("AGY_APP_REGISTRY", "/veta/app-brains/registry.json"))
    return _APP_MGR[0]


def _app_lane_turn(conversation_id: str, user_text: str, timeout: float = 180.0):
    """Run one turn on the app lane (conversation-pinned or stateless).
    Returns (content, usage_dict). Raises for upstream/registration errors."""
    last_err = None
    for attempt in range(_APP_RETRIES + 1):
        mgr = _get_app_manager()
        try:
            if conversation_id:
                return mgr.chat_lane(conversation_id, user_text, timeout=timeout)
            return mgr.chat(user_text)
        except Exception as ex:
            last_err = ex
            if attempt == _APP_RETRIES:
                break
            time.sleep(1.5)
    raise last_err


def _last_user_text(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


CLI_LANE_MISSING_MSG = (
    "CLI lane unavailable: no agy binary at %s. This bridge serves gemini-* through "
    "the SDK lane only. claude/gpt/deepseek need the (removed) AGY CLI. If the root "
    "cause was a 429 quota error, wait for the free-tier window to reset or route "
    "heavy-context work to a non-bridge provider." % AGY
)
LOG = os.path.join(HOME, ".hermes", "agy-bridge", "bridge.log")
AGY_ACCOUNT = os.path.join(HOME, ".hermes", "bin", "agy-account.py")

# Automatically load ~/.hermes/.env for credentials (GEMINI_API_KEY, etc.)
ENV_FILE = os.path.join(HOME, ".hermes", ".env")
if os.path.isfile(ENV_FILE):
    try:
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.split("#")[0].strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

DEFAULT_MODEL = os.environ.get("AGY_BRIDGE_MODEL", "gemini-3.5-flash-lite")
FAST_FLASH_MODEL = os.environ.get("AGY_FLASH_MODEL", "gemini-3.8-flash")
LOW_MODEL = os.environ.get("AGY_LOW_MODEL", "gemini-3.1-flash-lite")   # paid-served lite tier
PRO_MODEL = os.environ.get("AGY_PRO_MODEL", "gemini-3.8-flash")   # high tier: same model + HIGH thinking
TIMEOUT = os.environ.get("AGY_BRIDGE_TIMEOUT", "300s")

MODELS = [
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash",
    "gemini-pro",
    "agy",
    "agy-auto",
]

CLI_MODELS = [
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
]

# Base fallback ladder for CLI queries
MODEL_LADDER = [
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.7-flash-medium",
    "gemini-3.6-flash-medium",
]


def get_fallback_ladder(target_model: str) -> List[str]:
    """Generate an intelligent fallback ladder that respects the requested reasoning tier."""
    m = target_model.lower()
    if "-high" in m or "claude" in m or "thinking" in m:
        candidates = [
            target_model,
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-medium",
            "gemini-3.7-flash-high",
            "gemini-3.8-flash-low",
        ]
    elif "-medium" in m or "120b" in m:
        candidates = [
            target_model,
            "gemini-3.8-flash-medium",
            "gemini-3.7-flash-medium",
            "gemini-3.8-flash-low",
        ]
    else:
        candidates = [
            target_model,
            "gemini-3.8-flash-low",
            "gemini-3.7-flash-low",
            "gemini-3.6-flash-low",
        ]
    ladder = []
    for c in candidates:
        if c in CLI_MODELS and c not in ladder:
            ladder.append(c)
    for c in MODEL_LADDER:
        if c in CLI_MODELS and c not in ladder:
            ladder.append(c)
    return ladder

_SEM_SLOTS = int(os.environ.get("AGY_BRIDGE_CONCURRENCY", "4"))
_cli_sem = threading.BoundedSemaphore(_SEM_SLOTS)

# Import Antigravity SDK
try:
    from google.antigravity import (
        Agent,
        CapabilitiesConfig,
        GeminiAPIEndpoint,
        GeminiModelOptions,
        LocalAgentConfig,
        ModelAPIRetryConfig,
        RetryConfig,
        ThinkingLevel,
    )
    from google.antigravity.models import ModelTarget
    from google.antigravity.types import Text
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def log_line(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def resolve_model(name: str) -> str:
    """Map requested model name onto concrete target model id."""
    if not name or name in ("agy-auto", "auto", "default", "agy"):
        return FAST_FLASH_MODEL
    if name in ("gemini-flash", "flash"):
        return "gemini-3.8-flash-medium"
    if name in ("gemini-pro", "pro"):
        return "gemini-3.1-pro-high"
    return name


def resolve_model_for_cli(name: str, effort: Optional[str] = None) -> str:
    """Map model name onto a valid CLI model id for agy CLI subprocess."""
    eff = (effort or "medium").lower()
    if not name or name in ("agy-auto", "auto", "default", "gemini-flash", "flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "agy"):
        if eff == "high":
            return "gemini-3.8-flash-high"
        elif eff == "low":
            return "gemini-3.8-flash-low"
        return "gemini-3.8-flash-medium"

    if name in CLI_MODELS:
        return name

    for base in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"):
        if name.startswith(base):
            cand = f"{base}-{eff}"
            if cand in CLI_MODELS:
                return cand
            return f"{base}-medium"

    for base in ("gemini-3.1-pro",):
        if name.startswith(base):
            if eff in ("high", "medium"):
                return f"{base}-high"
            return f"{base}-low"

    if name.startswith("claude-sonnet"):
        return "claude-sonnet-4-6"
    if name.startswith("claude-opus"):
        return "claude-opus-4-6-thinking"
    if "gpt-oss" in name:
        return "gpt-oss-120b-medium"

    return "gemini-3.8-flash-medium"


def is_flash_tier(model: str) -> bool:
    """Determine if a model can run on the persistent low-latency flash tier."""
    # AGY_FORCE_CLI=1 (paid subscription route): NEVER route through the free-tier
    # SDK pool — every request goes to the agy CLI/subscription lane instead.
    if os.environ.get("AGY_FORCE_CLI"):
        return False
    if not _SDK_AVAILABLE:
        return False
    if not model:
        return True
    m = model.lower()
    # Every gemini-* model goes to the SDK lane (native Python, works in the Linux
    # container). Only claude/gpt/deepseek genuinely need the CLI binary lane.
    if m.startswith("gemini") or m in ("flash", "agy-auto", "auto", "agy"):
        return True
    if any(k in m for k in ("claude", "gpt", "deepseek", "-high", "-medium", "pro")):
        return False
    if any(k in m for k in ("lite", "flash-lite", "minimal", "fast")):
        return True
    return False


def is_quota_error(err_dict: Any) -> bool:
    s = str(err_dict or {}).lower()
    return any(
        k in s
        for k in (
            "quota",
            "429",
            "rate limit",
            "resource exhausted",
            "resource_exhausted",
            "billing",
            "limit: 15",
            "limit: 5",
            "503",
            "high demand",
            "temporarily unavailable",
            "try again later",
        )
    )


def _text_of(content: Any) -> str:
    """Flatten OpenAI content to plain text with multimodal awareness."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(str(p.get("text", "")))
                elif p.get("type") == "image_url":
                    img_info = p.get("image_url") or {}
                    url = img_info.get("url") if isinstance(img_info, dict) else str(img_info)
                    if url.startswith("data:"):
                        out.append(f"[Embedded Image data ({len(url)} bytes)]")
                    else:
                        out.append(f"[Image URL: {url}]")
                else:
                    out.append(str(p))
            else:
                out.append(str(p))
        return "\n".join(out)
    if content is None:
        return ""
    return str(content)


def build_tool_catalog(tools: list) -> str:
    """Render OpenAI tool definitions into a compact catalog for AGY."""
    if not tools:
        return ""
    lines = ["TOOL CATALOG (you may call these):"]
    for t in tools:
        fn = (t.get("function") or {}) if isinstance(t, dict) else {}
        name = fn.get("name", "?")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        desc = fn.get("description", "")
        arg_sig = ", ".join(
            "%s%s" % (k, "" if (props.get(k) or {}).get("type") == "string" else "=?")
            for k in list(props.keys())[:12]
        )
        lines.append(f"- {name}({arg_sig}): {desc[:200]}")
    return "\n".join(lines)


def extract_tool_call_args(tc: dict, fn: dict) -> str:
    """Extract arguments from tool call dictionary, handling 'arguments', 'parameters', and 'args'."""
    raw_args = None
    for source in (tc, fn):
        if not isinstance(source, dict):
            continue
        for key in ("arguments", "parameters", "args"):
            if key in source and source[key] is not None:
                raw_args = source[key]
                break
        if raw_args is not None:
            break

    if raw_args is None:
        return "{}"
    if isinstance(raw_args, str):
        return raw_args
    try:
        return json.dumps(raw_args)
    except Exception:
        return "{}"


def clean_tool_call_content(content: Any) -> Optional[str]:
    """Ensure assistant message content is None when tool calls are present, unless non-envelope text exists."""
    if not content or not isinstance(content, str):
        return None
    c_str = content.strip()
    if not c_str:
        return None
    c_clean = c_str
    if c_clean.startswith("```"):
        c_clean = re.sub(r"^```(?:json)?\s*", "", c_clean)
        c_clean = re.sub(r"\s*```$", "", c_clean).strip()
    if (c_clean.startswith("{") or c_clean.startswith("[")) and any(
        k in c_clean for k in ("tool_calls", "name", "function", '"content": null', '"content":null')
    ):
        try:
            c_obj = json.loads(c_clean)
            if isinstance(c_obj, dict):
                inner = c_obj.get("content")
                if inner and isinstance(inner, str) and inner.strip():
                    return clean_tool_call_content(inner)
                return None
            elif isinstance(c_obj, list):
                return None
        except Exception:
            pass
    return c_str


def build_envelope_schema(
    require_tool: bool = False,
    target_function: Optional[str] = None,
    tools: Optional[list] = None,
) -> dict:
    """JSON schema that forces agy CLI final answer into {content, tool_calls}."""
    name_prop: Dict[str, Any] = {"type": "string"}
    allowed_names = []
    if target_function:
        allowed_names = [target_function]
    elif tools:
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                fn_name = fn.get("name")
                if fn_name and fn_name not in allowed_names:
                    allowed_names.append(fn_name)

    if allowed_names:
        name_prop["enum"] = allowed_names

    tool_call_item = {
        "type": "object",
        "properties": {
            "name": name_prop,
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }

    tool_calls_prop: Dict[str, Any] = {
        "type": "array",
        "items": tool_call_item,
    }
    if require_tool:
        tool_calls_prop["minItems"] = 1

    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The complete, comprehensive, detailed assistant markdown response when no tool is called. "
                    "Do NOT summarize, truncate, or output brief status sentences here. Provide the full, uncompressed content."
                ),
            },
            "tool_calls": tool_calls_prop,
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def messages_to_prompt(messages: list, tools: Optional[list] = None, tool_choice: Any = None) -> str:
    """Render OpenAI messages + tool framing into the AGY prompt."""
    parts = []
    if not tools or tool_choice == "none":
        parts.append(
            "CRITICAL INSTRUCTION: You are serving as an OpenAI-compatible text inference "
            "backend. You have ZERO local tool execution permissions. Do NOT call run_command, "
            "view_file, write_to_file, or any built-in tools. Output only your direct text response."
        )
    else:
        parts.append(
            "CRITICAL INSTRUCTION: You are serving as an OpenAI-compatible function calling backend. "
            "You have ZERO local execution permissions. Do NOT execute built-in tools (run_command, view_file, etc.) directly. "
            "To use tools, format your decision as a structured JSON object according to the TOOL CATALOG below."
        )
        parts.append(build_tool_catalog(tools))
        if isinstance(tool_choice, dict) and "function" in tool_choice:
            target_fn = (tool_choice.get("function") or {}).get("name", "")
            parts.append(
                f"STRICT OUTPUT RULE: You MUST call the specific tool '{target_fn}' in this turn. "
                "Respond ONLY with a valid JSON object matching this schema:\n"
                f'{{"content": null, "tool_calls": [{{"name": "{target_fn}", "arguments": {{<json_arguments>}}}}]}}\n'
                "Do not describe the call; do not answer in prose."
            )
        elif tool_choice == "required":
            parts.append(
                "STRICT OUTPUT RULE: You MUST call exactly one tool from the TOOL CATALOG "
                "in this turn. Respond ONLY with a valid JSON object matching this schema:\n"
                '{"content": null, "tool_calls": [{"name": "<function_name>", "arguments": {<json_arguments>}}]}\n'
                "Do not describe the call; do not answer in prose."
            )
        else:
            parts.append(
                "OUTPUT RULE: If the user's request requires a tool, respond ONLY with a JSON object:\n"
                '{"content": null, "tool_calls": [{"name": "<function_name>", "arguments": {<json_arguments>}}]}\n'
                "If NO tool is needed, respond directly with your text response."
            )

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            name = m.get("name") or ""
            if not name and isinstance(content, list):
                name = " ".join(
                    p.get("name", "") for p in content if isinstance(p, dict) and p.get("type") == "tool_call_id"
                )
            tool_id = m.get("tool_call_id", "")
            id_attr = f' id="{tool_id}"' if tool_id else ""
            name_attr = f' name="{name}"' if name else ""
            parts.append(f"<tool_result{id_attr}{name_attr}>\n{_text_of(content)}\n</tool_result>")
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = tc.get("name") or fn.get("name") or "?"
                args = tc.get("arguments") if "arguments" in tc else fn.get("arguments", {})
                if isinstance(args, str):
                    args_str = args
                else:
                    try:
                        args_str = json.dumps(args, separators=(",", ":"))
                    except Exception:
                        args_str = "{}"
                parts.append(
                    '<assistant_tool_call name="%s" arguments=%s>'
                    % (fn_name, args_str)
                )
            if content:
                parts.append(f"<assistant>\n{_text_of(content)}\n</assistant>")
            continue
        parts.append(f"<{role}>\n{_text_of(content)}\n</{role}>")

    return "\n\n".join(parts)


INTERNAL_TOOL_NAMES = {"finish", "final_answer", "done", "complete"}


def _normalize_single_tool_call(item: Any, valid_names: Optional[set] = None) -> Optional[dict]:
    """Normalize tool call representation from various model formats into standard dict."""
    if not isinstance(item, dict):
        return None
    name = None
    args = None
    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        fn = item["function"]
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("parameters") or fn.get("args") or {}
    elif "function_call" in item and isinstance(item["function_call"], dict):
        fn = item["function_call"]
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("parameters") or fn.get("args") or {}
    elif "name" in item:
        name = item["name"]
        args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
    elif "function" in item and isinstance(item["function"], str):
        name = item["function"]
        args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
    elif "action" in item and isinstance(item["action"], str):
        name = item["action"]
        args = item.get("action_input") or item.get("arguments") or item.get("parameters") or {}

    if not name or not isinstance(name, str):
        return None

    if name.lower() in INTERNAL_TOOL_NAMES and (not valid_names or name not in valid_names):
        return None

    tc = {"name": name, "arguments": args if args is not None else {}}
    if "id" in item:
        tc["id"] = item["id"]
    return tc


def _clean_accumulated_text(acc: str) -> str:
    """Strip trailing JSON schema artifacts and markdown wrapping from intermediate steps."""
    if not acc:
        return ""
    cleaned = re.sub(r'```(?:json)?\s*\{.*?\"tool_calls\":\s*\[\]\}\s*```\s*$', '', acc.strip(), flags=re.DOTALL).strip()
    cleaned = re.sub(r'\{.*?\"tool_calls\":\s*\[\]\}\s*$', '', cleaned, flags=re.DOTALL).strip()
    return cleaned


def _extract_finish_response(item: Any) -> str:
    """Extract any content or response embedded inside an internal finish tool call."""
    if not isinstance(item, dict):
        return ""
    name = item.get("name") or (item.get("function") if isinstance(item.get("function"), str) else "")
    if str(name).lower() in INTERNAL_TOOL_NAMES:
        args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
        if isinstance(args, dict):
            return str(args.get("response") or args.get("content") or args.get("message") or "")
        elif isinstance(args, str):
            return args
    return ""


def parse_envelope(raw_text: Any, tools: Optional[list] = None, result_obj: Optional[dict] = None) -> Tuple[str, list]:
    """Return (content, tool_calls) from response text or structured object.
    
    CRITICAL: When tools are NOT requested, returns raw_text as content directly
    without stripping or corrupting user-requested JSON outputs.
    Supports JSON dict envelopes, nested content envelopes, JSON arrays, and ndjson streams.
    """
    if not tools:
        if isinstance(raw_text, (dict, list)):
            return json.dumps(raw_text), []
        return str(raw_text or ""), []

    valid_tool_names = set()
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if fn_name := fn.get("name"):
                valid_tool_names.add(fn_name)

    if isinstance(raw_text, list):
        valid_tcs = []
        finish_responses = []
        for item in raw_text:
            tc = _normalize_single_tool_call(item, valid_names=valid_tool_names)
            if tc:
                valid_tcs.append(tc)
            elif fr := _extract_finish_response(item):
                finish_responses.append(fr)
            elif isinstance(item, dict) and "tool_calls" in item and isinstance(item["tool_calls"], list):
                for sub_tc in item["tool_calls"]:
                    normalized = _normalize_single_tool_call(sub_tc, valid_names=valid_tool_names)
                    if normalized:
                        valid_tcs.append(normalized)
        if valid_tcs:
            return "", valid_tcs
        if finish_responses:
            return "\n".join(finish_responses), []
        return json.dumps(raw_text), []

    if isinstance(raw_text, dict):
        finish_responses = []
        candidate_lists = []
        for k in ("tool_calls", "calls", "functions"):
            val = raw_text.get(k)
            if isinstance(val, list):
                candidate_lists.extend(val)
            elif isinstance(val, dict):
                candidate_lists.append(val)

        valid_tcs = []
        for item in candidate_lists:
            tc = _normalize_single_tool_call(item, valid_names=valid_tool_names)
            if tc:
                valid_tcs.append(tc)
            elif fr := _extract_finish_response(item):
                finish_responses.append(fr)

        c = raw_text.get("content") or ""
        if valid_tcs:
            return c, valid_tcs

        direct_tc = _normalize_single_tool_call(raw_text, valid_names=valid_tool_names)
        if direct_tc:
            return "", [direct_tc]

        if c:
            nested_c, nested_tcs = parse_envelope(c, tools=tools, result_obj=result_obj)
            if nested_tcs:
                return nested_c, nested_tcs

        if finish_responses and not c:
            return "\n".join(finish_responses), []

        # Substantial text recovery: if tool_calls is empty, check if result_obj holds
        # the full analytical markdown that was summarized into content:
        if result_obj:
            acc = result_obj.get("_accumulated_text") or result_obj.get("response") or ""
            cleaned_acc = _clean_accumulated_text(acc)
            if cleaned_acc and len(cleaned_acc) > len(c) + 60:
                return cleaned_acc, []
            elif not c and cleaned_acc:
                return cleaned_acc, []

        if raw_text.get("content"):
            return raw_text.get("content"), []
        # Fall through to result_obj inspection if dict content was empty

    if isinstance(raw_text, str):
        txt = raw_text.strip()
    elif isinstance(raw_text, (dict, list)):
        txt = ""
    else:
        txt = str(raw_text or "").strip()

    if not txt:
        if result_obj:
            captured = result_obj.get("_captured_tool_calls") or []
            for tc in captured:
                tname = tc.get("name")
                params = tc.get("parameters") or {}
                if tname == "run_command":
                    cmd = params.get("CommandLine") or params.get("command") or ""
                    if "terminal" in valid_tool_names:
                        return "", [{"name": "terminal", "arguments": {"command": cmd}}]
                    elif "run_command" in valid_tool_names:
                        return "", [{"name": "run_command", "arguments": params}]
                    elif "bash" in valid_tool_names:
                        return "", [{"name": "bash", "arguments": {"command": cmd}}]
                    elif "execute_code" in valid_tool_names:
                        return "", [{"name": "execute_code", "arguments": {"code": cmd, "language": "bash"}}]
                elif tname in valid_tool_names:
                    return "", [{"name": tname, "arguments": params}]

            recovered = result_obj.get("_accumulated_text") or result_obj.get("response") or ""
            if recovered and recovered.strip():
                cleaned_rec = _clean_accumulated_text(recovered)
                if cleaned_rec:
                    return cleaned_rec, []
                return recovered, []
        return "", []

    cleaned = txt
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, (dict, list)):
            c, tcs = parse_envelope(obj, tools=tools)
            if tcs:
                return c, tcs
            if c and c != json.dumps(obj):
                return c, []
            return txt, []
    except Exception:
        pass

    try:
        decoder = json.JSONDecoder()
        pos = 0
        decoded_objs = []
        while pos < len(cleaned):
            while pos < len(cleaned) and cleaned[pos].isspace():
                pos += 1
            if pos >= len(cleaned):
                break
            try:
                val, next_pos = decoder.raw_decode(cleaned, pos)
                decoded_objs.append(val)
                pos = next_pos
            except Exception:
                p1 = cleaned.find("{", pos + 1)
                p2 = cleaned.find("[", pos + 1)
                candidates = [p for p in (p1, p2) if p != -1]
                if not candidates:
                    break
                pos = min(candidates)

        if decoded_objs:
            collected_tcs = []
            text_parts = []
            for dobj in decoded_objs:
                c, n_tcs = parse_envelope(dobj, tools=tools)
                if n_tcs:
                    collected_tcs.extend(n_tcs)
                if c and c != json.dumps(dobj):
                    text_parts.append(c)
            if collected_tcs:
                return "\n".join(text_parts), collected_tcs
    except Exception:
        pass

    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, dict):
                c, tcs = parse_envelope(obj, tools=tools)
                if tcs:
                    return c, tcs
                if c and c != json.dumps(obj):
                    return c, []
    except Exception:
        pass

    try:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, list):
                c, tcs = parse_envelope(obj, tools=tools)
                if tcs:
                    return c, tcs
    except Exception:
        pass

    if result_obj:
        captured = result_obj.get("_captured_tool_calls") or []
        for tc in captured:
            tname = tc.get("name")
            params = tc.get("parameters") or {}
            if tname == "run_command":
                cmd = params.get("CommandLine") or params.get("command") or ""
                if "terminal" in valid_tool_names:
                    return "", [{"name": "terminal", "arguments": {"command": cmd}}]
                elif "run_command" in valid_tool_names:
                    return "", [{"name": "run_command", "arguments": params}]
                elif "bash" in valid_tool_names:
                    return "", [{"name": "bash", "arguments": {"command": cmd}}]
                elif "execute_code" in valid_tool_names:
                    return "", [{"name": "execute_code", "arguments": {"code": cmd, "language": "bash"}}]
            elif tname in valid_tool_names:
                return "", [{"name": tname, "arguments": params}]

        recovered = result_obj.get("_accumulated_text") or result_obj.get("response") or ""
        if recovered and recovered.strip():
            cleaned_rec = re.sub(r'\{"content":.*?"tool_calls":\[\]\}', '', recovered).strip()
            if cleaned_rec:
                return cleaned_rec, []
            return recovered, []

    return txt, []


# ---------------------------------------------------------------------------
# Persistent SDK Agent Pool (Zero Cold-Boot Overhead + FIFO Queue Dispatch)
# ---------------------------------------------------------------------------
class PersistentAgentPool:
    """Manages a pool of persistent Google Antigravity Agent instances."""

    def __init__(self, model_name: str = FAST_FLASH_MODEL, pool_size: int = 3,
                 thinking_level: Any = None):
        self.model_name = model_name
        self.pool_size = pool_size
        self.thinking_level = thinking_level
        self._ThinkingLevel = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.agents: List[Any] = []
        self._locks: List[asyncio.Lock] = []
        self._available_queue: Optional[asyncio.Queue] = None
        self._pool_lock = threading.Lock()
        self._started = False
        self._turn_count = 0
        self._error_count = 0
        self._last_used = 0.0
        # Conversation lanes: conv_id -> held slot idx (slot NOT returned to pool
        # until the lane closes). A lane keeps its agent's history across turns.
        self._lanes: Dict[str, int] = {}
        self._lane_last: Dict[str, float] = {}

    def start(self) -> None:
        if not _SDK_AVAILABLE:
            return
        with self._pool_lock:
            if self._started:
                return
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="agy-sdk-pool")
            self.thread.start()
            fut = asyncio.run_coroutine_threadsafe(self._init_pool(), self.loop)
            fut.result(timeout=35)
            self._started = True

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_agent(self, idx: int) -> Any:
        g_key = os.environ.get("GEMINI_API_KEY")
        o_key = os.environ.get("GOOGLE_API_KEY")
        # vetafleet key is the generation-enabled lane (personal AQ key 402s on generateContent).
        api_key = g_key or o_key
        # Strict fail-fast retry config (max_retries=0):
        # On 429 quota exhaustion, fails fast (~150ms) instead of sleeping for localharness's
        # 20-60 second retryDelay, allowing immediate, non-blocking failover to smart CLI.
        api_retry = ModelAPIRetryConfig(max_retries=0)
        retry_cfg = RetryConfig(api_retry=api_retry)
        tl = self.thinking_level
        if tl is None:
            from google.antigravity import ThinkingLevel
            tl = ThinkingLevel.MINIMAL
        endpoint = GeminiAPIEndpoint(api_key=api_key, options=GeminiModelOptions(thinking_level=tl))
        model_target = ModelTarget(name=self.model_name, endpoint=endpoint)
        caps = CapabilitiesConfig(enabled_tools=[])
        config = LocalAgentConfig(model=model_target, capabilities=caps, api_key=api_key, retry_config=retry_cfg)
        agent = Agent(config)
        await agent.__aenter__()
        return agent

    async def _init_pool(self) -> None:
        self.agents = []
        self._locks = [asyncio.Lock() for _ in range(self.pool_size)]
        self._available_queue = asyncio.Queue()
        for i in range(self.pool_size):
            agent = await self._init_agent(i)
            self.agents.append(agent)
            self._available_queue.put_nowait(i)

    async def _restart_slot(self, idx: int) -> None:
        try:
            if idx < len(self.agents) and self.agents[idx]:
                await self.agents[idx].__aexit__(None, None, None)
        except Exception:
            pass
        self.agents[idx] = await self._init_agent(idx)

    async def _acquire_slot(self, timeout: float = 12.0) -> int:
        if self._available_queue is None:
            raise RuntimeError("Agent pool not initialized")
        try:
            return await asyncio.wait_for(self._available_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("All SDK agent slots in pool are busy")

    def _release_slot(self, slot_idx: int) -> None:
        if self._available_queue is not None:
            self._available_queue.put_nowait(slot_idx)

    def execute_chat(self, prompt: str, timeout: float = 15.0) -> Tuple[str, dict]:
        """Execute chat turn synchronously using an available persistent agent."""
        if not self._started:
            self.start()
        fut = asyncio.run_coroutine_threadsafe(self._execute_with_available_agent(prompt, timeout=timeout), self.loop)
        return fut.result(timeout=timeout + 15.0)

    async def _execute_with_available_agent(self, prompt: str, timeout: float = 15.0) -> Tuple[str, dict]:
        slot_idx = await self._acquire_slot(timeout=timeout)
        try:
            async with self._locks[slot_idx]:
                self._turn_count += 1
                self._last_used = time.time()
                agent = self.agents[slot_idx]
                try:
                    resp = await asyncio.wait_for(agent.chat(prompt), timeout=timeout)
                    txt = await asyncio.wait_for(resp.text(), timeout=timeout)
                    usage_dict = {}
                    try:
                        usage = agent.conversation.last_turn_usage
                        if usage:
                            usage_dict = {
                                "prompt_tokens": getattr(usage, "input_tokens", 0),
                                "completion_tokens": getattr(usage, "output_tokens", 0),
                                "total_tokens": getattr(usage, "total_tokens", 0),
                            }
                    except Exception:
                        pass
                    return txt, usage_dict
                except Exception as e:
                    self._error_count += 1
                    if "429" not in str(e).lower() and "quota" not in str(e).lower():
                        try:
                            await self._restart_slot(slot_idx)
                        except Exception:
                            pass
                    raise e
                finally:
                    try:
                        agent.conversation.clear_history()
                    except Exception:
                        pass
        finally:
            self._release_slot(slot_idx)

    def stream_chat(self, prompt: str, out_queue: queue.Queue) -> None:
        """Stream chunks from an available persistent agent into threadsafe queue."""
        if not self._started:
            self.start()
        asyncio.run_coroutine_threadsafe(self._stream_with_available_agent(prompt, out_queue), self.loop)

    async def _stream_with_available_agent(self, prompt: str, out_queue: queue.Queue, timeout: float = 15.0) -> None:
        try:
            slot_idx = await self._acquire_slot(timeout=timeout)
        except Exception as ex:
            out_queue.put(("error", ex))
            return

        try:
            async with self._locks[slot_idx]:
                self._turn_count += 1
                self._last_used = time.time()
                agent = self.agents[slot_idx]
                try:
                    resp = await asyncio.wait_for(agent.chat(prompt), timeout=timeout)
                    async for chunk in resp.chunks:
                        if isinstance(chunk, Text) and chunk.text:
                            out_queue.put(("text", chunk.text))
                    out_queue.put(("done", None))
                except Exception as e:
                    self._error_count += 1
                    if "429" not in str(e).lower() and "quota" not in str(e).lower():
                        try:
                            await self._restart_slot(slot_idx)
                        except Exception:
                            pass
                    out_queue.put(("error", e))
                finally:
                    try:
                        agent.conversation.clear_history()
                    except Exception:
                        pass
        finally:
            self._release_slot(slot_idx)

    # -----------------------------------------------------------------------
    # Conversation lanes: a held slot whose agent keeps history across turns.
    # "1 conversation = 1 lane = 1 Agent". Open pins a slot, turn runs chat
    # WITHOUT clearing history, close clears + releases back to the pool.
    # -----------------------------------------------------------------------
    def _acquire_for_lane(self, conv_id: str, timeout: float = 12.0) -> int:
        """Acquire a pool slot and pin it to a conversation id (held, not auto-released)."""
        slot = self._acquire_slot_sync(timeout)
        self._lanes[conv_id] = slot
        self._lane_last[conv_id] = time.time()
        return slot

    def _acquire_slot_sync(self, timeout: float = 12.0) -> int:
        if not self._started:
            self.start()
        fut = asyncio.run_coroutine_threadsafe(self._acquire_slot(timeout=timeout), self.loop)
        return fut.result(timeout=timeout + 15.0)

    def chat_lane(self, conv_id: str, prompt: str, timeout: float = 20.0) -> Tuple[str, dict]:
        """Run a turn on a pinned lane agent, KEEPING its conversation history."""
        if conv_id not in self._lanes:
            raise KeyError(f"unknown conversation lane: {conv_id}")
        slot = self._lanes[conv_id]
        fut = asyncio.run_coroutine_threadsafe(self._chat_lane_async(slot, prompt, timeout), self.loop)
        return fut.result(timeout=timeout + 15.0)

    async def _chat_lane_async(self, slot: int, prompt: str, timeout: float) -> Tuple[str, dict]:
        async with self._locks[slot]:
            self._turn_count += 1
            self._last_used = time.time()
            agent = self.agents[slot]
            resp = await asyncio.wait_for(agent.chat(prompt), timeout=timeout)
            txt = await asyncio.wait_for(resp.text(), timeout=timeout)
            usage_dict = {}
            try:
                usage = agent.conversation.last_turn_usage
                if usage:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "input_tokens", 0),
                        "completion_tokens": getattr(usage, "output_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
            except Exception:
                pass
            return txt, usage_dict
        # NOTE: NO clear_history() here — lane memory is the point.

    def close_lane(self, conv_id: str) -> bool:
        """Clear a lane's history and release its slot back to the pool."""
        slot = self._lanes.pop(conv_id, None)
        self._lane_last.pop(conv_id, None)
        if slot is None:
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(self._clear_lane_async(slot), self.loop)
            fut.result(timeout=15)
        except Exception:
            pass
        self._release_slot(slot)
        return True

    async def _clear_lane_async(self, slot: int) -> None:
        try:
            agent = self.agents[slot]
            agent.conversation.clear_history()
        except Exception:
            pass

    def active_lanes(self) -> List[str]:
        return list(self._lanes.keys())


# Initialize singleton agent pool (default 3 warm agents)
_sdk_manager = PersistentAgentPool(model_name=FAST_FLASH_MODEL, pool_size=int(os.environ.get("AGY_POOL_SIZE", "3")))
# High-reasoning tier pool: same 3.8-flash model with HIGH thinking level (the API has
# no -high/-low model variants; "high" = thinking budget on the API lane).
_sdk_manager_pro = PersistentAgentPool(model_name=PRO_MODEL, pool_size=int(os.environ.get("AGY_PRO_POOL_SIZE", "2")),
    thinking_level=__import__("google.antigravity", fromlist=["ThinkingLevel"]).ThinkingLevel.HIGH)


def resolve_sdk_model(model: str) -> str:
    """Map requested model onto a PAID-SERVED SDK model name (vetafleet/personal keys
    serve 3.1-pro-preview, 3.5-flash, 3.1-flash-lite; NOT 3.1-pro-high/3.8-*)."""
    m = (model or "").lower()
    # 3.8 flash family (user directive 2026-09-19): -high rides the high pool,
    # lite/-low drops to the lite model, everything else serves 3.8-flash.
    if "pro" in m or m.endswith("-high") or m in ("gemini-pro",):
        return PRO_MODEL
    if "lite" in m or m.endswith("-low") or m in ("gemini-flash-low", "gemini-3.5-flash-lite"):
        return LOW_MODEL
    return FAST_FLASH_MODEL


def _pick_mgr(model: str) -> "PersistentAgentPool":
    return _sdk_manager_pro if resolve_sdk_model(model) == PRO_MODEL else _sdk_manager


# ---------------------------------------------------------------------------
# Smart CLI Fallback (run_agy_smart with account auto-switching)
# ---------------------------------------------------------------------------
def run_agy_stream_oneshot(
    prompt: str, model: str, json_schema=None, effort: Optional[str] = None, on_ping=None
):
    """Run one turn against an agy stream-json process."""
    cmd = [
        AGY, "--print-timeout", TIMEOUT, "--model", model,
        "--input-format", "stream-json", "--output-format", "stream-json",
        "--disable-slash-commands",
    ]
    # Do NOT pass --effort if the model name already specifies high/medium/low
    # to avoid agy's "conflicts with --effort" CLI error
    has_effort_in_name = any(model.endswith(f"-{s}") for s in ("high", "medium", "low"))
    if not has_effort_in_name and effort in ("low", "medium", "high"):
        cmd += ["--effort", effort]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]

    if not os.path.exists(AGY):
        return None, None, {"error": CLI_LANE_MISSING_MSG}
    t0 = time.time()
    child = None
    try:
        child = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        child.stdin.write(json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}) + "\n")
        child.stdin.flush()
        try:
            child.stdin.close()
        except Exception:
            pass

        result_obj = None
        err_text = ""
        accumulated_text = []
        captured_tool_calls = []
        last_ping_time = time.time()

        while True:
            line = child.stdout.readline()
            if not line:
                if child.poll() is not None:
                    break
                if on_ping and (time.time() - last_ping_time > 1.5):
                    on_ping()
                    last_ping_time = time.time()
                continue

            if on_ping and (time.time() - last_ping_time > 1.5):
                on_ping()
                last_ping_time = time.time()

            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                err_text = line[-1500:]
                continue

            if ev.get("event") == "step_update":
                su = ev.get("step_update", {})
                stype = su.get("step_type")
                if stype == "agent_response":
                    d = su.get("text_delta") or ""
                    if d:
                        accumulated_text.append(d)
                elif stype == "tool" and su.get("state") == "ACTIVE":
                    t_info = su.get("tool_info") or {}
                    t_name = su.get("tool_name") or t_info.get("name")
                    t_params = t_info.get("parameters") or {}
                    captured_tool_calls.append({"name": t_name, "parameters": t_params})

            if ev.get("event") == "result":
                result_obj = ev.get("result") or {}
                if not result_obj.get("response") and accumulated_text:
                    result_obj["response"] = "".join(accumulated_text)
                if captured_tool_calls:
                    result_obj["_captured_tool_calls"] = captured_tool_calls
                if accumulated_text:
                    result_obj["_accumulated_text"] = "".join(accumulated_text)
                break

        if result_obj is None:
            stderr_tail = ""
            try:
                stderr_tail = (child.stderr.read() or "")[-1500:]
            except Exception:
                pass
            return None, None, {
                "error": "agy result not received (rc=%s): %s%s"
                % (child.poll(), err_text[:400], stderr_tail[:1100]),
                "cmd": " ".join(cmd[:6]),
            }

        dur = round(time.time() - t0, 3)
        result_obj.setdefault("_bridge_duration", dur)
        if result_obj.get("status") != "SUCCESS":
            status_str = str(result_obj.get("status"))
            detail = result_obj.get("error") or result_obj.get("message") or ""
            err_msg = f"agy status {status_str}: {detail}".strip() if detail else f"agy status {status_str}"
            return None, result_obj, {"error": err_msg, "status": status_str, "detail": detail}
        return result_obj, result_obj, None
    except subprocess.TimeoutExpired:
        return None, None, {"error": f"agy timed out ({TIMEOUT})", "cmd": " ".join(cmd[:6])}
    except Exception as ex:
        return None, None, {"error": f"agy spawn failed: {ex}"}
    finally:
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
            except Exception:
                pass
            try:
                if child.stdout:
                    child.stdout.close()
            except Exception:
                pass
            try:
                if child.stderr:
                    child.stderr.close()
            except Exception:
                pass
            try:
                # Reaping exit status prevents <defunct> zombie process leakage
                child.wait(timeout=2)
            except Exception:
                pass


def run_agy_smart(
    prompt: str, requested_model: str, json_schema=None, effort: Optional[str] = None, on_ping=None
):
    """Execute agy CLI with auto-account failover and dynamic model ladder fallback."""
    if not os.path.exists(AGY):
        return None, None, {"error": CLI_LANE_MISSING_MSG}, requested_model
    target_model = resolve_model_for_cli(requested_model, effort=effort)
    models_to_try = get_fallback_ladder(target_model)

    last_err = None
    last_raw_obj = None
    for model_cand in models_to_try:
        obj, raw_obj, err = run_agy_stream_oneshot(prompt, model_cand, json_schema, effort=effort, on_ping=on_ping)
        if obj and not err:
            return obj, raw_obj, None, model_cand
        last_err = err
        last_raw_obj = raw_obj

        if (is_quota_error(err) or is_quota_error(raw_obj)) and os.path.exists(AGY_ACCOUNT):
            try:
                log_line({"ts": time.time(), "event": "quota_detected", "model": model_cand, "action": "switching_account"})
                sw = subprocess.run(
                    [sys.executable, AGY_ACCOUNT, "round-robin"],
                    capture_output=True, text=True, timeout=60,
                )
                if sw.returncode == 0:
                    obj2, raw_obj2, err2 = run_agy_stream_oneshot(prompt, model_cand, json_schema, effort=effort, on_ping=on_ping)
                    if obj2 and not err2:
                        return obj2, raw_obj2, None, model_cand
                    last_err = err2
                    last_raw_obj = raw_obj2
            except Exception as ex:
                log_line({"ts": time.time(), "event": "account_switch_failed", "error": str(ex)})

    return None, last_raw_obj, last_err, target_model


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # silence default stderr logging

    def _send(self, code: int, payload: dict) -> None:
        try:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, OSError):
            pass  # client went away mid-response; never kill the bridge server

    def _sse(self, payload: dict) -> bytes:
        return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send(
                200,
                {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "agy"} for m in MODELS]},
            )
            return
        if path == "/v1/conversations":
            lanes = _sdk_manager.active_lanes()
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": cid, "active": True, "pinned_pool_slot": _sdk_manager._lanes.get(cid)} for cid in lanes],
                    "active": len(lanes),
                    "pool_size": _sdk_manager.pool_size,
                },
            )
            return
        if path in ("/health", "/"):
            self._send(
                200,
                {
                    "status": "healthy",
                    "sdk_available": _SDK_AVAILABLE,
                    "sdk_started": _sdk_manager._started,
                    "sdk_pool_size": getattr(_sdk_manager, "pool_size", 1),
                    "sdk_turns": _sdk_manager._turn_count,
                    "sdk_errors": _sdk_manager._error_count,
                    "flash_model": FAST_FLASH_MODEL,
                    "port": PORT,
                },
            )
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/v1/conversations/"):
            conv_id = path[len("/v1/conversations/"):].strip()
            if not conv_id:
                self._send(400, {"error": {"message": "conversation_id required", "type": "invalid_request_error"}})
                return
            released = _sdk_manager.close_lane(conv_id)
            if not released:
                self._send(404, {"error": {"message": f"unknown conversation_id: {conv_id}", "type": "invalid_request_error"}})
                return
            self._send(200, {"conversation_id": conv_id, "status": "closed", "active": len(_sdk_manager.active_lanes())})
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})


    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/conversations":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or "0")) or b"{}")
            except Exception:
                body = {}
            if not _SDK_AVAILABLE:
                self._send(503, {"error": {"message": "SDK lane unavailable, cannot open conversation", "type": "server_error"}})
                return
            conv_id = (body.get("conversation_id") or "").strip() or ("lane_%08x" % (time.time_ns() & 0xFFFFFFFF))
            if AGY_LANE == "app":
                # App lane conversations are brain cascades, created lazily on first turn.
                if _APP_MGR[0]:
                    try:
                        _APP_MGR[0].acquire_for_lane(conv_id)
                    except Exception:
                        pass
                self._send(
                    200,
                    {
                        "conversation_id": conv_id,
                        "status": "open",
                        "model": "gemini-3.8-flash",
                        "active": len(_APP_MGR[0].lanes()) if _APP_MGR[0] else 0,
                        "pool_size": "app-lane (elastic)",
                    },
                )
                return
            try:
                _sdk_manager._acquire_for_lane(conv_id)
            except TimeoutError:
                self._send(503, {"error": {"message": "all SDK pool slots are in active use (lanes or concurrent turns)", "type": "server_error"}})
                return
            self._send(
                200,
                {
                    "conversation_id": conv_id,
                    "status": "open",
                    "model": _sdk_manager.model_name,
                    "active": len(_sdk_manager.active_lanes()),
                    "pool_size": _sdk_manager.pool_size,
                },
            )
            return
        if path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except Exception as e:
            self._send(400, {"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})
            return

        model = resolve_model(body.get("model", DEFAULT_MODEL))
        messages = body.get("messages") or []
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        conversation_id = (body.get("conversation_id") or "").strip()
        prompt = messages_to_prompt(messages, tools, tool_choice)
        req_id = "%08x" % (time.time_ns() & 0xFFFFFFFF)

        # Conversation lane: run the turn on the pinned agent, keeping its history.
        # The agent already holds prior context, so send only the latest user message.
        if conversation_id:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        last_user = c
                    elif isinstance(c, list):
                        last_user = " ".join(
                            p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                        )
                    break
            if not last_user.strip():
                self._send(400, {"error": {"message": "conversation lane requires a user message", "type": "invalid_request_error"}})
                return
            t0 = time.time()
            if AGY_LANE == "app":
                # App lane: pinned conversation = one brain cascade (memory kept server-side).
                try:
                    raw_text, usage_info = _app_lane_turn(conversation_id, last_user, timeout=180.0)
                except Exception as ex:
                    self._send(502, {"error": {"message": f"app lane upstream error: {ex}", "type": "upstream_error"}})
                    return
                _sdk_manager._lane_last[conversation_id] = time.time()
                log_line({"ts": time.time(), "req": req_id, "lane": "app-convo", "conv_id": conversation_id, "duration": round(time.time() - t0, 3)})
                content, _tcs = parse_envelope(raw_text, tools=None, result_obj=None)
                safe_content = content if (content and content.strip()) else raw_text or "I encountered an empty response from the inference engine."
                self._send(
                    200,
                    {
                        "id": "chatcmpl-" + req_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "gemini-3.8-flash",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": safe_content}, "finish_reason": "stop"}],
                        "usage": usage_info,
                    },
                )
                return
            if not _SDK_AVAILABLE:
                self._send(503, {"error": {"message": "SDK lane unavailable", "type": "server_error"}})
                return
            t0 = time.time()
            try:
                raw_text, usage_info = _sdk_manager.chat_lane(conversation_id, last_user, timeout=20.0)
            except KeyError:
                self._send(404, {"error": {"message": f"unknown conversation_id: {conversation_id}. Open one first via POST /v1/conversations", "type": "invalid_request_error"}})
                return
            except Exception as ex:
                self._send(502, {"error": {"message": str(ex), "type": "upstream_error"}})
                return
            _sdk_manager._lane_last[conversation_id] = time.time()
            log_line({"ts": time.time(), "req": req_id, "lane": "convo", "conv_id": conversation_id, "duration": round(time.time() - t0, 3)})
            content, _tcs = parse_envelope(raw_text, tools=None, result_obj=None)
            safe_content = content if (content and content.strip()) else raw_text or "I encountered an empty response from the inference engine."
            self._send(
                200,
                {
                    "id": "chatcmpl-" + req_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": _sdk_manager.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": safe_content}, "finish_reason": "stop"}],
                    "usage": usage_info,
                },
            )
            return

        effective_tools = None if tool_choice == "none" else tools
        target_fn = None
        if isinstance(tool_choice, dict) and "function" in tool_choice:
            target_fn = (tool_choice.get("function") or {}).get("name")
        require_tool = bool(effective_tools) and (tool_choice == "required" or bool(target_fn))

        if body.get("stream"):
            self._handle_streaming(body, model, messages, prompt, req_id, effective_tools, tool_choice, require_tool, target_fn)
        else:
            self._handle_non_streaming(body, model, messages, prompt, req_id, effective_tools, tool_choice, require_tool, target_fn)

    # -----------------------------------------------------------------------
    # Non-Streaming Execution
    # -----------------------------------------------------------------------
    def _handle_non_streaming(
        self, body: dict, model: str, messages: list, prompt: str, req_id: str,
        tools: Optional[list], tool_choice: Optional[str], require_tool: bool = False,
        target_fn: Optional[str] = None
    ) -> None:
        t0 = time.time()
        use_sdk = is_flash_tier(model)
        raw_text = ""
        served_model = model
        usage_info = {}
        lane = "app" if (AGY_LANE == "app" and use_sdk) else "sdk"

        if AGY_LANE == "app" and use_sdk:
            # App lane: the Antigravity app's language_server (host brain).
            # Send the user's real message only — the brain keeps its own memory.
            try:
                raw_text, usage_info = _app_lane_turn(None, _last_user_text(messages) or prompt, timeout=180.0)
            except Exception as ex:
                self._send(502, {"error": {"message": f"app lane upstream error: {ex}", "type": "upstream_error"}})
                return
            served_model = "gemini-3.8-flash"

        if use_sdk and not raw_text:
            try:
                raw_text, usage_info = _pick_mgr(model).execute_chat(prompt, timeout=15.0)
            except Exception as ex:
                log_line({"ts": time.time(), "req": req_id, "event": "sdk_fallback_to_cli", "error": str(ex)})
                use_sdk = False

        if not use_sdk:
            lane = "cli"
            env_schema = build_envelope_schema(require_tool=require_tool, target_function=target_fn, tools=tools) if tools else None
            ok = _cli_sem.acquire(timeout=120)
            if not ok:
                self._send(503, {"error": {"message": "bridge saturated (concurrency queue full)", "type": "server_error"}})
                return
            effort = body.get("reasoning_effort")
            try:
                obj, raw_obj, err, served_model = run_agy_smart(prompt, model, env_schema, effort=effort)
            finally:
                _cli_sem.release()

            if err or not obj:
                msg = (err or {}).get("error", "unknown agy error")
                self._send(502, {"error": {"message": msg, "type": "upstream_error"}})
                return

            raw_text = obj.get("structured_output") or obj.get("response", "")
            usage = (obj or {}).get("usage", {})
            usage_info = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        dur = round(time.time() - t0, 3)
        log_line({
            "ts": time.time(), "req": req_id, "lane": lane, "model": served_model,
            "duration": dur, "tools": bool(tools), "prompt_chars": len(prompt),
        })

        content, tool_calls = parse_envelope(raw_text, tools=tools, result_obj=obj if not use_sdk else None)

        # Build OpenAI assistant message & finish_reason
        if tool_calls:
            formatted_calls = []
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = tc.get("name") or fn.get("name") or "?"
                args_str = extract_tool_call_args(tc, fn)
                call_id = tc.get("id") or f"call_{req_id}_{i}"
                formatted_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": args_str,
                    },
                })
            message = {
                "role": "assistant",
                "content": clean_tool_call_content(content),
                "tool_calls": formatted_calls,
            }
            finish_reason = "tool_calls"
        else:
            # Prevent empty-response errors in Hermes: content must never be empty when finish is stop
            safe_content = content if (content and content.strip()) else (
                (obj or {}).get("_accumulated_text") or (obj or {}).get("response") or "I encountered an empty response from the inference backend. Please retry your request."
            )
            message = {
                "role": "assistant",
                "content": safe_content,
            }
            finish_reason = "stop"

        resp = {
            "id": "chatcmpl-" + req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage_info,
        }
        self._send(200, resp)

    # -----------------------------------------------------------------------
    # Streaming Execution (SSE)
    # -----------------------------------------------------------------------
    def _handle_streaming(
        self, body: dict, model: str, messages: list, prompt: str, req_id: str,
        tools: Optional[list], tool_choice: Optional[str], require_tool: bool = False,
        target_fn: Optional[str] = None
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        t0 = time.time()
        effort = body.get("reasoning_effort")
        conversation_id = (body.get("conversation_id") or "").strip()
        last_user = _last_user_text(messages)
        base = {"id": "chatcmpl-" + req_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

        # Case 0: App lane (default) — run the turn on the Antigravity brain, emit SSE.
        if AGY_LANE == "app" and is_flash_tier(model):
            try:
                if conversation_id:
                    raw_text, usage_info = _app_lane_turn(conversation_id, last_user, timeout=180.0)
                else:
                    raw_text, usage_info = _app_lane_turn(None, last_user or prompt, timeout=180.0)
            except Exception as ex:
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"role": "assistant", "content": f"[app lane error] {ex}"}, "finish_reason": None,
                }])))
                self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self.wfile.write(self._sse(dict(base, choices=[{
                "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None,
            }])))
            # Content produced genuinely by the brain; emitted as one delta (sequential passthrough).
            for chunk in (raw_text[i:i + 40] for i in range(0, len(raw_text), 40)):
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"content": chunk}, "finish_reason": None,
                }])))
                self.wfile.flush()
            self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # Case 1: Pure text streaming via persistent SDK agent
        if is_flash_tier(model) and not tools:
            q = queue.Queue()
            _pick_mgr(model).stream_chat(prompt, q)
            try:
                # Wait for the first chunk to ensure SDK initialized without immediate error
                try:
                    first_kind, first_val = q.get(timeout=10.0)
                except queue.Empty:
                    first_kind, first_val = "error", "timeout waiting for stream"

                if first_kind == "error":
                    # SDK failed immediately (e.g. 429 quota); seamlessly fall back to CLI
                    log_line({"ts": time.time(), "req": req_id, "event": "stream_sdk_fail_cli_fallback", "error": str(first_val)})
                    ok = _cli_sem.acquire(timeout=60)
                    if ok:
                        try:
                            obj, _, err, served_model = run_agy_smart(prompt, model, None, effort=effort)
                            if obj and not err:
                                text_out = obj.get("response", "") or obj.get("_accumulated_text") or "I encountered an empty response from the inference backend. Please retry your request."
                                self.wfile.write(self._sse(dict(base, model=served_model, choices=[{
                                    "index": 0, "delta": {"role": "assistant", "content": text_out}, "finish_reason": None,
                                }])))
                                self.wfile.write(self._sse(dict(base, model=served_model, choices=[{
                                    "index": 0, "delta": {}, "finish_reason": "stop",
                                }])))
                                self.wfile.write(b"data: [DONE]\n\n")
                                self.wfile.flush()
                                return
                        finally:
                            _cli_sem.release()

                    # If CLI also failed, emit clean fallback text
                    self.wfile.write(self._sse(dict(base, choices=[{
                        "index": 0, "delta": {"role": "assistant", "content": "I encountered an error communicating with the inference backend. Please retry your request."}, "finish_reason": None,
                    }])))
                    self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                # Normal SDK streaming path
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None,
                }])))
                if first_kind == "text":
                    self.wfile.write(self._sse(dict(base, choices=[{
                        "index": 0, "delta": {"content": first_val}, "finish_reason": None,
                    }])))
                self.wfile.flush()

                while first_kind != "done":
                    try:
                        kind, val = q.get(timeout=60.0)
                    except queue.Empty:
                        break

                    if kind == "text":
                        self.wfile.write(self._sse(dict(base, choices=[{
                            "index": 0, "delta": {"content": val}, "finish_reason": None,
                        }])))
                        self.wfile.flush()
                    elif kind == "done":
                        break
                    elif kind == "error":
                        break

                # Single stop finish_reason
                self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            except (BrokenPipeError, OSError):
                return

        # Case 2: Tool-enabled or reasoning tier requests (requires envelope extraction)
        raw_text = ""
        served_model = model
        use_sdk = is_flash_tier(model)

        if use_sdk:
            try:
                raw_text, _ = _pick_mgr(model).execute_chat(prompt, timeout=15.0)
            except Exception as ex:
                log_line({"ts": time.time(), "req": req_id, "event": "stream_sdk_fallback_to_cli", "error": str(ex)})
                use_sdk = False

        if not use_sdk:
            env_schema = build_envelope_schema(require_tool=require_tool, target_function=target_fn, tools=tools) if tools else None
            ok = _cli_sem.acquire(timeout=120)
            if not ok:
                try:
                    self.wfile.write(self._sse({"error": {"message": "bridge saturated", "type": "server_error"}}))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, OSError):
                    pass
                return

            def _stream_ping():
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, OSError):
                    pass

            try:
                obj, raw_obj, err, served_model = run_agy_smart(
                    prompt, model, env_schema, effort=effort, on_ping=_stream_ping
                )
            finally:
                _cli_sem.release()

            if err or not obj:
                try:
                    self.wfile.write(self._sse({"error": {"message": (err or {}).get("error", "upstream error"), "type": "upstream_error"}}))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, OSError):
                    pass
                return
            raw_text = obj.get("structured_output") or obj.get("response", "")

        dur = round(time.time() - t0, 3)
        log_line({
            "ts": time.time(), "req": req_id, "lane": "cli" if not use_sdk else "sdk", "model": served_model,
            "duration": dur, "tools": bool(tools), "prompt_chars": len(prompt),
        })

        content, tool_calls = parse_envelope(raw_text, tools=tools, result_obj=obj if not use_sdk else None)
        base["model"] = served_model

        try:
            if tool_calls:
                formatted_calls = []
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    fn_name = tc.get("name") or fn.get("name") or "?"
                    args_str = extract_tool_call_args(tc, fn)
                    call_id = tc.get("id") or f"call_{req_id}_{i}"
                    formatted_calls.append({
                        "index": i,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    })

                # Emit ONE single tool_call delta chunk with finish_reason: "tool_calls"
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": formatted_calls,
                    },
                    "finish_reason": "tool_calls",
                }])))
                # STRICT REQUIREMENT: Trailing data: [DONE] IMMEDIATELY — NO extraneous stop chunk!
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                # Text output: emit chunks for smooth UI streaming
                safe_content = content if (content and content.strip()) else (
                    (obj or {}).get("_accumulated_text") or (obj or {}).get("response") or "I could not retrieve the requested data directly. Please try again."
                )
                chunk_size = 64
                for i in range(0, len(safe_content), chunk_size):
                    chunk = safe_content[i:i + chunk_size]
                    self.wfile.write(self._sse(dict(base, choices=[{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }])))
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }])))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass




# ---------------------------------------------------------------------------
# Idle lane reaper: close conversation lanes that have been idle too long so
# abandoned lanes don't pin pool slots forever.
# ---------------------------------------------------------------------------
def _lane_idle_reaper() -> None:
    ttl = float(os.environ.get("AGY_LANE_IDLE_TTL", "1800"))  # 30 min default
    while True:
        time.sleep(30)
        try:
            now = time.time()
            stale = [cid for cid, last in _sdk_manager._lane_last.items() if now - last > ttl]
            for cid in stale:
                _sdk_manager.close_lane(cid)
                log_line({"ts": time.time(), "event": "lane_idle_closed", "conv_id": cid, "idle_ttl": ttl})
        except Exception:
            pass


def main() -> None:
    os.makedirs(os.path.join(HOME, ".hermes", "agy-bridge"), exist_ok=True)
    if not os.path.exists(AGY):
        print(f"Warning: agy CLI binary not found at {AGY}", file=sys.stderr)

    # Warm up persistent SDK agent pool in background
    if _SDK_AVAILABLE:
        try:
            print(f"Initializing persistent Antigravity SDK agent pool ({FAST_FLASH_MODEL}, size 2)...", flush=True)
            _sdk_manager.start()
            _sdk_manager_pro.start()
            print("Antigravity SDK persistent agent pool ready.", flush=True)
        except Exception as e:
            print(f"Warning: SDK agent pool pre-warmup failed: {e}. Will lazily initialize.", file=sys.stderr)

    server = ThreadingHTTPServer((HOST, PORT), GatewayHandler)
    threading.Thread(target=_lane_idle_reaper, daemon=True, name="agy-lane-reaper").start()
    print(f"agy-bridge gateway listening on {HOST}:{PORT} (model {DEFAULT_MODEL})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down agy-bridge gateway...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

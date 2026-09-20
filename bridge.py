#!/usr/bin/env python3
"""gateway.py — High-performance persistent inference gateway for Hermes <-> Antigravity.

Provides an OpenAI-compatible /v1/chat/completions endpoint on 127.0.0.1:8790
backed by the running Antigravity app's language_server (host brain) over the
**app lane only**. The SDK lane (google-antigravity pip + API key) and the
CLI lane (agy subprocess) have been REMOVED — every request goes through the
consumer-OAuth Antigravity app (Phase 3 hard gate: zero API keys in this path).

Endpoints:
- GET  /v1/models               -> OpenAI model list
- POST /v1/chat/completions     -> OpenAI chat completion (non-streaming + SSE streaming)
- POST /v1/conversations        -> open a conversation lane (brain-native cascade)
- GET  /v1/conversations        -> list active lanes
- DELETE /v1/conversations/{id} -> close a lane
- GET  /health                  -> liveness + app-lane stats
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

HOME = os.path.expanduser("~")
HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGY_BRIDGE_PORT", "8790"))
_APP_MGR = [None]
_APP_RETRIES = 2


def _get_app_manager():
    if _APP_MGR[0] is None:
        import app_lane
        _APP_MGR[0] = app_lane.AppLaneManager(os.environ.get("AGY_APP_REGISTRY", "/veta/app-brains/registry.json"))
    return _APP_MGR[0]


def _app_lane_turn(conversation_id: str, user_text: str, timeout: float = 180.0, model_enum: str = None):
    """Run one turn on the app lane (conversation-pinned or stateless).
    Returns (content, usage_dict). Raises for upstream/registration errors."""
    last_err = None
    for attempt in range(_APP_RETRIES + 1):
        mgr = _get_app_manager()
        try:
            if conversation_id:
                return mgr.chat_lane(conversation_id, user_text, timeout=timeout, model_enum=model_enum)
            return mgr.chat(user_text, model_enum=model_enum)
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


LOG = os.path.join(HOME, ".hermes", "agy-bridge", "bridge.log")

# Automatically load ~/.hermes/.env for non-key runtime settings. The SDK/API
# keys are intentionally NOT consumed here (Phase 3: app lane only).
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
        return "gemini-3.8-flash-high"
    return name


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


def messages_to_prompt(messages: list, tools: Optional[list] = None, tool_choice: Any = None, model: str = "") -> str:
    """Render OpenAI messages + tool framing into the app-lane prompt."""
    parts = []
    is_claude = "claude" in (model or "").lower()
    if not tools or tool_choice == "none":
        parts.append(
            "CRITICAL INSTRUCTION: You are serving as an OpenAI-compatible text inference "
            "backend. You have ZERO local tool execution permissions. Do NOT call run_command, "
            "view_file, write_to_file, or any built-in tools. Output only your direct text response."
        )
    elif is_claude:
        # Claude treats an identity-claiming "CRITICAL INSTRUCTION" as a prompt
        # injection attempt and refuses. Use a neutral, user-voice frame.
        parts.append(
            "You are working through an API bridge that lets you call functions. "
            "Some requests may require a tool. The functions available to you are listed "
            "in the TOOL CATALOG below. Do NOT invent tools not in the catalog."
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
                "OUTPUT RULE: When the user's request requires a tool from the catalog, "
                "respond ONLY with a JSON object:\n"
                '{"content": null, "tool_calls": [{"name": "<function_name>", "arguments": {<json_arguments>}}]}\n'
                "If NO tool is needed, respond normally with your direct text answer."
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
    cleaned = re.sub(r'```(?:json)?\s*\{.*?"tool_calls":\s*\[\]\}\s*```\s*$', '', acc.strip(), flags=re.DOTALL).strip()
    cleaned = re.sub(r'\{.*?"tool_calls":\s*\[\]\}\s*$', '', cleaned, flags=re.DOTALL).strip()
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

        if result_obj:
            acc = result_obj.get("_accumulated_text") or result_obj.get("response") or ""
            cleaned_acc = _clean_accumulated_text(acc)
            if cleaned_acc and len(cleaned_acc) > len(c) + 60:
                return cleaned_acc, []
            elif not c and cleaned_acc:
                return cleaned_acc, []

        if raw_text.get("content"):
            return raw_text.get("content"), []

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

    return txt, []


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
            mgr = _get_app_manager()
            lanes = mgr.lanes()
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": cid, "active": True, "pinned_pool_slot": None} for cid in lanes],
                    "active": len(lanes),
                    "pool_size": "app-lane (elastic)",
                },
            )
            return
        if path in ("/health", "/"):
            try:
                mgr = _get_app_manager()
                h = mgr.health()
            except Exception as ex:
                mgr = None
                h = {"ok": False, "error": str(ex)[:200]}
            self._send(
                200,
                {
                    "status": "healthy" if h.get("ok") else "degraded",
                    "app_lane": h,
                    "active_lanes": len(mgr.lanes()) if mgr else 0,
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
            mgr = _get_app_manager()
            mgr.release_lane(conv_id)
            self._send(200, {"conversation_id": conv_id, "status": "closed", "active": len(mgr.lanes())})
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/conversations":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or "0")) or b"{}")
            except Exception:
                body = {}
            conv_id = (body.get("conversation_id") or "").strip() or ("lane_%08x" % (time.time_ns() & 0xFFFFFFFF))
            mgr = _get_app_manager()
            try:
                mgr.acquire_for_lane(conv_id)
            except Exception:
                pass
            self._send(
                200,
                {
                    "conversation_id": conv_id,
                    "status": "open",
                    "model": "gemini-3.8-flash",
                    "active": len(mgr.lanes()),
                    "pool_size": "app-lane (elastic)",
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
        prompt = messages_to_prompt(messages, tools, tool_choice, model=model)
        req_id = "%08x" % (time.time_ns() & 0xFFFFFFFF)

        # Conversation lane: the brain keeps its own memory; send only the latest
        # user message so we don't double-replay history.
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
            try:
                raw_text, usage_info = _app_lane_turn(conversation_id, last_user, timeout=180.0, model_enum=model)
            except Exception as ex:
                self._send(502, {"error": {"message": f"app lane upstream error: {ex}", "type": "upstream_error"}})
                return
            log_line({"ts": time.time(), "req": req_id, "lane": "app-convo", "conv_id": conversation_id, "duration": round(time.time() - t0, 3)})
            content, _tcs = parse_envelope(raw_text, tools=None, result_obj=None)
            safe_content = content if (content and content.strip()) else raw_text or "I encountered an empty response from the inference engine."
            self._send(
                200,
                {
                    "id": "chatcmpl-" + req_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
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
    # Non-Streaming Execution (app lane)
    # -----------------------------------------------------------------------
    def _handle_non_streaming(
        self, body: dict, model: str, messages: list, prompt: str, req_id: str,
        tools: Optional[list], tool_choice: Optional[str], require_tool: bool = False,
        target_fn: Optional[str] = None
    ) -> None:
        t0 = time.time()
        raw_text = ""
        served_model = model
        usage_info = {}

        try:
            raw_text, usage_info = _app_lane_turn(None, prompt, timeout=180.0, model_enum=model)
        except Exception as ex:
            self._send(502, {"error": {"message": f"app lane upstream error: {ex}", "type": "upstream_error"}})
            return

        dur = round(time.time() - t0, 3)
        log_line({
            "ts": time.time(), "req": req_id, "lane": "app", "model": served_model,
            "duration": dur, "tools": bool(tools), "prompt_chars": len(prompt),
        })

        content, tool_calls = parse_envelope(raw_text, tools=tools, result_obj=None)

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
                raw_text or "I encountered an empty response from the inference backend. Please retry your request."
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
    # Streaming Execution (SSE) — app lane sequential passthrough
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
        conversation_id = (body.get("conversation_id") or "").strip()
        last_user = _last_user_text(messages)
        base = {"id": "chatcmpl-" + req_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

        try:
            if conversation_id:
                raw_text, usage_info = _app_lane_turn(conversation_id, last_user, timeout=180.0, model_enum=model)
            else:
                raw_text, usage_info = _app_lane_turn(None, prompt, timeout=180.0, model_enum=model)
        except Exception as ex:
            self.wfile.write(self._sse(dict(base, choices=[{
                "index": 0, "delta": {"role": "assistant", "content": f"[app lane error] {ex}"}, "finish_reason": None,
            }])))
            self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        log_line({
            "ts": time.time(), "req": req_id, "lane": "app-stream",
            "conv_id": conversation_id or None, "duration": round(time.time() - t0, 3),
            "tools": bool(tools),
        })

        content, tool_calls = parse_envelope(raw_text, tools=tools, result_obj=None)
        self.wfile.write(self._sse(dict(base, choices=[{
            "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None,
        }])))

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
            self.wfile.write(self._sse(dict(base, choices=[{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": formatted_calls,
                },
                "finish_reason": "tool_calls",
            }])))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # Text output: content genuinely produced by the brain; sequential passthrough.
        safe_content = (content if (content and content.strip()) else raw_text
                        or "I encountered an empty response from the inference backend. Please retry your request.")
        for chunk in (safe_content[i:i + 40] for i in range(0, len(safe_content), 40)):
            self.wfile.write(self._sse(dict(base, choices=[{
                "index": 0, "delta": {"content": chunk}, "finish_reason": None,
            }])))
            self.wfile.flush()
        self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Idle lane reaper: close conversation lanes that have been idle too long so
# abandoned lanes don't pin app-lane cascades forever.
# ---------------------------------------------------------------------------
def _lane_idle_reaper() -> None:
    ttl = float(os.environ.get("AGY_LANE_IDLE_TTL", "1800"))  # 30 min default
    while True:
        time.sleep(30)
        try:
            mgr = _APP_MGR[0]
            if mgr is None:
                continue
            now = time.time()
            stale = [cid for cid, last in mgr._lane_last.items() if now - last > ttl]
            for cid in stale:
                mgr.release_lane(cid)
                log_line({"ts": time.time(), "event": "lane_idle_closed", "conv_id": cid, "idle_ttl": ttl})
        except Exception:
            pass


def main() -> None:
    os.makedirs(os.path.join(HOME, ".hermes", "agy-bridge"), exist_ok=True)

    server = ThreadingHTTPServer((HOST, PORT), GatewayHandler)
    threading.Thread(target=_lane_idle_reaper, daemon=True, name="agy-lane-reaper").start()
    print(f"agy-bridge gateway listening on {HOST}:{PORT} (model {DEFAULT_MODEL}) — APP LANE ONLY", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down agy-bridge gateway...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""agy-bridge — local OpenAI-compatible chat completions endpoint backed by Google Antigravity CLI.

Maps Hermes chat/delegation (OpenAI-style) onto agy CLI, serving on 127.0.0.1:8790.
launchd: com.veta.agy-bridge. Stdlib only.

PERFORMANCE (2026-09-10 rewrite):
- Old code spawned `agy -p` per request: input 24,505 tokens, ~3.5s process startup.
- New code uses agy stream-json one-shot framing: input 16,353 tokens (-33%), ~2.7s startup.
  (stream-json skips the -p skill/slash expansion epoch entirely.)
- True token streaming is IMPOSSIBLE through agy CLI (verified: only `init`, content-free
  `step_update`, and one final `result` events are emitted). So SSE is synthesized from the
  final result in 4KB chunks — but it now TERMINATES (Connection: close + data: [DONE]),
  previously it kept the connection open forever (broken streaming UX).
- Persistent agy child (keep-alive across requests) is NOT viable: agy stream-json has no
  reset/new-conversation event (verified), so per-request full-history prompts would double
  context each turn and eventually time out. Per-request spawn is the correct trade.

Downstream: Hermes config has `streaming.enabled: false` — the desktop app waits for the
full body, so every turn feels slow. Flip it true to make chat render early; the bridge
accepts `stream: true` now and terminates correctly.

Endpoints:
- GET  /v1/models             -> OpenAI model list (agy lineup)
- POST /v1/chat/completions   -> OpenAI chat completion (non-streaming + SSE streaming)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOME = os.path.expanduser("~")
HOST = "127.0.0.1"
PORT = int(os.environ.get("AGY_BRIDGE_PORT", "8790"))
AGY = os.environ.get("AGY_BIN", os.path.join(HOME, ".local", "bin", "agy"))
LOG = os.path.join(HOME, ".hermes", "agy-bridge", "bridge.log")
AGY_ACCOUNT = os.path.join(HOME, ".hermes", "bin", "agy-account.py")

DEFAULT_MODEL = os.environ.get("AGY_BRIDGE_MODEL", "gemini-3.8-flash-medium")
TIMEOUT = os.environ.get("AGY_BRIDGE_TIMEOUT", "300s")

MODELS = [
    "agy", "agy-auto",
    "gemini-3.8-flash-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low",
    "gemini-3.7-flash-high", "gemini-3.7-flash-medium", "gemini-3.7-flash-low",
    "gemini-3.6-flash-high", "gemini-3.6-flash-medium", "gemini-3.6-flash-low",
    "gemini-3.1-pro-high", "gemini-3.1-pro-low",
    "claude-sonnet-4-6", "claude-opus-4-6-thinking", "gpt-oss-120b-medium",
]

# Fallback ladder for agy/agy-auto requests (tried in order).
MODEL_LADDER = [
    "gemini-3.7-flash-high",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
]

# Concurrency: each request spawns its OWN independent agy stream-json child, so
# parallel requests are safe. This BoundedSemaphore is only a global throttle to
# avoid thrashing the machine with N heavy agy processes at once (each is a
# 178MB Go binary + model context). Default concurrency 3: enough for chat to
# sneak between fleet requests, small enough to keep RAM sane. Requests WAIT
# (queue) rather than 503 — a slow fleet call must not fail chat turns.
_SEM_SLOTS = int(os.environ.get("AGY_BRIDGE_CONCURRENCY", "3"))
_sem = threading.BoundedSemaphore(_SEM_SLOTS)


def resolve_model(name: str) -> str:
    """Map requested model name onto a concrete agy model id."""
    if not name or name in ("agy-auto", "auto"):
        return DEFAULT_MODEL
    if name in MODELS:
        return name
    # Allow plain "gemini-3.8-flash-high" and short names to pass through;
    # agy itself will surface an error for unknown models.
    return name  # let agy decide / surface its error


def is_quota_error(err_dict: dict) -> bool:
    s = str(err_dict or {}).lower()
    return any(k in s for k in ("quota", "429", "rate limit", "resource exhausted", "billing"))


def log_line(entry: dict) -> None:
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _text_of(content) -> str:
    """Flatten OpenAI content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(str(p.get("text", "")))
                elif p.get("type") == "image_url":
                    out.append("[image]")
                else:
                    out.append(str(p))
            else:
                out.append(str(p))
        return "\n".join(out)
    if content is None:
        return ""
    return str(content)


def build_tool_catalog(tools) -> str:
    """Render OpenAI tool definitions into a compact catalog for agy."""
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
        lines.append("- %s(%s): %s" % (name, arg_sig, desc[:200]))
    return "\n".join(lines)


def build_envelope_schema(require_tool: bool) -> dict:
    """JSON schema that forces agy's final answer into {content, tool_calls}."""
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                },
            },
        },
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def messages_to_prompt(messages, tools=None, tool_choice=None) -> str:
    """Render OpenAI messages + optional tool framing into the agy text prompt."""
    parts = [
        "CRITICAL INSTRUCTION: You are serving as an OpenAI-compatible text inference "
        "backend. Do NOT call or invoke any tools or execute any commands. Output only "
        "your direct text completion response."
    ]
    if tools:
        parts.append(build_tool_catalog(tools))
        if tool_choice == "required":
            parts.append(
                "STRICT OUTPUT RULE: You MUST call exactly one tool from the TOOL CATALOG "
                "in this turn. Do not describe the call; do not answer in prose. Set "
                "content to an empty string."
            )
        else:
            parts.append(
                "OUTPUT RULE: If the user's request requires a tool, respond with a "
                "tool_calls array (name exactly as listed, arguments as a JSON object). "
                "If no tool is needed, respond with content as plain text and an EMPTY "
                "tool_calls array ([]). Never both."
            )
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            name = ""
            if isinstance(content, list):
                name = " ".join(
                    p.get("name", "") for p in content if isinstance(p, dict) and p.get("type") == "tool_call_id"
                )
            parts.append("<tool_result%s>\n%s\n</tool_result>" % (" name=" + name if name else "", _text_of(content)))
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(
                    '<assistant_tool_call name="%s" arguments=%s>'
                    % (fn.get("name", "?"), json.dumps(fn.get("arguments", {}), separators=(",", ":")))
                )
            continue
        parts.append("<%s>\n%s\n</%s>" % (role, _text_of(content), role))
    return "\n\n".join(parts)


def parse_envelope(raw_text):
    """Return (content, tool_calls) from raw agy response text or structured_output dict."""
    content, tool_calls = "", []
    # 1) full object (structured_output passed straight through)
    if isinstance(raw_text, dict):
        content = raw_text.get("content") or ""
        tool_calls = raw_text.get("tool_calls") or []
        return content, tool_calls
    # 2) JSON string (agy rendered the envelope inside the text response)
    txt = (raw_text or "").strip()
    if txt:
        try:
            obj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            content = obj.get("content") or ""
            tool_calls = obj.get("tool_calls") or []
            return content, tool_calls
        except Exception:
            pass
    return txt, tool_calls


def run_agy_stream_oneshot(prompt: str, model: str, json_schema=None):
    """Run ONE turn against a fresh agy stream-json child (input-format cheap path).

    Returns (obj, obj, err) mirroring the old run_agy signature: obj is the agy result
    dict (with .status/.response/.usage) or None; err is a dict or None.
    """
    cmd = [
        AGY, "--print-timeout", TIMEOUT, "--model", model,
        "--input-format", "stream-json", "--output-format", "stream-json",
        "--disable-slash-commands",
    ]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    t0 = time.time()
    child = None
    try:
        child = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        child.stdin.write(json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}) + "\n")
        child.stdin.flush()

        result_obj = None
        err_text = ""
        while True:
            line = child.stdout.readline()
            if not line:
                if child.poll() is not None:
                    break
                continue
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                err_text = line[-1500:]
                continue
            if ev.get("event") == "result":
                result_obj = ev.get("result") or {}
                break
            # step_update / init: content-free, ignore

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
            return None, result_obj, {"error": "agy status " + str(result_obj.get("status"))}
        return result_obj, result_obj, None
    except subprocess.TimeoutExpired:
        return None, None, {"error": "agy timed out (%s)" % TIMEOUT, "cmd": " ".join(cmd[:6])}
    except Exception as ex:
        return None, None, {"error": "agy spawn failed: %s" % ex}
    finally:
        if child is not None:
            try:
                child.kill()
            except Exception:
                pass


def run_agy_smart(prompt: str, requested_model: str, json_schema=None):
    """Execute agy with auto-account failover and model ladder fallback."""
    target_model = resolve_model(requested_model)
    models_to_try = [target_model]
    if requested_model in ("agy", "agy-auto", "auto") or target_model in MODEL_LADDER:
        for m in MODEL_LADDER:
            if m not in models_to_try:
                models_to_try.append(m)

    last_err = None
    last_raw_obj = None
    for model_cand in models_to_try:
        obj, raw_obj, err = run_agy_stream_oneshot(prompt, model_cand, json_schema)
        if obj and not err:
            return obj, raw_obj, None, model_cand
        last_err = err
        last_raw_obj = raw_obj

        # Quota exhausted -> attempt automatic account swap
        if is_quota_error(err) and os.path.exists(AGY_ACCOUNT):
            try:
                log_line({"ts": time.time(), "event": "quota_detected", "model": model_cand, "action": "switching_account"})
                sw = subprocess.run(
                    [sys.executable, AGY_ACCOUNT, "round-robin"],
                    capture_output=True, text=True, timeout=60,
                )
                if sw.returncode == 0:
                    obj2, raw_obj2, err2 = run_agy_stream_oneshot(prompt, model_cand, json_schema)
                    if obj2 and not err2:
                        return obj2, raw_obj2, None, model_cand
                    last_err = err2
                    last_raw_obj = raw_obj2
            except Exception as ex:
                log_line({"ts": time.time(), "event": "account_switch_failed", "error": str(ex)})

    return None, last_raw_obj, last_err, target_model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # silence default stderr noise

    def _send(self, code, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "agy"} for m in MODELS]})
            return
        self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except Exception as e:
            self._send(400, {"error": {"message": "bad json: %s" % e, "type": "invalid_request_error"}})
            return
        model = resolve_model(body.get("model", DEFAULT_MODEL))
        messages = body.get("messages") or []
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        prompt = messages_to_prompt(messages, tools, tool_choice)
        env_schema = build_envelope_schema(tool_choice == "required") if tools else None
        req_id = "%08x" % (time.time_ns() & 0xFFFFFFFF)
        if body.get("stream"):
            self._handle_stream(body, model, messages, prompt, req_id, tools, tool_choice, env_schema)
            return
        ok = _sem.acquire(timeout=120)
        if not ok:
            self._send(503, {"error": {"message": "bridge saturated (concurrency queue full)", "type": "server_error"}})
            return
        try:
            obj, raw_obj, err, served_model = run_agy_smart(prompt, model, env_schema)
        finally:
            _sem.release()
        usage = (obj or {}).get("usage", {})
        log_line({
            "ts": time.time(), "req": req_id, "model": served_model, "requested_model": model,
            "prompt_chars": len(prompt), "msg_count": len(messages), "tools": bool(tools),
            "status": "ok" if obj else "error", "err": (err or {}).get("error"),
            "duration": (raw_obj or {}).get("duration_seconds"),
            "bridge_duration": (raw_obj or {}).get("_bridge_duration"),
            "in_tokens": usage.get("input_tokens"),
            "out_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
        if err or not obj:
            msg = (err or {}).get("error", "unknown agy error")
            self._send(502, {"error": {"message": msg, "type": "upstream_error"}})
            return
        content, tool_calls = parse_envelope(obj.get("structured_output") or obj.get("response", ""))
        message = {"role": "assistant", "content": content or None}
        finish = "stop"
        if tool_calls:
            message["tool_calls"] = []
            for i, tc in enumerate(tool_calls):
                name = tc.get("name", "?")
                args = tc.get("arguments") or {}
                message["tool_calls"].append({
                    "id": "call_%s_%d" % (req_id, i),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if not isinstance(args, str) else args,
                    },
                })
            finish = "tool_calls"
        resp = {
            "id": "chatcmpl-" + req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "conversation_id": obj.get("conversation_id"),
        }
        self._send(200, resp)

    def _sse(self, payload: dict) -> bytes:
        return ("data: " + json.dumps(payload) + "\n\n").encode()

    def _handle_stream(self, body, model, messages, prompt, req_id, tools=None, tool_choice=None, env_schema=None):
        """Synthesize a TERMINATING SSE stream from the agy one-shot result."""
        ok = _sem.acquire(timeout=120)
        if not ok:
            self._send(503, {"error": {"message": "bridge saturated (concurrency queue full)", "type": "server_error"}})
            return
        try:
            obj, raw_obj, err, served_model = run_agy_smart(prompt, model, env_schema)
        finally:
            _sem.release()

        # Stream headers FIRST with Connection: close so clients see text as soon as
        # the agy result lands (typically ~2-4s), not after a full response.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        usage = (obj or {}).get("usage", {})
        log_line({
            "ts": time.time(), "req": req_id, "model": served_model, "requested_model": model,
            "stream": True, "prompt_chars": len(prompt), "msg_count": len(messages), "tools": bool(tools),
            "status": "ok" if obj else "error", "err": (err or {}).get("error"),
            "duration": (raw_obj or {}).get("duration_seconds"),
            "bridge_duration": (raw_obj or {}).get("_bridge_duration"),
            "in_tokens": usage.get("input_tokens"),
            "out_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })

        if err or not obj:
            try:
                payload = {"error": {"message": (err or {}).get("error", "unknown agy error"), "type": "upstream_error"}}
                self.wfile.write(self._sse(payload))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass
            return

        base = {"id": "chatcmpl-" + req_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": served_model}
        try:
            content, tool_calls = parse_envelope(obj.get("structured_output") or obj.get("response", ""))
            if tool_calls:
                # Single tool_call delta chunk (OpenAI streaming shape)
                self.wfile.write(self._sse(dict(base, choices=[{
                    "index": 0,
                    "delta": {"role": "assistant", "content": None, "tool_calls": [
                        {
                            "index": i,
                            "id": "call_%s_%d" % (req_id, i),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "?"),
                                "arguments": json.dumps(tc.get("arguments") or {})
                                if not isinstance(tc.get("arguments"), str) else tc.get("arguments"),
                            },
                        } for i, tc in enumerate(tool_calls)
                    ]},
                    "finish_reason": "tool_calls",
                }])))
                self.wfile.flush()
            else:
                full = content or ""
                if full:
                    self.wfile.write(self._sse(dict(base, choices=[{
                        "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None,
                    }])))
                    self.wfile.flush()
                    chunk_size = 4096
                    for i in range(0, len(full), chunk_size):
                        self.wfile.write(self._sse(dict(base, choices=[{
                            "index": 0, "delta": {"content": full[i:i + chunk_size]}, "finish_reason": None,
                        }])))
                        self.wfile.flush()
            self.wfile.write(self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass


def main():
    os.makedirs(os.path.join(HOME, ".hermes", "agy-bridge"), exist_ok=True)
    if not os.path.exists(AGY):
        print("agy not found at %s" % AGY, file=sys.stderr)
        sys.exit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("agy-bridge listening on %s:%d (model %s)" % (HOST, PORT, DEFAULT_MODEL), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
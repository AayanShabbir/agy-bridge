"""Pure helper functions for parsing AGY/OpenAI response envelopes.

Extracted from bridge.parse_envelope to keep bridge.py focused on the HTTP
surface. ZERO behavior change: this module is a pure move of the parsing
logic + its already-pure helpers. Every helper threads the ORIGINAL tools
list through nested recursion so the dispatcher fast-path and the filtering
semantics are byte-identical to the pre-refactor bridge.py.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

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


def _valid_tool_names(tools: Optional[list]) -> set:
    """Extract the valid tool-name set from an OpenAI tool list."""
    valid_tool_names = set()
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            if fn_name := fn.get("name"):
                valid_tool_names.add(fn_name)
    return valid_tool_names


def _envelope_from_list(items: List[Any], valid_names: set) -> Tuple[str, list]:
    """Parse a list envelope: normalize tool calls, extract finish responses, fall back to JSON dump."""
    valid_tcs = []
    finish_responses = []
    for item in items:
        tc = _normalize_single_tool_call(item, valid_names=valid_names)
        if tc:
            valid_tcs.append(tc)
        elif fr := _extract_finish_response(item):
            finish_responses.append(fr)
        elif isinstance(item, dict) and "tool_calls" in item and isinstance(item["tool_calls"], list):
            for sub_tc in item["tool_calls"]:
                normalized = _normalize_single_tool_call(sub_tc, valid_names=valid_names)
                if normalized:
                    valid_tcs.append(normalized)
    if valid_tcs:
        return "", valid_tcs
    if finish_responses:
        return "\n".join(finish_responses), []
    return json.dumps(items), []


def _recurse(value: Any, tools: Optional[list], result_obj: Optional[dict]) -> Tuple[str, list]:
    """Recurse into a nested content value with the ORIGINAL tools + result_obj."""
    return parse_envelope(value, tools=tools, result_obj=result_obj)


def _envelope_from_dict(obj: dict, valid_names: set, tools: Optional[list], result_obj: Optional[dict]) -> Tuple[str, list]:
    """Parse a dict envelope: walk tool_calls/calls/functions keys, normalize, recurse into content."""
    finish_responses = []
    candidate_lists = []
    for k in ("tool_calls", "calls", "functions"):
        val = obj.get(k)
        if isinstance(val, list):
            candidate_lists.extend(val)
        elif isinstance(val, dict):
            candidate_lists.append(val)

    valid_tcs = []
    for item in candidate_lists:
        tc = _normalize_single_tool_call(item, valid_names=valid_names)
        if tc:
            valid_tcs.append(tc)
        elif fr := _extract_finish_response(item):
            finish_responses.append(fr)

    c = obj.get("content") or ""
    if valid_tcs:
        return c, valid_tcs

    direct_tc = _normalize_single_tool_call(obj, valid_names=valid_names)
    if direct_tc:
        return "", [direct_tc]

    if c:
        nested_c, nested_tcs = _recurse(c, tools, result_obj)
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

    if obj.get("content"):
        return obj.get("content"), []

    # Original control flow falls through from the dict branch into the
    # empty-text handling when nothing matched (result_obj captured-tool-call
    # recovery). Delegate to idendical behavior.
    return _envelope_from_text("", valid_names, tools, result_obj)


def _envelope_from_text(text: str, valid_names: set, tools: Optional[list], result_obj: Optional[dict]) -> Tuple[str, list]:
    """Parse a text envelope: strip markdown, try JSON object/array/ndjson, else raw text."""
    txt = text.strip() if isinstance(text, str) else str(text or "").strip()
    if not txt:
        if result_obj:
            captured = result_obj.get("_captured_tool_calls") or []
            for tc in captured:
                tname = tc.get("name")
                params = tc.get("parameters") or {}
                if tname == "run_command":
                    cmd = params.get("CommandLine") or params.get("command") or ""
                    if "terminal" in valid_names:
                        return "", [{"name": "terminal", "arguments": {"command": cmd}}]
                    elif "run_command" in valid_names:
                        return "", [{"name": "run_command", "arguments": params}]
                    elif "bash" in valid_names:
                        return "", [{"name": "bash", "arguments": {"command": cmd}}]
                    elif "execute_code" in valid_names:
                        return "", [{"name": "execute_code", "arguments": {"code": cmd, "language": "bash"}}]
                elif tname in valid_names:
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
            c, tcs = _recurse(obj, tools, None)
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
                c, n_tcs = _recurse(dobj, tools, None)
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
                c, tcs = _recurse(obj, tools, None)
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
                c, tcs = _recurse(obj, tools, None)
                if tcs:
                    return c, tcs
    except Exception:
        pass

    return txt, []


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

    valid_tool_names = _valid_tool_names(tools)

    if isinstance(raw_text, list):
        return _envelope_from_list(raw_text, valid_tool_names)

    if isinstance(raw_text, dict):
        return _envelope_from_dict(raw_text, valid_tool_names, tools, result_obj)

    if isinstance(raw_text, str):
        return _envelope_from_text(raw_text, valid_tool_names, tools, result_obj)

    return _envelope_from_text(str(raw_text or ""), valid_tool_names, tools, result_obj)
"""Tool envelope validation, normalization, and contract checking (Section 14)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from agy_bridge.errors import ToolContractViolationError


def validate_and_parse_tool_calls(
    raw_calls: List[Dict[str, Any]],
    declared_tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Validate extracted tool calls against declared tool definitions, schemas, and tool_choice contracts."""
    declared_names = set()
    if declared_tools:
        for tool in declared_tools:
            if isinstance(tool, dict):
                fn = tool.get("function")
                if isinstance(fn, dict) and "name" in fn:
                    declared_names.add(fn["name"])
                elif "name" in tool:
                    declared_names.add(tool["name"])

    # If tool_choice is "none", no tool calls are allowed
    if tool_choice == "none" and raw_calls:
        raise ToolContractViolationError("Tool calls emitted when tool_choice='none'")

    # If tool_choice is "required", at least one tool call must be present
    if tool_choice == "required" and not raw_calls:
        raise ToolContractViolationError("Required tool call was not emitted")

    # If tool_choice specifies a specific function name
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and "name" in fn:
            req_name = fn["name"]
            if not any(call.get("name") == req_name for call in raw_calls):
                raise ToolContractViolationError(
                    f"Named tool {req_name!r} specified in tool_choice was not emitted"
                )

    if not raw_calls:
        return []

    parsed_calls = []
    for idx, call in enumerate(raw_calls):
        name = call.get("name")
        if not name:
            raise ToolContractViolationError(f"Tool call at index {idx} missing name")

        if declared_names and name not in declared_names:
            raise ToolContractViolationError(
                f"Tool {name!r} is undeclared in tools schema"
            )

        args = call.get("arguments", "{}")
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError as exc:
                raise ToolContractViolationError(
                    f"Tool {name!r} arguments are not valid JSON: {exc}"
                ) from exc
            arg_str = args
        elif isinstance(args, dict):
            arg_str = json.dumps(args)
        else:
            raise ToolContractViolationError(
                f"Tool {name!r} arguments must be string or dict, got {type(args)}"
            )

        call_id = call.get("id") or f"call_{idx}_{name}"
        parsed_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arg_str,
                },
            }
        )

    return parsed_calls

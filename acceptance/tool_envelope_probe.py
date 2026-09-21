#!/usr/bin/env python3
"""Tool envelope probe for AGY bridge & language server.
Checks:
1. Sends a chat completion with an OpenAI tool definition and tool_choice='required'.
2. Verifies the response returns a valid tool_calls structure or correctly formatted tool call.
3. Tests tool result return in conversation history.
"""
import sys
import os
import json
import urllib.request
import urllib.error

BRIDGE_URL = os.environ.get("AGY_BRIDGE_URL", "http://127.0.0.1:8790")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Boston, MA"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def test_tool_envelope():
    print("=== [1/2] Testing Tool Call Generation (tool_choice='required') ===")
    payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [
            {"role": "user", "content": "Check the weather in Boston"}
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        "max_tokens": 100
    }
    try:
        req = urllib.request.Request(
            f"{BRIDGE_URL}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls")
            print(f"Response message: {json.dumps(msg, indent=2)}")
            if tool_calls and len(tool_calls) > 0:
                fn = tool_calls[0].get("function", {})
                print(f"PASS: Structured tool_calls detected: {fn.get('name')} args={fn.get('arguments')}")
            else:
                # In case model returned envelope text in content
                content = msg.get("content") or ""
                if "lookup_weather" in content:
                    print(f"PASS (envelope in content): Found tool call pattern in content: {content[:100]}")
                else:
                    print(f"FAIL: Neither tool_calls nor tool envelope found in response: {msg}")
                    return False
    except Exception as e:
        print(f"FAIL: Tool call generation request failed: {e}")
        return False

    print("\n=== [2/2] Testing Multi-turn Tool Return Round-Trip ===")
    conversation = [
        {"role": "user", "content": "Check the weather in Boston"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_test123",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": json.dumps({"city": "Boston"})
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_test123",
            "name": "lookup_weather",
            "content": json.dumps({"temperature": "68F", "condition": "Sunny"})
        }
    ]
    payload2 = {
        "model": "gemini-3.8-flash-high",
        "messages": conversation,
        "max_tokens": 50
    }
    try:
        req = urllib.request.Request(
            f"{BRIDGE_URL}/v1/chat/completions",
            data=json.dumps(payload2).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content", "")
            print(f"PASS: Post-tool assistant reply: {content.strip()[:100]!r}")
    except Exception as e:
        print(f"FAIL: Post-tool multi-turn turn failed: {e}")
        return False

    print("\nTOOL ENVELOPE PROBE PASSED")
    return True

if __name__ == "__main__":
    ok = test_tool_envelope()
    sys.exit(0 if ok else 1)

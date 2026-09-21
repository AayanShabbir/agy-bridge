#!/usr/bin/env python3
"""Auth probe for AGY bridge & language server.
Checks:
1. Health endpoint returns 200 and healthy status.
2. Bridge can query language server without auth rejection (401/403).
3. Local OAuth token / credential artifacts exist and are readable.
"""
import sys
import os
import json
import urllib.request
import urllib.error

BRIDGE_URL = os.environ.get("AGY_BRIDGE_URL", "http://127.0.0.1:8790")

def test_auth():
    print("=== [1/3] Probing Bridge Health Endpoint ===")
    try:
        req = urllib.request.Request(f"{BRIDGE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"PASS: /health returned {resp.status}: {data}")
    except Exception as e:
        print(f"FAIL: /health check failed: {e}")
        return False

    print("\n=== [2/3] Probing App Lane Consumer-OAuth Gate ===")
    try:
        # A simple completion tests whether the LS is authenticated with Google OAuth
        payload = json.dumps({
            "model": os.environ.get("AGY_PROBE_MODEL", "gemini-3.8-flash-high"),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{BRIDGE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            print(f"PASS: Upstream authenticated response received: {content!r}")
    except Exception as e:
        print(f"FAIL: Authenticated chat completion failed: {e}")
        return False

    print("\n=== [3/3] Inspecting Host Auth Token Artifacts ===")
    home = os.path.expanduser("~")
    token_files = [
        os.path.join(home, ".gemini", "jetski-standalone-oauth-token"),
        os.path.join(home, ".gemini", "gemini-credentials.json"),
        os.path.join(home, ".gemini", "antigravity", "antigravity_state.pbtxt"),
    ]
    found = 0
    for tf in token_files:
        if os.path.exists(tf):
            sz = os.path.getsize(tf)
            print(f"PASS: Found token/state file: {tf} ({sz} bytes)")
            found += 1
        else:
            print(f"WARN: Token file not found: {tf}")
    
    if found > 0:
        print(f"\nAUTH PROBE PASSED ({found} token/state artifacts verified)")
        return True
    else:
        print("\nFAIL: No host authentication files detected")
        return False

if __name__ == "__main__":
    ok = test_auth()
    sys.exit(0 if ok else 1)

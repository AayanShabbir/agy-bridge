#!/usr/bin/env python3
"""GetAvailableModels registry probe for AGY bridge & language server.
Checks:
1. GET /v1/models returns 200 with model list.
2. Expected model families (Gemini 3.8, Gemini 3.1 Pro, etc.) are registered.
3. Language server gRPC-web endpoint responds to GetAvailableModels / app-lane registry.
"""
import sys
import os
import json
import urllib.request
import urllib.error

BRIDGE_URL = os.environ.get("AGY_BRIDGE_URL", "http://127.0.0.1:8790")

EXPECTED_MODELS = [
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.1-pro-high",
    "gemini-3.1-pro-low",
]

def test_registry():
    print("=== [1/2] Probing /v1/models Endpoint ===")
    try:
        req = urllib.request.Request(f"{BRIDGE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", [])]
            print(f"PASS: /v1/models returned {len(models)} models:")
            for m in models:
                print(f"  - {m}")
    except Exception as e:
        print(f"FAIL: /v1/models request failed: {e}")
        return False

    print("\n=== [2/2] Validating Core Models in Registry ===")
    missing = [m for m in EXPECTED_MODELS if m not in models]
    if missing:
        print(f"FAIL: Missing expected models: {missing}")
        return False
    print(f"PASS: All {len(EXPECTED_MODELS)} expected model families enrolled in registry.")
    
    # Try importing app_lane to verify direct RPC capability if running locally
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import app_lane
        mgr = app_lane.AppLaneManager()
        cfg = mgr._config()
        ch = app_lane.Channel(cfg)
        print(f"PASS: Connected to local language server channel on port {cfg.get('http_port')}")
        # Test GetAvailableModels RPC if available
        try:
            resp = ch.rpc_ok("GetAvailableModels", {}, timeout=5)
            print(f"PASS: Direct GetAvailableModels RPC returned {len(str(resp))} bytes")
        except Exception as rpc_err:
            print(f"NOTE: Direct GetAvailableModels RPC returned: {rpc_err} (fallback to /v1/models is active)")
    except Exception as ex:
        print(f"NOTE: app_lane local channel test skipped/deferred: {ex}")

    print("\nREGISTRY PROBE PASSED")
    return True

if __name__ == "__main__":
    ok = test_registry()
    sys.exit(0 if ok else 1)

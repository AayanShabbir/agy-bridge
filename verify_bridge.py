#!/usr/bin/env python3
"""Verify the AGY bridge rewrite (bridge.py) on a spare port or the live port.

Usage:
  AGY_BRIDGE_PORT=8791 python3 verify_bridge.py      # test server must be running
  python3 verify_bridge.py --live                     # use 127.0.0.1:8790

Checks:
  1. GET /v1/models returns list with gemini-3.8-flash-low
  2. Non-stream chat completion returns 200 + content
  3. SSE stream returns 200, text/event-stream, TERMINATES with data: [DONE]
  4. SSE terminates within hard timeout (default 45s)
  5. Non-stream terminates within hard timeout
Prints PASS/FAIL per check; exit 0 only if all pass.
"""
import argparse
import json
import sys
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--live", action="store_true", help="test the live :8790 port")
p.add_argument("--port", type=int, default=None)
p.add_argument("--model", default="gemini-3.8-flash-low")
p.add_argument("--timeout", type=float, default=60)
args = p.parse_args()

port = args.port or (8790 if args.live else 8791)
base = "http://127.0.0.1:%d" % port
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok:
        fails += 1


# 1. models
try:
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        data = json.loads(r.read().decode())
    ids = [m["id"] for m in data.get("data", [])]
    ok = "gemini-3.8-flash-low" in ids and "gemini-3.8-flash-high" in ids
    check("models", ok, "(%d ids)" % len(ids))
except Exception as e:
    check("models", False, str(e))

payload = {"model": args.model, "messages": [{"role": "user", "content": "Reply with exactly: VETACHECK"}], "max_tokens": 30}

# 2. non-stream
try:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        data = json.loads(r.read().decode())
    dt = time.time() - t0
    content = data["choices"][0]["message"].get("content", "")
    ok = "VETACHECK" in content.upper() or "VETA" in content.upper() or content.strip().lower().startswith(("hi", "hello", "vetach"))
    check("nonstream", ok, "(%.1fs: %r)" % (dt, content[:60]))
except Exception as e:
    check("nonstream", False, str(e))

# 3/4. stream + termination
try:
    payload["stream"] = True
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        data = r.read().decode()
    dt = time.time() - t0
    ok = "text/event-stream" in ctype
    check("stream-ctype", ok, "(%s)" % ctype)
    ok = data.rstrip().endswith("data: [DONE]")
    check("stream-terminates", ok, "(%.1fs, %d bytes)" % (dt, len(data)))
    check("stream-fast", dt < 45, "(%.1fs < 45s)" % dt)
except Exception as e:
    check("stream", False, str(e))

sys.exit(1 if fails else 0)
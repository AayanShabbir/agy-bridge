#!/usr/bin/env python3
"""Benchmark the AGY bridge :8790 — non-stream and stream (SSE first-delay).

Usage:
  python3 bench.py [--n 5] [--model gemini-3.8-flash-high] [--prompt "Say hi"]
Prints median + p95 of: total time, TTFB (time to first byte), and for
for streaming: time-to-first-delta + total.
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request

BASE = os.environ.get("AGY_BRIDGE_PORT") and "http://127.0.0.1:%s/v1/chat/completions" % os.environ["AGY_BRIDGE_PORT"] or "http://127.0.0.1:8790/v1/chat/completions"


def post(payload: dict, stream: bool):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=120)
        ttfb = time.time() - t0
        if stream:
            first_delta = None
            chunks = 0
            for raw in r:
                chunks += 1
                if first_delta is None and b'"delta"' in raw:
                    first_delta = time.time() - t0
            total = time.time() - t0
            return ttfb, total, first_delta, chunks
        data = r.read()
        total = time.time() - t0
        return ttfb, total, None, len(data)
    except Exception as e:
        return None, None, None, f"ERR {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="gemini-3.8-flash-low")
    ap.add_argument("--prompt", default="Say hi")
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 60,
    }
    for stream in (False, True):
        print(f"=== stream={stream} model={args.model} n={args.n} ===")
        ttfb_list, total_list, fd_list, chunk_list = [], [], [], []
        errs = []
        for _ in range(args.n):
            ttfb, total, fd, ch = post({**payload, "stream": stream}, stream)
            if isinstance(ch, str) or ttfb is None:
                errs.append(ch)
                continue
            ttfb_list.append(ttfb)
            total_list.append(total)
            if fd is not None:
                fd_list.append(fd)
                chunk_list.append(ch)
        print(f"  ttfb:  median={statistics.median(ttfb_list):.2f}s p95={sorted(ttfb_list)[-1]:.2f}s" if ttfb_list else "  ttfb: no data")
        print(f"  total: median={statistics.median(total_list):.2f}s p95={sorted(total_list)[-1]:.2f}s" if total_list else "  total: no data")
        if fd_list:
            print(f"  first-delta: median={statistics.median(fd_list):.2f}s | chunks={chunk_list}")
        if errs:
            print(f"  ERRORS: {errs[:3]}")


if __name__ == "__main__":
    main()
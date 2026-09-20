#!/usr/bin/env python3
"""App lane for agy-bridge: drives the Antigravity language_server (host) via its
gRPC-web+JSON API. Zero external deps (stdlib only) so it runs inside the Docker
container unchanged.

Flow per turn:
    StartCascade({source: CASCADE_CLIENT, prompt})          -> cascadeId
    SendUserCascadeMessage({cascadeId, items:[{text}]})     -> kicks the model loop
    StreamAgentStateUpdates({conversationId, subscriberId, initialStepsPageBounds,
                             trajectoryVerbosity: FULL})    -> frames until fullyIdle
    answer = last steps[N].plannerResponse.response

Conversation lanes: one cascadeId per conversation_id (brain-native memory).
"""
from __future__ import annotations

import json
import os
import time
import uuid
import queue
import threading
import http.client
import urllib.request
import urllib.error

SERVICE = "exa.language_server_pb.LanguageServerService"
DEFAULT_REGISTRY = os.environ.get("AGY_APP_REGISTRY", "/veta/app-brains/registry.json")
SOURCE_CASCADE_CLIENT = "CORTEX_TRAJECTORY_SOURCE_CASCADE_CLIENT"
VERBOSITY_FULL = "CLIENT_TRAJECTORY_VERBOSITY_FULL"
PING_PROMPT = "Reply with exactly: PING_OK"
DEFAULT_MODEL_ENUM = os.environ.get("AGY_APP_MODEL_ENUM", "MODEL_PLACEHOLDER_M319")  # Gemini 3.8 Flash (Medium)

MODEL_MAP = {
    "gemini-3.8-flash-low": "MODEL_PLACEHOLDER_M318",
    "gemini-3.8-flash-medium": "MODEL_PLACEHOLDER_M319",
    "gemini-3.8-flash-high": "MODEL_PLACEHOLDER_M320",
    "gemini-3.8-flash": "MODEL_PLACEHOLDER_M319",
    "gemini-3.7-flash-low": "MODEL_PLACEHOLDER_M310",
    "gemini-3.7-flash-medium": "MODEL_PLACEHOLDER_M311",
    "gemini-3.7-flash-high": "MODEL_PLACEHOLDER_M312",
    "gemini-3.7-flash": "MODEL_PLACEHOLDER_M311",
    "gemini-3.1-pro-low": "MODEL_PLACEHOLDER_M307",
    "gemini-3.1-pro-high": "MODEL_PLACEHOLDER_M309",
    "claude-sonnet-4-6": "MODEL_PLACEHOLDER_M35",
    "gpt-oss-120b-medium": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
}

def resolve_model_enum(m: str | None) -> str:
    if not m:
        return DEFAULT_MODEL_ENUM
    if m.startswith("MODEL_"):
        return m
    return MODEL_MAP.get(m.lower().strip(), DEFAULT_MODEL_ENUM)


def _cascade_config(model_enum=None):
    """Minimal cascadeConfig the executor needs: a valid planner/requested model."""
    m = resolve_model_enum(model_enum)
    return {
        "plannerConfig": {
            "planModel": m,
            "requestedModel": {"model": m},
        }
    }


class AppLaneError(Exception):
    pass


def _frame(msg: dict) -> bytes:
    data = json.dumps(msg).encode()
    return b"\x00" + len(data).to_bytes(4, "big") + data


def _parse_frames(raw: bytes):
    out = []
    pos = 0
    while pos + 5 <= len(raw):
        flag = raw[pos]
        ln = int.from_bytes(raw[pos + 1:pos + 5], "big")
        if pos + 5 + ln > len(raw):
            break
        payload = raw[pos + 5:pos + 5 + ln]
        if flag >= 128:
            out.append(("trailer", payload.decode(errors="replace")))
        else:
            try:
                out.append(("data", json.loads(payload)))
            except Exception:
                out.append(("data_raw", payload.decode(errors="replace")))
        pos += 5 + ln
    return out


def load_registry(path=None):
    path = path or DEFAULT_REGISTRY
    try:
        with open(path) as f:
            reg = json.load(f)
    except FileNotFoundError:
        raise AppLaneError(f"app lane: registry not found at {path} — run lane_radar.py on the host")
    lanes = reg.get("lanes") or {}
    main = lanes.get(reg.get("active") or "main") or next(iter(lanes.values()), None)
    if not main:
        raise AppLaneError(f"app lane: no lane entries in registry {path}")
    return main


# ---------------------------------------------------------------------------
# Phase 4 failover: multi-lane registry + exhaustion-triggered one-at-a-time flip
# ---------------------------------------------------------------------------
EXHAUSTION_TOKENS = (
    "quota", "429", "402", "403", "402s", "exhausted", "not logged in",
    "not logged into", "billing", "daily limit", "rate limit",
    "resource exhausted", "high demand", "503", "temporarily unavailable",
    "payment required", "sign in", "authentication required", "cloudcode",
)


def is_exhaustion_error(ex) -> bool:
    """True when a lane error is an exhaustion signal (quota / not-logged-in /
    payment / 403-429) that should trigger failover to the other account."""
    s = (str(ex) or "").lower()
    return any(t in s for t in EXHAUSTION_TOKENS)


def load_registry_lanes(path=None):
    """Return all complete lane entries keyed by name + the active lane name.
    A lane is complete when it has a live http_port and csrf (a real app door)."""
    path = path or DEFAULT_REGISTRY
    try:
        with open(path) as f:
            reg = json.load(f)
    except FileNotFoundError:
        raise AppLaneError(f"app lane: registry not found at {path} — run lane_radar.py on the host")
    raw = reg.get("lanes") or {}
    lanes = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        port = int(entry.get("http_port") or 0)
        csrf = entry.get("csrf") or ""
        if port and csrf:
            lanes[key] = entry
    active = reg.get("active") or "main"
    if active not in lanes:
        active = next(iter(lanes.keys()), None)
    if not lanes:
        raise AppLaneError(f"app lane: no complete lane entries in registry {path}")
    return lanes, active


def resolve_config():
    """Registry first; env overrides for tests (AGY_APP_HTTP_PORT, AGY_APP_CSRF)."""
    port = os.environ.get("AGY_APP_HTTP_PORT") or ""
    csrf = os.environ.get("AGY_APP_CSRF") or ""
    if port and csrf:
        return {"http_port": int(port), "csrf": csrf, "source": "env"}
    main = load_registry()
    cfg = {
        "http_port": int(main.get("http_port") or 0) or int(os.environ.get("AGY_APP_HTTP_PORT", "0")),
        "csrf": main.get("csrf") or os.environ.get("AGY_APP_CSRF", ""),
        "model": main.get("model", "gemini-3.8-flash"),
        "source": "registry",
        "registry": {k: main.get(k) for k in ("app_pid", "ls_pid", "grpc_port", "updated_at")},
    }
    if not cfg["http_port"] or not cfg["csrf"]:
        raise AppLaneError("app lane: registry missing http_port/csrf — re-run lane_radar.py")
    return cfg


class Channel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.host = os.environ.get("AGY_APP_HOST_OVERRIDE", "127.0.0.1")  # containers: host.docker.internal
        self.port = cfg["http_port"]
        # The brain starts cascade executions only for loopback-origin clients.
        # When reached from a container (host.docker.internal), spoof a loopback Host.
        self.spoof_host = None if self.host == "127.0.0.1" else f"127.0.0.1:{self.port}"

    def _headers(self, extra=None):
        h = {
            "content-type": "application/grpc-web+json",
            "x-grpc-web": "1",
            "x-codeium-csrf-token": self.cfg["csrf"],
        }
        if extra:
            h.update(extra)
        if self.spoof_host:
            h["Host"] = self.spoof_host
        return h

    def rpc(self, method, msg, timeout=30):
        body = _frame(msg)
        c = http.client.HTTPConnection(f"{self.host}:{self.port}", timeout=timeout + 5)
        c.request("POST", f"/{SERVICE}/{method}", body=body, headers=self._headers())
        resp = c.getresponse()
        raw = resp.read()
        frames = _parse_frames(raw)
        gh = resp.getheader("Grpc-Message")
        if resp.status != 200:
            raise AppLaneError(f"app lane: HTTP {resp.status} from {method}: {gh or raw[:120]!r}")
        return frames

    def rpc_ok(self, method, msg, timeout=30):
        frames, hdrs = self.rpc(method, msg, timeout), {}
        for kind, val in frames:
            if kind == "trailer" and "grpc-status: 0" not in val:
                m = [l for l in val.split("\r\n") if l.startswith("grpc-message:")]
                raise AppLaneError(m[0][len("grpc-message: "):] if m else val.strip())
        datas = [v for k, v in frames if k == "data"]
        return datas[0] if datas else {}

    def stream(self, method, msg, on_frame, timeout=60):
        secs = int(timeout)
        c = http.client.HTTPConnection(f"{self.host}:{self.port}", timeout=secs + 5)
        c.request("POST", f"/{SERVICE}/{method}", body=_frame(msg), headers=self._headers({
            "grpc-timeout": f"{secs * 1000}m",
        }))
        r = c.getresponse()
        gs = r.getheader("Grpc-Status")
        gm = r.getheader("Grpc-Message")
        if gs is not None and gs != "0":
            print(f"[app_lane] stream {method} FAILED IMMEDIATELY: grpc-status={gs} grpc-message={gm}", flush=True)
            raise AppLaneError(f"app lane stream {method} failed: status={gs} message={gm}")
        if os.environ.get("AGY_APP_TRACE"):
            print(f"[trace-s] stream {method} status={r.status} grpc-status={gs} ctype={r.getheader('Content-Type')}", flush=True)
        buf = b""
        end = time.time() + secs
        while time.time() < end:
            try:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1024)
            except Exception as e:
                if os.environ.get("AGY_APP_TRACE"):
                    print(f"[app_lane] read exception: {type(e).__name__}: {e}", flush=True)
                break
            if not chunk:
                if os.environ.get("AGY_APP_TRACE"):
                    print("[app_lane] chunk is empty / stream closed by server", flush=True)
                break
            if os.environ.get("AGY_APP_TRACE"):
                print(f"[trace-s] chunk {len(chunk)}", flush=True)
            buf += chunk
            pos = 0
            while pos + 5 <= len(buf):
                flag = buf[pos]
                ln = int.from_bytes(buf[pos + 1:pos + 5], "big")
                if pos + 5 + ln > len(buf):
                    break
                payload = buf[pos + 5:pos + 5 + ln]
                if not (flag >= 128):
                    try:
                        data = json.loads(payload)
                    except Exception:
                        data = None
                    stop = on_frame(data)
                    if stop:
                        return True
                pos += 5 + ln
            buf = buf[pos:]
        return False


class CascadeClient:
    """One brain conversation (cascadeId) with memory."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.ch = Channel(cfg)
        self.cascade_id = None

    def ensure_cascade(self, prompt):
        if self.cascade_id:
            return self.cascade_id
        resp = self.ch.rpc_ok("StartCascade", {
            "source": SOURCE_CASCADE_CLIENT, "prompt": prompt}, timeout=40)
        cid = resp.get("cascadeId")
        if not cid:
            raise AppLaneError(f"app lane: StartCascade returned no cascadeId: {resp}")
        self.cascade_id = cid
        return cid

    @staticmethod
    def _planner_texts(update):
        out = []
        try:
            steps = update["update"]["mainTrajectoryUpdate"]["stepsUpdate"]["steps"]
        except Exception:
            return out
        for s in steps:
            pr = s.get("plannerResponse") or {}
            t = pr.get("modifiedResponse") or pr.get("response")
            if t and t.strip():
                out.append((s.get("stepIndex", s.get("idx", 0)), t.strip()))
        return out

    def turn(self, text, task_note="", timeout=240, model_enum=None, delete_after=False):
        cid = self.ensure_cascade(text)
        subscriber = f"agy-bridge-{uuid.uuid4().hex[:12]}"
        answers = []
        frame_q = queue.Queue()

        def _stream_worker():
            def on_frame(data):
                frame_q.put(data)
                return False  # read everything; the main loop decides when to stop
            try:
                self.ch.stream("StreamAgentStateUpdates", {
                    "conversationId": cid,
                    "subscriberId": subscriber,
                    "initialStepsPageBounds": {"startIndex": -50},
                    "trajectoryVerbosity": VERBOSITY_FULL,
                }, on_frame, timeout=max(10.0, min(timeout, 120.0)))
            except Exception:
                pass
            finally:
                frame_q.put(None)   # stream ended

        # Rule 1: SUBSCRIBE FIRST (executor pushes updates only to attached
        # subscribers), THEN send the message — exactly like the app's own UI.
        th = threading.Thread(target=_stream_worker, daemon=True)
        th.start()
        attach_t = time.time()
        while time.time() - attach_t < 8.0:
            try:
                first = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                break
            frame_q.put(first)
            break
        try:
            self.ch.rpc_ok("SendUserCascadeMessage", {
                "cascadeId": cid,
                "items": [{"text": text}],
                "cascadeConfig": _cascade_config(model_enum),
            }, timeout=40)
        except AppLaneError:
            self._stop_invocation(cid, force=True)
            raise

        # Rule 2: the PRE-RUN snapshot frame (frame 0) carries fullyIdle:true and
        # must NEVER end the loop. On conversation turns it ALSO replays the prior
        # turn's answer (e.g. steps=[…, "ONE"]) — so answers must be ignored until
        # this turn's stream transitions to RUNNING (saw_running), otherwise a stale
        # replayed answer + the snapshot's fullyIdle ends the turn instantly.
        deadline = time.time() + min(max(timeout, 10.0), 60.0)
        last_ans_at = [0.0]
        saw_running = False
        while time.time() < deadline:
            try:
                data = frame_q.get(timeout=0.2)
            except queue.Empty:
                if answers and saw_running and (time.time() - last_ans_at[0] > 1.5):
                    break
                continue
            if data is None:
                break
            u = data.get("update") or {}
            if u.get("status") == "CASCADE_RUN_STATUS_RUNNING":
                saw_running = True
            if not saw_running:
                continue  # skip pre-run snapshot / replayed prior-turn history
            new_ans = self._planner_texts(data)
            if new_ans:
                answers.extend(new_ans)
                last_ans_at[0] = time.time()
            if answers and (u.get("fullyIdle") is True or u.get("status") == "CASCADE_RUN_STATUS_IDLE"):
                break
            if answers and time.time() - last_ans_at[0] > 2.0:
                break
        th.join(timeout=2)

        if not answers:
            try:
                saved = [t for i, t in self._pull_persisted_texts(cid) if t != text]
                if saved:
                    answers.extend((i, t) for i, t in enumerate(saved))
            except Exception:
                pass
        if not answers:
            self._stop_invocation(cid, force=True)
            raise AppLaneError("app lane: no planner response text")

        self._stop_invocation(cid, force=False)
        if delete_after:
            # Stateless one-shot: free the cascade in the LS so turns don't
            # pile up (each stateless turn otherwise leaves a cascade + DB).
            try:
                self.ch.rpc_ok("DeleteCascadeTrajectory", {"conversationId": cid}, timeout=10)
            except Exception:
                pass
        answers.sort(key=lambda t: t[0])
        if not answers:
            raise AppLaneError("app lane: no planner response text")
        # Prefer a real answer over executor noise: drop echoes of the prompt,
        # markup fragments, and single-char/symbol junk the planner can emit.
        clean = [t for _, t in answers
                 if t.strip() and t.strip() != text and len(t.strip()) >= 2
                 and not t.strip().startswith(("<", ";", "=", "{", ">", "[", "\"", "'", "`", ")"))]
        selected = (clean or [answers[-1][1]])[-1]
        return selected, cid

    def _stop_invocation(self, cid, force=False):
        tries = ("ForceStopCascadeTree", "CancelCascadeInvocation") if force else ("CancelCascadeInvocation", "ForceStopCascadeTree")
        for method in tries:
            try:
                self.ch.rpc_ok(method, {"cascadeId": cid}, timeout=10)
            except Exception:
                pass

    def _pull_persisted_texts(self, cascade_id):
        """Backup answer source: the trajectory's own persisted conversation DB (host path)."""
        cands = [
            os.path.expanduser(f"~/.gemini/antigravity/conversations/{cascade_id}.db"),
        ]
        for path in cands:
            if not os.path.exists(path):
                continue
            try:
                import sqlite3
                db = sqlite3.connect(path)
                for (i,) in db.execute("SELECT idx FROM steps ORDER BY idx"):
                    row = db.execute("SELECT step_payload FROM steps WHERE idx=?", (i,)).fetchone()
                    if not row:
                        continue
                    s = row[0].decode("utf-8", errors="replace")
                    for m in __import__("re").findall(r"[\x20-\x7e]{4,}", s):
                        if m.strip() and m.strip() != "Reply with exactly: PING_OK" and not m.startswith(("(Agent", "failed", "Wraps", "--", "| ", "  -", "google3", "third_party")):
                            yield i, m.strip()
            except Exception:
                continue


class AppLaneManager:
    """Stateless turns + pinned conversation lanes, mirroring PersistentAgentPool API.

    Phase 4 failover: reads ALL complete lanes from the registry (each = one app
    door / one OAuth account). `active` = the lane used for NEW turns. When the
    active lane raises an exhaustion signal (quota / not-logged-in / 402-429-403),
    it is marked EXHAUSTED for AGY_FAILOVER_COOLDOWN seconds, active flips to the
    next available account (one at a time), and the turn retries there. An
    in-flight cascade always finishes on its own lane; the flip only affects new
    turns. Conversation lanes pin to the account they started on.
    """
    def __init__(self, registry=None, account_labels=None):
        self._reg_path = registry or DEFAULT_REGISTRY
        self._cfg = {}
        self._lanes = {}          # conversation_id -> CascadeClient
        self._lane_last = {}
        self._conv_lane = {}      # conversation_id -> registry lane key (pin)
        self._client_lock = __import__("threading").Lock()
        self._stateless = None
        self.model_name = "gemini-3.8-flash"
        # failover state
        self._exhausted_until = {}   # lane key -> epoch when it re-arms
        self._active = None
        self._refresh()

    def _refresh(self):
        """Re-read the registry for lane set/active. Cheap; called on config load."""
        lanes, active = load_registry_lanes(self._reg_path)
        for k, entry in lanes.items():
            if k not in self._cfg:
                self._cfg[k] = {
                    "key": k,
                    "http_port": int(entry.get("http_port") or 0),
                    "csrf": entry.get("csrf") or "",
                    "model": entry.get("model", "gemini-3.8-flash"),
                    "source": "registry",
                    "registry": {x: entry.get(x) for x in ("app_pid", "ls_pid", "grpc_port", "updated_at")},
                }
                self._exhausted_until.setdefault(k, 0.0)
        # drop lanes that vanished
        for k in list(self._cfg.keys()):
            if k not in lanes:
                self._cfg.pop(k, None)
                self._exhausted_until.pop(k, None)
        if active and active in self._cfg:
            self._active = active
        elif self._cfg:
            self._active = next(iter(self._cfg.keys()))

    def _config(self, lane_key=None):
        self._refresh()
        lane = lane_key or self._active
        if lane and lane in self._cfg:
            self.model_name = self._cfg[lane].get("model", self.model_name)
            return self._cfg[lane]
        # fallback: single-lane legacy path
        cfg = resolve_config()
        return cfg

    def _new_client(self, lane_key=None):
        return CascadeClient(self._config(lane_key))

    def _available_lanes(self):
        """Ordered lane keys usable for NEW turns: active first, then others."""
        now = time.time()
        lanes = list(self._cfg.keys())
        if not lanes:
            return []
        ordered = [self._active] if self._active and self._active in lanes else []
        ordered += [k for k in lanes if k not in ordered]
        return [k for k in ordered if self._exhausted_until.get(k, 0.0) <= now]

    def _mark_exhausted(self, lane_key, err):
        cd = float(os.environ.get("AGY_FAILOVER_COOLDOWN", "600"))  # default 10 min
        self._exhausted_until[lane_key] = time.time() + cd
        self._last_exhaustion = {"lane": lane_key, "error": str(err)[:200], "at": time.time()}
        # only flip active now; never mid-turn (new turns pick the new active).
        if lane_key == self._active:
            rest = [k for k in self._cfg.keys() if k != lane_key
                    and self._exhausted_until.get(k, 0.0) <= time.time()]
            if rest:
                self._active = rest[0]

    def chat(self, prompt, model_enum=None):
        """Stateless turn: try active, then other non-exhausted lanes (failover)."""
        last_err = None
        for lane in self._available_lanes():
            try:
                client = CascadeClient(self._config(lane))
                text, _ = client.turn(prompt, model_enum=model_enum, delete_after=True)
                self._active = lane
                return text, {}
            except AppLaneError as ex:
                last_err = ex
                if is_exhaustion_error(ex):
                    self._mark_exhausted(lane, ex)
                    continue
                raise
        if last_err:
            raise AppLaneError(f"all app lanes exhausted: {last_err}")

    def lanes(self):
        return list(self._lanes.keys())

    def chat_lane(self, conversation_id, user_text, timeout=240.0, model_enum=None):
        with self._client_lock:
            client = self._lanes.get(conversation_id)
            if conversation_id not in self._lanes:
                # pin to the ACCOUNT active when the lane was opened (no cross-account bleed)
                lane_key = self._available_lanes()[0] if self._available_lanes() else (self._active or "main")
                client = CascadeClient(self._config(lane_key))
                client._lane_key = lane_key  # type: ignore[attr-defined]
                self._lanes[conversation_id] = client
                self._conv_lane[conversation_id] = lane_key
                self._lane_last[conversation_id] = time.time()
        text, _ = client.turn(user_text, timeout=timeout, model_enum=model_enum)
        return text, {}

    def acquire_for_lane(self, conversation_id):
        with self._client_lock:
            if conversation_id in self._lanes:
                return True
            lane_key = self._available_lanes()[0] if self._available_lanes() else (self._active or "main")
            client = CascadeClient(self._config(lane_key))
            client._lane_key = lane_key  # type: ignore[attr-defined]
            self._lanes[conversation_id] = client
            self._conv_lane[conversation_id] = lane_key
            self._lane_last[conversation_id] = time.time()
            return True

    def release_lane(self, conversation_id):
        with self._client_lock:
            self._lanes.pop(conversation_id, None)
            self._lane_last.pop(conversation_id, None)
            self._conv_lane.pop(conversation_id, None)

    def active_lanes(self):
        return list(self._lanes.keys())

    def lanes_summary(self):
        """Phase 4 status: each registry lane + exhaustion/held-lane state."""
        self._refresh()
        out = []
        now = time.time()
        for k, cfg in self._cfg.items():
            held = [cid for cid, lk in self._conv_lane.items() if lk == k]
            ex = self._exhausted_until.get(k, 0.0)
            out.append({
                "name": k,
                "port": cfg["http_port"],
                "active": (k == self._active),
                "exhausted": ex > now,
                "exhausted_until": ex if ex > now else None,
                "held_lanes": held,
            })
        return out

    def health(self):
        self._refresh()
        if not self._cfg:
            return {"ok": False, "error": "no app lane entries in registry", "lanes": []}
        # probe the ACTIVE lane; report all lanes' status
        lane = self._active or next(iter(self._cfg.keys()))
        cfg = self._cfg[lane]
        try:
            resp = CascadeClient(cfg).ch.rpc_ok("GetUserStatus", {}, timeout=15)
            email = (((resp.get("userStatus") or {}).get("email")) or "unknown")
            return {"ok": True, "email": email, "port": cfg["http_port"],
                    "source": cfg.get("source"), "lane": lane,
                    "lanes": self.lanes_summary()}
        except Exception as ex:
            return {"ok": False, "error": str(ex)[:200], "port": cfg["http_port"],
                    "lane": lane, "lanes": self.lanes_summary()}


if __name__ == "__main__":
    import sys
    mgr = AppLaneManager()
    if "--ping" in sys.argv:
        h = mgr.health()
        print("health:", json.dumps(h))
        if not h["ok"]:
            sys.exit(1)
        print("health OK")
        if "--no-turn" not in sys.argv:
            text, _ = mgr.chat(PING_PROMPT)
            print("PING:", text[:120])
    elif "--lane" in sys.argv:
        cid = "test-lane-" + uuid.uuid4().hex[:6]
        mgr.acquire_for_lane(cid)
        t1, _ = mgr.chat_lane(cid, "Count 1. Reply with exactly: ONE")
        print("turn1:", t1[:120])
        t2, _ = mgr.chat_lane(cid, "Count 2. Reply with exactly: TWO")
        print("turn2:", t2[:120])
    else:
        text, _ = mgr.chat(" ".join(sys.argv[1:]) or PING_PROMPT)
        print(text)
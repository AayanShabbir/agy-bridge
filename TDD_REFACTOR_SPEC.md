# TDD Refactor Spec — AGY Bridge (agy-bridge)

**Owner:** Aayan Shabbir (aayan.ahmed.shabbir@gmail.com)
**Status:** APPROVED - PENDING ENGINEER EXECUTION
**Date:** 2026-09-22
**Repo:** ~/.hermes/agy-bridge (canonical origin: https://github.com/AayanShabbir/agy-bridge.git)

---

## 0. Law — read first, no exceptions

1. **ZERO behavior change.** The bridge serves the live AGY lane. Any change that alters wire
   behavior, SSE framing, tool-call shape, or error mapping is a FAILURE even if tests pass.
2. **TDD strictly.** Each gate = (a) write/collect failing tests FIRST, (b) make them pass on
   current code (characterization), (c) refactor, (d) re-run — all green, (e) commit.
   Never refactor before the tests exist for that gate. No "one big rewrite."
3. **Never touch the live lane.** Tests use a FAKE app manager (no Antigravity, no :57718,
   no :8790). The only live-lane check is Gate 4 acceptance probes, and even they run against
   the running bridge only if the lane is healthy (curl /health first; if not healthy, SKIP
   and note it — never a fail).
4. **Single identity.** Every commit: `Aayan Shabbir <aayan.ahmed.shabbir@gmail.com>`.
   (Repo-local git config is already set. Do not change it.)
5. **Sentinel discipline.** The ONLY success condition is the sentinel file. No "looks done".
6. **Import safety on host.** Verified: `import bridge` succeeds with zero side effects
   (lazy app manager). All tests MUST continue to import cleanly without the container env.

---

## 1. Current state (measured, don't re-derive)

| File | Complexity | Findings | Key hotspots |
| :--- | :--- | :--- | :--- |
| bridge.py | 368 | 24 | `parse_envelope` L440-637 (198 lines), `GatewayHandler` L643-983 (391 lines, 10 blocks), `_handle_streaming` L884-983, `_trace_call` L71 (9 params), `_handle_non_streaming` L814-879 |
| app_lane.py | 142 | 104 | deep nesting L579 (depth 7), `turn`/`_turn_impl` 7 params, `chat_lane` 6 params; 65 "unreachable_code" findings are FALSE POSITIVES (naive rule, no indent check) |
| langfuse_hook.py | 58 | 1 | clean — DO NOT TOUCH except import-safety test |

Complexity measured via Savant analyze_code (`mcp__savant_context__analyze_code` on repo agy-bridge).

---

## 2. Harness (Gate 0 — do this first, commit it)

Create `tests/` in repo root + `pytest.ini`:

- `pytest.ini`:
  ```
  [pytest]
  testpaths = tests
  addopts = -q
  ```
- `tests/conftest.py`: make `repo_root` importable (insert repo root into sys.path), define
  fixtures: `fake_app_manager` (subclass of `app_lane.AppLaneManager` overriding `chat()` /
  `chat_lane()` / `turn()` to return recorded responses without any network), `fake_registry`
  (write a temp registry.json with a fake lane entry pointing at a dead port).
- Requirement: `python3 -m pytest tests/ -q` collects 0 tests and exits 5 (no tests) — that is
  the GREEN baseline. Then Gate 1 tests make it collect real tests.
- Add `tests/` and `.pytest_cache/` and `__pycache__/` to `.gitignore` if not present.

Commit: `test(harness): pytest scaffold + fake app manager`.

---

## 3. Gate 1 — parse_envelope characterization + split

### 3.1 Write tests FIRST (tests/test_parse_envelope.py)

Characterization tests against **current** behavior — NO refactor yet. Cover, with real
messages the lane produces:

- no tools → returns `(str(raw_text), [])` for str, and `(json.dumps(x), [])` for dict/list.
- tools + list → tool_calls normalized via `_normalize_single_tool_call`, finish responses
  extracted via `_extract_finish_response`, JSON-array fallback.
- tools + dict → walks `tool_calls`/`calls`/`functions` keys, normalizes candidates.
- dict with nested `content` envelope → recursion into content.
- JSON array of dicts → ndjson semantics.
- garbage input (None, non-JSON str, empty list) → no exception, sensible default.
- tool-call args that aren't valid JSON → cleaned via `clean_tool_call_content`.
- Include 3-4 envelopes captured from the REAL bridge log (`acceptance/` probes + live
  `bridge.log` examples) as fixtures in `tests/fixtures/envelopes/*.json`.
- PATH TO GREEN: all pass against current `parse_envelope`. That IS the behavior lock.

Run: `python3 -m pytest tests/test_parse_envelope.py -q` → all green. Commit:
`test(parse_envelope): characterization tests lock current behavior`.

### 3.2 Refactor parse_envelope (bridge.py L440-637)

Split into small pure helpers **in bridge.py** (or a new `bridge_envelope.py` module —
owner preference: new module `envelope.py` keeps bridge.py focused; import it in bridge.py):

- `parse_envelope(raw_text, tools=None, result_obj=None) -> Tuple[str, list]` — thin public
  dispatcher: chooses list/dict/str path, delegates, folds `result_obj`.
- `_envelope_from_list(items, valid_names) -> Tuple[str, list]`
- `_envelope_from_dict(obj, valid_names) -> Tuple[str, list]`
- `_envelope_from_text(text, valid_names) -> Tuple[str, list]` (ndjson/JSON-array handling)
- keep `_normalize_single_tool_call`, `_extract_finish_response`, `_clean_accumulated_text`,
  `clean_tool_call_content` — already pure helpers; move them into `envelope.py`
  (update imports in bridge.py).

Hard rules:
- Do NOT change return shapes, ordering (valid_tc vs finish_responses priority: tool calls
  win over finish responses), whitespace behavior, or JSON dumps separators.
- No-tools path stays FIRST and returns raw text verbatim (documented CRITICAL behavior).
- Every helper: return type hint + 1-2 sentence docstring.

Run: full `parse_envelope` test file → green. THEN run complexity check:
`mcp__savant_context__analyze_code(repo="agy-bridge", path="bridge.py")` → expect
`parse_envelope` gone from large-block findings OR complexity of that region materially down.
Commit: `refactor(envelope): split parse_envelope into focused pure helpers (TDD, tests green)`.

---

## 4. Gate 2 — GatewayHandler decomposition

### 4.1 Write tests FIRST (tests/test_gateway.py)

Characterization tests against current handler. In-process `ThreadingHTTPServer` on
`127.0.0.1:0` (random free port), monkeypatch `bridge._app_lane_turn` (and
`_get_app_manager`) with fake that returns recorded (content, tool_calls). Cover:

- `GET /v1/models` → 200, model list JSON (use current CURATED list).
- `GET /health` → 200 with healthy shape.
- `DELETE` → existing behavior.
- `POST /v1/chat/completions` streaming=false → 200, `choices[0].message.content`
  == fake content; tool_calls passthrough shape preserved.
- `POST /v1/chat/completions` streaming=true → SSE parse: data lines contain
  `[DONE]` terminator and content chunks in order.
- Error path: `_app_lane_turn` raises → mapping to HTTP 5xx + `error` JSON shape (as today).
- Unknown route → 404.
- Each test records the CURRENT wire bytes (response body) as the expected fixture where
  the shape must be frozen.

Run → all green against current code. Commit: `test(gateway): characterize HTTP surface`.

### 4.2 Refactor GatewayHandler (bridge.py L643-983)

Split into: `GatewayHandler` (thin `do_*` dispatchers + `_send`/`_sse`) and new module
`gateway_handlers.py` with:
- `handle_models(handler) -> None`
- `handle_health(handler) -> None`
- `handle_chat_completion(handler, body, streaming: bool) -> None`
- request body parsing/validation helpers (extract model/messages/tools/tool_choice/stream
  from the OpenAI-compatible envelope) → `gateway_request.py` (pure functions).
- `_trace_call` (L71, 9 params) → collapse to `(ctx: TraceContext) -> None` where
  `TraceContext` is a small dataclass (conversation_id, prompt, model, res, t0, status,
  error, error_class) in `gateway_handlers.py`.

Hard rules:
- SSE framing bytes identical (event/data line endings, `[DONE]`, content-encoding).
- Status codes + JSON bodies identical.
- `do_POST` still handles /v1/chat/completions AND any other POST routes today.
- Keep `main()` wiring unchanged.

Run: full test_gateway.py → green. Complexity re-check: expect GatewayHandler bloat
finding resolved/gone, `_handle_streaming` split into smaller steps. Commit:
`refactor(gateway): decompose handler; _trace_call -> TraceContext (tests green)`.

---

## 5. Gate 3 — app_lane.py targeted fixes

### 5.1 Tests FIRST (tests/test_app_lane_units.py)
- `load_registry` / `load_registry_lanes` with temp registry: valid lane parses, missing
  file raises AppLaneError, empty registry raises, exhausted lane flags.
- `AppLaneManager._refresh/config load` with fake registry file: no network needed,
  uses local temp file. Current behavior = freeze.
- `resolve_model_enum` / model mappings: known names, unknown fallback, MODEL_ passthrough.
- `THINKING_BUDGET` lookup path.
- All against current code → green. Commit: `test(app_lane): unit-cover registry/model/manager init`.

### 5.2 Refactor
- L579 deep nesting (depth 7): extract the lane-drop loop into a helper
  `_drop_vanished_lanes(new_lanes: set) -> None`; early-continue structure. Depth must
  drop to <=4 there.
- `turn`/`_turn_impl` (7 params) + `chat_lane` (6 params): collate related knobs into a
  `TurnOptions` dataclass (timeout, model_enum, delete_after, on_heartbeat...). Public
  callers (bridge.py) pass options object; internal plumbing unchanged.
- Add return type hints to the flagged 34 functions (mechanical).

Hard rules: registry file format compatibility, exhaustion semantics (one-at-a-time lane
flip behavior), AppLaneError classes unchanged. No behavior drift on the live lane.

Run: full suite → green. Complexity: app_lane.py 142 → target <= 100. Commit:
`refactor(app_lane): flatten nesting, TurnOptions, type hints (tests green)`.

---

## 6. Gate 4 — Full regression + live acceptance (state gate)

1. Full suite: `python3 -m pytest tests/ -q` → ALL GREEN. 
2. Static import + compile: `python3 -m py_compile bridge.py app_lane.py envelope.py gateway_handlers.py gateway_request.py langfuse_hook.py`.
3. Live-lane health: `curl -s http://127.0.0.1:8790/health` → MUST show `healthy`.
   - IF healthy: run `AGY_BRIDGE_PORT=8791 python3 verify_bridge.py` (spare port) plus
     acceptance probes with `AGY_BRIDGE_URL=http://127.0.0.1:8790` — record outputs.
   - IF NOT healthy: SKIP live probes, note "skipped: lane not healthy at gate time" in
     the report. NOT a failure.
4. Re-run Savant analyze_code on all files; record complexity + findings diff vs [§1].
5. Write `/tmp/agy-tdd-<date>.report` with: gates, test counts, complexity before/after,
   acceptance probe outputs, any skipped live checks. Print it.

---

## 7. Deliverables & success criteria

- [ ] tests/ suite: parse_envelope, gateway, app_lane units — all green
- [ ] bridge.py complexity 368 → target <= 180
- [ ] app_lane.py complexity 142 → target <= 100
- [ ] langfuse_hook.py untouched (except import-safety)
- [ ] SENTINEL: `touch ~/.hermes/agy-bridge/.tdd-gate-all-green` ONLY after §6.1-6.2 pass; if gate-4 live probes were skipped, append `-lane-skipped` to the sentinel name.
- [ ] Commits: one per gate (§2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2), single identity, message convention per gate. Final `git log` shows 7+ commits.
- [ ] Report saved + printed.

## 8. Explicitly FORBIDDEN (fail = sentinel never reached)
- Touching :5790/:57718/:4740/:7811/:7815 or any live port.
- Editing langfuse_hook.py behavior (only `move import` if required by module split — and only with tests green first).
- Changing JSON separators, SSE framing, error JSON shapes, or tool-call arg escaping.
- Introducing new third-party deps beyond pytest.
- Committing with identity other than Aayan Shabbir <aayan.ahmed.shabbir@gmail.com>.
- "Refactor first, test later" — order is LAW.

## 9. Skip list (do NOT touch)
- README.md, LICENSE, TOKEN_TRIM.md, Dockerfile, docker-compose.yml, langfuse_hook.py (behavior),
  .veta/app-brains/registry.json, any launchd/daemon config, savant-gateway repo.
# AGY Bridge

OpenAI-compatible inference gateway for **Google Antigravity** — no API keys.

`agy-bridge` is a thin, dependency-free (Python stdlib) server that exposes an
OpenAI-style `/v1` API and forwards every request to the **running Antigravity
desktop app** on the host over its local `language_server` bridge (gRPC-web +
JSON). Auth is the app's own signed-in consumer session; the bridge itself
stores **zero credentials** and requires **no Google API keys**.

Designed to be containerized: the bridge runs in Docker, the Antigravity app
stays host-side as the auth + agent engine.

## Why

The Antigravity desktop app ships with a local agent bridge (`language_server`)
that authenticates as the signed-in consumer account. `agy-bridge` rides that
bridge and turns it into a standard OpenAI-compatible endpoint that any tool —
LLM agents, scripts, IDEs — can point at with zero cloud configuration.

- No `GEMINI_API_KEY`, no SDK credentials, no OAuth flow to manage.
- Works with the consumer (paid subscription) account the app is logged into.
- Streaming + non-streaming, stateless calls and stateful conversation lanes.

## Architecture

```
your agent / any OpenAI client
        │  POST /v1/chat/completions, /v1/conversations
        ▼
agy-bridge (this repo, Docker, :8790)
        │  host.docker.internal:<port>  (gRPC-web + JSON, x-codeium-csrf-token)
        ▼
Antigravity app (host) — language_server agent bridge
        │  consumer OAuth session
        ▼
Google Antigravity backend (paid subscription lane)
```

The bridge discovers the app's current "door" (local port + CSRF token) from a
small JSON registry that a host-side watcher keeps up to date — so it follows
the app across restarts instead of holding a stale port.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/v1/models` | OpenAI model list (app-served model enums) |
| POST | `/v1/chat/completions` | Chat completion (non-streaming + SSE streaming) |
| POST | `/v1/conversations` | Open a stateful conversation lane |
| GET | `/v1/conversations` | List active lanes |
| DELETE | `/v1/conversations/{id}` | Close a lane |
| GET | `/health` | Liveness + app-lane stats |

## Quickstart

1. Install and sign in to the Antigravity desktop app on the host.
2. Point a watcher at the app's local bridge so the registry stays current
   (the registry file defaults to `/veta/app-brains/registry.json` inside the
   container and must contain `{ "http_port": <port>, "csrf": "<token>" }`).
3. Configure and run:

```bash
cp docker-compose.yml docker-compose.override.yml   # set your paths, see below
docker compose up -d --build
curl http://127.0.0.1:8790/health
```

```bash
curl http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"ping"}]}'
```

### docker-compose volume mounts

The compose file uses these variables (set them in your environment or an
override file):

| Variable | Purpose |
| --- | --- |
| `HERMES_ENV_FILE` | Host `.env` with non-secret runtime settings, mounted read-only to `/root/.hermes/.env` |
| `APP_BRAINS_DIR` | Host directory holding the app-brains registry JSON, mounted read-only to `/veta/app-brains` |

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BIND_HOST` | `127.0.0.1` | Listen interface |
| `AGY_BRIDGE_PORT` | `8790` | Listen port |
| `AGY_APP_REGISTRY` | `/veta/app-brains/registry.json` | Registry file (door + csrf) |
| `AGY_APP_HOST_OVERRIDE` | `host.docker.internal` | Host address of the Antigravity app |
| `AGY_APP_MODEL_ENUM` | `MODEL_PLACEHOLDER_M319` | Default model enum |
| `AGY_BRIDGE_MODEL` / `AGY_FLASH_MODEL` / `AGY_LOW_MODEL` / `AGY_PRO_MODEL` | — | Model alias defaults |
| `LANGFUSE_ENABLED` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | — | Optional LangFuse telemetry (fail-open, disabled when keys absent) |

The bridge reads runtime settings from `~/.hermes/.env` if present. **No secret
values are required** — the lane is app-auth only.

## Security notes

- The bridge binds to `127.0.0.1` by default (host mapping keeps it out of LAN
  exposure); the compose file maps `127.0.0.1:8790` explicitly.
- Credentials never enter the container: auth lives in the host app's session.
- LangFuse telemetry is **fail-open** — it never blocks or slows inference.
- The Antigravity app's internal agent bridge is an undocumented, consumer-OAuth
  interface. Use at your own risk; this is an unofficial integration and may
  break or violate the provider's terms of service.

## Development

```bash
# Run on the host (no Docker)
AGY_APP_REGISTRY=/path/to/registry.json python3 bridge.py

# Probes / acceptance checks
python3 acceptance/auth_probe.py
python3 acceptance/registry_probe.py
python3 acceptance/tool_envelope_probe.py

# Quick throughput check
python3 bench.py
python3 verify_bridge.py
```

## Author

Aayan Shabbir — aayan.ahmed.shabbir@gmail.com

## License

MIT — see [LICENSE](LICENSE).

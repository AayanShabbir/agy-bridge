"""Tests for Phase 11: Host Deployment & Bootstrapping (Section 15)."""
import json
import pytest
from pathlib import Path
from starlette.testclient import TestClient

from agy_bridge.bootstrap import build_bridge_app
from agy_bridge.config import BridgeConfig


def test_bootstrap_builds_and_runs_app(tmp_path: Path):
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "active": "main",
                "lanes": {
                    "main": {
                        "http_port": 63613,
                        "csrf": "secret-csrf-token",
                        "model": "gemini-3.8-flash",
                    }
                },
            }
        )
    )

    config = BridgeConfig(
        registry_path=str(reg_file),
        host="127.0.0.1",
        port=8790,
        enable_supervisor=False,
    )

    app = build_bridge_app(config)
    client = TestClient(app)

    # Health check
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Models check
    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    assert len(r_models.json()["data"]) > 0


def test_bootstrap_wires_real_engine_that_hits_transport(tmp_path: Path):
    """The bootstrap engine is the real AgyCompletionEngine: a chat request
    reaches the transport and fails on the dead upstream instead of returning
    the fabricated '[Attributable answer]' placeholder."""
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "active": "main",
                "lanes": {
                    "main": {
                        "http_port": 9,
                        "csrf": "secret-csrf-token",
                        "model": "gemini-3.8-flash",
                    }
                },
            }
        )
    )
    config = BridgeConfig(
        registry_path=str(reg_file),
        host="127.0.0.1",
        port=8790,
        enable_supervisor=False,
    )
    app = build_bridge_app(config)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    # Dead upstream -> typed 503, proving the real transport was driven.
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_unavailable"

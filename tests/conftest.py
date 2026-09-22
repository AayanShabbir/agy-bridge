"""Shared pytest fixtures for agy-bridge characterization tests.

All tests are OFFLINE: no Antigravity, no :57718, no :8790. The fake app
manager substitutes the real AppLaneManager and returns recorded responses;
the fake registry writes a temp registry.json pointing at a dead port.
"""
import json
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `import bridge` / `import app_lane` work
# without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fake_registry(tmp_path):
    """Write a temp registry.json with one fake lane on a dead port.

    Returns the dict with keys: path (the registry file), and the lane
    config (http_port dead + csrf + model). The dead port (9) never accepts
    connections, so no path can accidentally reach a real lane.
    """
    lane = {"http_port": 9, "csrf": "fake-csrf-token", "model": "gemini-3.8-flash"}
    data = {"active": "main", "lanes": {"main": lane}}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    return {"path": str(path), "lane": dict(lane), "data": data}


class FakeAppLaneManager:
    """AppLaneManager-compatible stub returning recorded responses.

    Overrides the chat / chat_lane / turn entry points so no network is ever
    touched, and records calls for assertion. Mirrors the methods bridge.py
    and app_lane expose: chat(), chat_lane(), turn(), acquire_for_lane(),
    release_lane(), lanes(), health().
    """

    def __init__(self, response=("hello from fake lane", {}), tool_calls=None):
        self._response = response
        self._tool_calls = tool_calls or []
        self.calls = []
        self._lanes = set()
        self._health = {"ok": True, "email": "fake@example.com", "port": 9,
                        "source": "registry", "lane": "main", "lanes": []}

    def chat(self, prompt, model_enum=None, timeout=240, on_heartbeat=None):
        self.calls.append(("chat", prompt, model_enum))
        return self._response

    def chat_lane(self, conversation_id, user_text, timeout=240.0, model_enum=None, on_heartbeat=None):
        self.calls.append(("chat_lane", conversation_id, user_text, model_enum))
        return self._response

    def turn(self, user_text, timeout=240.0, model_enum=None, on_heartbeat=None, delete_after=False):
        self.calls.append(("turn", user_text, model_enum))
        return self._response

    def acquire_for_lane(self, conversation_id):
        self._lanes.add(conversation_id)
        return True

    def release_lane(self, conversation_id):
        self._lanes.discard(conversation_id)

    def lanes(self):
        return list(self._lanes)

    def health(self):
        h = dict(self._health)
        h["lanes"] = [{"name": "main", "port": 9, "active": True, "exhausted": False,
                       "exhausted_until": None, "held_lanes": []}]
        return h


@pytest.fixture
def fake_app_manager():
    """Yield a FakeAppLaneManager whose response can be mutated per-test."""
    mgr = FakeAppLaneManager()
    yield mgr
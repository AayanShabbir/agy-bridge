import sqlite3
import time

from agy_bridge import telemetry


def test_emit_call_is_fail_open_and_writes_ring(monkeypatch, tmp_path):
    db = tmp_path / "jev_ring.db"
    monkeypatch.setattr(telemetry, "_RING_DB_PATH", str(db))
    monkeypatch.setattr(telemetry, "_RING_ENABLED", True)
    monkeypatch.setattr(telemetry._ring_tls, "conn", None, raising=False)
    monkeypatch.setenv("JEV_RING_ENABLED", "1")
    monkeypatch.setattr(telemetry, "_emit", lambda *_: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    trace_id = telemetry.emit_call(model="test-model", prompt="input", output="output",
                                   start_time=time.time() - 1, end_time=time.time(), status="success")
    assert trace_id
    with sqlite3.connect(db) as conn:
        assert conn.execute("select model,status from jev_events").fetchone() == ("test-model", "success")

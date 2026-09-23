"""Test clean importability and absence of runtime side effects."""
import sys
import threading
from pathlib import Path


def test_clean_import_no_side_effects():
    """Importing agy_bridge must have zero runtime side effects."""
    # Ensure src is on sys.path
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    threads_before = threading.active_count()

    import agy_bridge
    from agy_bridge import errors, domain

    threads_after = threading.active_count()
    assert threads_after == threads_before, "Importing agy_bridge spawned background threads!"
    assert hasattr(agy_bridge, "__version__")
    assert agy_bridge.__version__ == "0.2.0"


def test_errors_hierarchy_and_rendering():
    """Verify error classes and standard OpenAI error payload generation."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agy_bridge.errors import (
        AgyBridgeError,
        ProtocolError,
        FrameTooLargeError,
        RateLimitExceeded,
        UpstreamAuthRequired,
        UpstreamTimeout,
        LeaseConflictError,
        UnsupportedImageError,
    )

    err = FrameTooLargeError("Frame size 10MB exceeds 8MB limit")
    assert isinstance(err, ProtocolError)
    assert isinstance(err, AgyBridgeError)
    assert err.http_status == 413
    assert err.error_code == "frame_too_large"

    payload = err.to_dict()
    assert "error" in payload
    assert payload["error"]["code"] == "frame_too_large"
    assert payload["error"]["message"] == "Frame size 10MB exceeds 8MB limit"

    rate_err = RateLimitExceeded(retry_after_s=60)
    assert rate_err.http_status == 429
    assert rate_err.retry_after_s == 60

    auth_err = UpstreamAuthRequired("Token expired")
    assert auth_err.http_status == 503

    conflict_err = LeaseConflictError("Turn already in flight")
    assert conflict_err.http_status == 409

    image_err = UnsupportedImageError("Images not allowed")
    assert image_err.http_status == 400


def test_domain_models():
    """Verify domain enums and dataclasses."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agy_bridge.domain import (
        TurnState,
        SubmissionCertainty,
        AccountState,
        EngineGeneration,
        Lease,
        TurnObservation,
    )

    assert TurnState.QUEUED.value == "QUEUED"
    assert TurnState.RUNNING.value == "RUNNING"
    assert SubmissionCertainty.OUTCOME_UNKNOWN.value == "OUTCOME_UNKNOWN"
    assert AccountState.COOLDOWN.value == "COOLDOWN"

    gen = EngineGeneration(
        generation_id="gen-123",
        observed_at=1000.0,
        endpoint="http://127.0.0.1:57718",
        csrf_token="secret-token",
    )
    assert "secret-token" not in repr(gen)  # CSRF token must be redacted/omitted from repr

    import time
    now = time.monotonic()
    lease = Lease(
        lease_id="l-1",
        resource_id="conv-42",
        acquired_at=now,
        expires_at=now + 100.0,
        holder_id="req-99",
    )
    assert not lease.is_expired
    expired_lease = Lease(
        lease_id="l-2",
        resource_id="conv-42",
        acquired_at=now - 200.0,
        expires_at=now - 100.0,
        holder_id="req-99",
    )
    assert expired_lease.is_expired

    obs = TurnObservation(
        request_id="req-1",
        turn_id="turn-1",
        conversation_id="conv-1",
        generation="gen-123",
    )
    assert obs.state == TurnState.QUEUED
    assert obs.certainty == SubmissionCertainty.NOT_SENT

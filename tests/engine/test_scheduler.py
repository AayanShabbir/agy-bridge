"""Tests for Phase 6: Scheduling, Account Resilience & Failover (Section 11)."""
import time
import pytest

from agy_bridge.domain import AccountState, SubmissionCertainty
from agy_bridge.errors import RateLimitExceeded, UpstreamUnavailable
from agy_bridge.engine.registry import EngineEndpoint
from agy_bridge.engine.scheduler import AccountScheduler, AccountStatusTracker


def make_endpoint(account_id: str, port: int) -> EngineEndpoint:
    return EngineEndpoint(
        account_id=account_id,
        http_port=port,
        csrf_secret=f"token-{account_id}",
        capability_profile="gemini-3.8-flash",
    )


def test_quota_exhaustion_sets_account_cooldown_without_restart():
    tracker = AccountStatusTracker()
    now = 1000.0

    # Report 429 quota exhaustion
    tracker.record_failure("personal", error_type="quota", retry_after_s=30, now=now)

    state = tracker.get_state("personal", now=now)
    assert state == AccountState.COOLDOWN
    assert tracker.is_available("personal", now=now) is False

    # After 31 seconds, cooldown expires -> READY for probe
    assert tracker.get_state("personal", now=now + 31.0) == AccountState.READY
    assert tracker.is_available("personal", now=now + 31.0) is True


def test_all_accounts_cooling_down_raises_rate_limit_exceeded():
    tracker = AccountStatusTracker()
    ep1 = make_endpoint("personal", 63613)
    ep2 = make_endpoint("vetafleet", 63614)
    scheduler = AccountScheduler(tracker=tracker, endpoints={"personal": ep1, "vetafleet": ep2})

    now = 1000.0
    tracker.record_failure("personal", error_type="quota", retry_after_s=20, now=now)
    tracker.record_failure("vetafleet", error_type="quota", retry_after_s=45, now=now)

    # Attempt to schedule should raise RateLimitExceeded with minimum retry-after (20s)
    with pytest.raises(RateLimitExceeded) as exc_info:
        scheduler.select_endpoint(conversation_id=None, pinned_account=None, now=now)
    assert exc_info.value.retry_after_s == 20


def test_all_accounts_unavailable_raises_upstream_unavailable():
    tracker = AccountStatusTracker()
    ep1 = make_endpoint("personal", 63613)
    scheduler = AccountScheduler(tracker=tracker, endpoints={"personal": ep1})

    now = 1000.0
    tracker.record_failure("personal", error_type="connection_refused", now=now)

    with pytest.raises(UpstreamUnavailable):
        scheduler.select_endpoint(conversation_id=None, pinned_account=None, now=now)


def test_stateless_failover_selects_next_ready_account():
    tracker = AccountStatusTracker()
    ep1 = make_endpoint("personal", 63613)
    ep2 = make_endpoint("vetafleet", 63614)
    scheduler = AccountScheduler(
        tracker=tracker,
        endpoints={"personal": ep1, "vetafleet": ep2},
        preferred_account="personal",
    )

    now = 1000.0
    tracker.record_failure("personal", error_type="quota", retry_after_s=30, now=now)

    # Stateless request (conversation_id=None) fails over to vetafleet
    selected = scheduler.select_endpoint(conversation_id=None, pinned_account=None, now=now)
    assert selected.account_id == "vetafleet"
    assert selected.http_port == 63614


def test_stateful_conversation_remains_pinned_to_account():
    tracker = AccountStatusTracker()
    ep1 = make_endpoint("personal", 63613)
    ep2 = make_endpoint("vetafleet", 63614)
    scheduler = AccountScheduler(
        tracker=tracker,
        endpoints={"personal": ep1, "vetafleet": ep2},
        preferred_account="personal",
    )

    now = 1000.0
    tracker.record_failure("personal", error_type="quota", retry_after_s=30, now=now)

    # Stateful conversation pinned to personal MUST NOT failover to vetafleet
    with pytest.raises(RateLimitExceeded):
        scheduler.select_endpoint(
            conversation_id="conv-1",
            pinned_account="personal",
            now=now,
        )


def test_half_open_probe_admission_after_cooldown():
    tracker = AccountStatusTracker()
    ep1 = make_endpoint("personal", 63613)
    scheduler = AccountScheduler(tracker=tracker, endpoints={"personal": ep1})

    now = 1000.0
    tracker.record_failure("personal", error_type="quota", retry_after_s=10, now=now)

    # During cooldown -> raises 429
    with pytest.raises(RateLimitExceeded):
        scheduler.select_endpoint(conversation_id=None, pinned_account=None, now=now + 5.0)

    # After cooldown -> admits probe
    selected = scheduler.select_endpoint(conversation_id=None, pinned_account=None, now=now + 11.0)
    assert selected.account_id == "personal"

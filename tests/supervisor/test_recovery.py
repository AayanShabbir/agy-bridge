"""Tests for Phase 7: Host Supervisor & Recovery Coordinator (Section 9 & 6.3)."""
import time
import pytest
from typing import Optional

from agy_bridge.domain import AccountState
from agy_bridge.errors import QuarantinedError
from agy_bridge.engine.recovery import (
    RecoveryCoordinator,
    RecoveryReason,
    RestartOutcome,
    RecoveryBudgetExhaustedError,
)
from agy_bridge.supervisor.interface import EngineSupervisor


class FakeSupervisor(EngineSupervisor):
    def __init__(self, succeeds: bool = True, new_gen: str = "gen-new-123") -> None:
        self.succeeds = succeeds
        self.new_gen = new_gen
        self.restart_calls = []

    def restart(self, engine_id: str, expected_generation: str, reason: RecoveryReason) -> RestartOutcome:
        self.restart_calls.append((engine_id, expected_generation, reason))
        if not self.succeeds:
            return RestartOutcome(success=False, error="Supervisor execution failed")
        return RestartOutcome(success=True, new_generation=self.new_gen)


def test_recovery_refuses_restart_for_forbidden_reasons():
    supervisor = FakeSupervisor()
    coordinator = RecoveryCoordinator(supervisor=supervisor, max_restarts_per_hour=2)

    forbidden = [
        RecoveryReason.QUOTA_EXHAUSTED,
        RecoveryReason.AUTH_EXPIRED,
        RecoveryReason.INVALID_INPUT,
        RecoveryReason.CLIENT_CANCELLED,
    ]

    for reason in forbidden:
        outcome = coordinator.request_restart(
            engine_id="eng-1",
            current_generation="gen-1",
            reason=reason,
            now=1000.0,
        )
        assert outcome.success is False
        assert "forbidden" in (outcome.error or "").lower()
    assert len(supervisor.restart_calls) == 0


def test_recovery_executes_for_engine_dead():
    supervisor = FakeSupervisor(succeeds=True, new_gen="gen-2")
    coordinator = RecoveryCoordinator(supervisor=supervisor, max_restarts_per_hour=2)

    outcome = coordinator.request_restart(
        engine_id="eng-1",
        current_generation="gen-1",
        reason=RecoveryReason.ENGINE_DEAD,
        now=1000.0,
    )
    assert outcome.success is True
    assert outcome.new_generation == "gen-2"
    assert len(supervisor.restart_calls) == 1


def test_recovery_budget_and_cooldown_enforced():
    supervisor = FakeSupervisor(succeeds=True, new_gen="gen-2")
    coordinator = RecoveryCoordinator(
        supervisor=supervisor,
        max_restarts_per_hour=2,
        cooldown_seconds=600.0,  # 10 min
    )

    now = 1000.0
    # First restart succeeds
    out1 = coordinator.request_restart("eng-1", "gen-1", RecoveryReason.ENGINE_DEAD, now=now)
    assert out1.success is True

    # Immediate second restart fails due to cooldown (within 10 min)
    out2 = coordinator.request_restart("eng-1", "gen-2", RecoveryReason.ENGINE_DEAD, now=now + 60.0)
    assert out2.success is False
    assert "cooldown" in (out2.error or "").lower()

    # After cooldown (11 min later), second restart succeeds
    out3 = coordinator.request_restart("eng-1", "gen-2", RecoveryReason.ENGINE_DEAD, now=now + 660.0)
    assert out3.success is True

    # Third restart within the same hour fails due to budget limit (2/hr)
    out4 = coordinator.request_restart("eng-1", "gen-2", RecoveryReason.ENGINE_DEAD, now=now + 1300.0)
    assert out4.success is False
    assert "budget" in (out4.error or "").lower()


def test_generation_mismatch_skips_restart():
    supervisor = FakeSupervisor(succeeds=True, new_gen="gen-2")
    coordinator = RecoveryCoordinator(supervisor=supervisor)

    # If active generation is already newer than expected, skip restart
    coordinator.record_active_generation("eng-1", "gen-2")
    out = coordinator.request_restart("eng-1", "gen-1", RecoveryReason.ENGINE_DEAD, now=1000.0)
    assert out.success is True
    assert out.skipped is True
    assert len(supervisor.restart_calls) == 0

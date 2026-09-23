"""Recovery coordinator and restart budgeting (Section 9)."""
from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Optional

from agy_bridge.errors import AgyBridgeError
from agy_bridge.supervisor.interface import EngineSupervisor, RestartOutcome


class RecoveryReason(str, Enum):
    """Allowed and forbidden recovery trigger reasons."""
    ENGINE_DEAD = "ENGINE_DEAD"
    HANG_CONFIRMED = "HANG_CONFIRMED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    INVALID_INPUT = "INVALID_INPUT"
    CLIENT_CANCELLED = "CLIENT_CANCELLED"


FORBIDDEN_RECOVERY_REASONS = {
    RecoveryReason.QUOTA_EXHAUSTED,
    RecoveryReason.AUTH_EXPIRED,
    RecoveryReason.INVALID_INPUT,
    RecoveryReason.CLIENT_CANCELLED,
}


class RecoveryBudgetExhaustedError(AgyBridgeError):
    """Raised when maximum restart rate budget has been exceeded."""
    http_status = 503
    error_code = "recovery_budget_exhausted"


class RecoveryCoordinator:
    """Manages restart rates, cooldowns, and supervisor coordination."""

    def __init__(
        self,
        supervisor: EngineSupervisor,
        max_restarts_per_hour: int = 2,
        cooldown_seconds: float = 600.0,
    ) -> None:
        self.supervisor = supervisor
        self.max_restarts_per_hour = max_restarts_per_hour
        self.cooldown_seconds = cooldown_seconds
        self._restart_history: Dict[str, List[float]] = {}
        self._last_restart: Dict[str, float] = {}
        self._active_generations: Dict[str, str] = {}

    def record_active_generation(self, engine_id: str, generation: str) -> None:
        self._active_generations[engine_id] = generation

    def request_restart(
        self,
        engine_id: str,
        current_generation: str,
        reason: RecoveryReason,
        now: Optional[float] = None,
    ) -> RestartOutcome:
        current_time = time.monotonic() if now is None else now

        if reason in FORBIDDEN_RECOVERY_REASONS:
            return RestartOutcome(
                success=False,
                error=f"Restart forbidden for reason: {reason.value}",
            )

        # Check if generation already moved ahead
        known_gen = self._active_generations.get(engine_id)
        if known_gen and known_gen != current_generation:
            return RestartOutcome(
                success=True,
                new_generation=known_gen,
                skipped=True,
            )

        # Enforce cooldown
        last = self._last_restart.get(engine_id, 0.0)
        if (current_time - last) < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (current_time - last))
            return RestartOutcome(
                success=False,
                error=f"Engine {engine_id!r} in restart cooldown ({remaining}s remaining)",
            )

        # Enforce hourly budget
        history = self._restart_history.setdefault(engine_id, [])
        one_hour_ago = current_time - 3600.0
        history = [t for t in history if t > one_hour_ago]
        self._restart_history[engine_id] = history

        if len(history) >= self.max_restarts_per_hour:
            return RestartOutcome(
                success=False,
                error=f"Engine {engine_id!r} restart budget exhausted ({len(history)}/{self.max_restarts_per_hour} in last hour)",
            )

        # Execute restart through supervisor
        outcome = self.supervisor.restart(engine_id, current_generation, reason.value)
        if outcome.success:
            history.append(current_time)
            self._last_restart[engine_id] = current_time
            if outcome.new_generation:
                self._active_generations[engine_id] = outcome.new_generation

        return outcome

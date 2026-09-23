"""Account status tracking, cooldown scheduling, and pinned failover (Section 11)."""
from __future__ import annotations

import time
import math
from typing import Dict, Optional, Tuple

from agy_bridge.domain import AccountState
from agy_bridge.errors import RateLimitExceeded, UpstreamUnavailable
from agy_bridge.engine.registry import EngineEndpoint


class AccountStatusTracker:
    """Tracks per-account health states, exponential cooldowns, and probe allowances."""

    def __init__(self, default_cooldown_s: float = 30.0, max_cooldown_s: float = 300.0) -> None:
        self.default_cooldown_s = default_cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self._states: Dict[str, AccountState] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._error_types: Dict[str, str] = {}

    def record_failure(
        self,
        account_id: str,
        error_type: str,
        retry_after_s: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        current_time = time.monotonic() if now is None else now
        failures = self._consecutive_failures.get(account_id, 0) + 1
        self._consecutive_failures[account_id] = failures
        self._error_types[account_id] = error_type

        if error_type in ("quota", "429", "resource_exhausted"):
            self._states[account_id] = AccountState.COOLDOWN
            if retry_after_s is not None and retry_after_s > 0:
                duration = float(retry_after_s)
            else:
                # Exponential backoff base default_cooldown_s
                duration = min(self.default_cooldown_s * math.pow(1.5, failures - 1), self.max_cooldown_s)
            self._cooldown_until[account_id] = current_time + duration
        elif error_type in ("auth_required", "not_logged_in"):
            self._states[account_id] = AccountState.AUTH_REQUIRED
        else:
            self._states[account_id] = AccountState.UNAVAILABLE

    def record_success(self, account_id: str) -> None:
        self._states[account_id] = AccountState.READY
        self._consecutive_failures[account_id] = 0
        if account_id in self._cooldown_until:
            del self._cooldown_until[account_id]
        if account_id in self._error_types:
            del self._error_types[account_id]

    def get_state(self, account_id: str, now: Optional[float] = None) -> AccountState:
        current_time = time.monotonic() if now is None else now
        current_state = self._states.get(account_id, AccountState.READY)

        if current_state == AccountState.COOLDOWN:
            cooldown_end = self._cooldown_until.get(account_id, 0.0)
            if current_time >= cooldown_end:
                # Cooldown expired; return READY to allow half-open probe
                return AccountState.READY
        return current_state

    def is_available(self, account_id: str, now: Optional[float] = None) -> bool:
        return self.get_state(account_id, now=now) == AccountState.READY

    def remaining_cooldown(self, account_id: str, now: Optional[float] = None) -> float:
        current_time = time.monotonic() if now is None else now
        cooldown_end = self._cooldown_until.get(account_id, 0.0)
        return max(0.0, cooldown_end - current_time)


class AccountScheduler:
    """Selects eligible endpoints honoring preferred accounts, cooldowns, and conversation pinning."""

    def __init__(
        self,
        tracker: AccountStatusTracker,
        endpoints: Dict[str, EngineEndpoint],
        preferred_account: Optional[str] = None,
    ) -> None:
        self.tracker = tracker
        self.endpoints = endpoints
        self.preferred_account = preferred_account

    def select_endpoint(
        self,
        conversation_id: Optional[str] = None,
        pinned_account: Optional[str] = None,
        now: Optional[float] = None,
    ) -> EngineEndpoint:
        current_time = time.monotonic() if now is None else now

        if not self.endpoints:
            raise UpstreamUnavailable("No endpoints registered in scheduler")

        # Stateful conversations must respect pinning
        if conversation_id and pinned_account:
            if pinned_account not in self.endpoints:
                raise UpstreamUnavailable(f"Pinned account {pinned_account!r} not available")
            if not self.tracker.is_available(pinned_account, now=current_time):
                remaining = int(math.ceil(self.tracker.remaining_cooldown(pinned_account, now=current_time)))
                raise RateLimitExceeded(
                    f"Pinned account {pinned_account!r} for conversation {conversation_id!r} is cooling down",
                    retry_after_s=max(1, remaining),
                )
            return self.endpoints[pinned_account]

        # Check preferred account first
        if self.preferred_account and self.preferred_account in self.endpoints:
            if self.tracker.is_available(self.preferred_account, now=current_time):
                return self.endpoints[self.preferred_account]

        # Failover search for any other READY account
        ready_endpoints = [
            ep for acct, ep in self.endpoints.items()
            if self.tracker.is_available(acct, now=current_time)
        ]
        if ready_endpoints:
            return ready_endpoints[0]

        # No accounts ready: determine if all are cooling down vs unavailable
        cooldown_remains = [
            self.tracker.remaining_cooldown(acct, now=current_time)
            for acct in self.endpoints
            if self.tracker.get_state(acct, now=current_time) == AccountState.COOLDOWN
        ]
        if cooldown_remains:
            min_cooldown = int(math.ceil(min(cooldown_remains)))
            raise RateLimitExceeded(
                f"All eligible accounts are currently cooling down ({len(cooldown_remains)} accounts)",
                retry_after_s=max(1, min_cooldown),
            )

        raise UpstreamUnavailable("All registered engine accounts are unavailable")

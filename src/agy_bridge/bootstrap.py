"""Bootstrap coordinator wiring domain engines to API façade (Section 3 & 15)."""
from __future__ import annotations

import os
from typing import Any
from fastapi import FastAPI

from agy_bridge.config import BridgeConfig
from agy_bridge.engine.agent import AgyCompletionEngine
from agy_bridge.engine.registry import RegistryManager
from agy_bridge.engine.leases import LeaseCoordinator
from agy_bridge.engine.scheduler import AccountStatusTracker, AccountScheduler
from agy_bridge.engine.recovery import RecoveryCoordinator
from agy_bridge.engine.transport import AgyHttpTransport
from agy_bridge.supervisor.interface import EngineSupervisor, RestartOutcome
from agy_bridge.api.app import create_app


class DisabledSupervisor(EngineSupervisor):
    def restart(self, engine_id: str, expected_generation: str, reason: str) -> RestartOutcome:
        return RestartOutcome(success=False, error="Supervisor is disabled on this host")


def build_bridge_app(config: BridgeConfig) -> FastAPI:
    """Build and wire all three layers into an executable FastAPI application."""
    registry = RegistryManager(registry_path=config.registry_path)
    leases = LeaseCoordinator(
        max_engine_concurrency=config.max_engine_concurrency,
        max_queue_size=config.max_queue_size,
    )
    tracker = AccountStatusTracker()

    # Discover initial endpoints if available
    try:
        active_ep = registry.get_active_endpoint()
        endpoints = {active_ep.account_id: active_ep}
        preferred = active_ep.account_id
    except Exception:
        endpoints = {}
        preferred = None

    scheduler = AccountScheduler(
        tracker=tracker,
        endpoints=endpoints,
        preferred_account=preferred,
    )

    supervisor = DisabledSupervisor()
    recovery = RecoveryCoordinator(
        supervisor=supervisor,
        max_restarts_per_hour=config.max_restarts_per_hour,
    )

    transport = AgyHttpTransport(host_override=os.environ.get("AGY_APP_HOST_OVERRIDE"))
    engine = AgyCompletionEngine(
        registry=registry,
        leases=leases,
        scheduler=scheduler,
        recovery=recovery,
        transport=transport,
    )

    app = create_app(engine=engine)
    return app

"""Bridge configuration dataclasses and environment loading."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BridgeConfig:
    registry_path: str = os.environ.get("AGY_APP_REGISTRY", "/veta/app-brains/registry.json")
    host: str = "127.0.0.1"
    port: int = 8790
    max_engine_concurrency: int = 1
    max_queue_size: int = 32
    max_restarts_per_hour: int = 2
    enable_supervisor: bool = False
    log_level: str = "info"

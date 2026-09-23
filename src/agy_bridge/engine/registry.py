"""Engine registry and generation handling (Section 10)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from agy_bridge.errors import AgyBridgeError as BridgeError
except ImportError:
    class BridgeError(Exception):
        pass


class RegistryError(BridgeError):
    """Base error for registry operations."""


class StaleRegistryError(RegistryError):
    """Raised when registry data exceeds maximum allowed staleness."""


class InvalidRegistryError(RegistryError):
    """Raised when registry format or values are invalid."""


@dataclass
class EngineEndpoint:
    """Normalized upstream language server endpoint descriptor."""
    account_id: str
    http_port: int
    csrf_secret: str
    engine_id: str = "language_server"
    host: str = "127.0.0.1"
    grpc_port: Optional[int] = None
    generation: str = ""
    observed_at: float = 0.0
    capability_profile: str = "gemini-3.8-flash"

    def __post_init__(self) -> None:
        if not self.generation:
            raw = f"{self.account_id}:{self.host}:{self.http_port}:{self.grpc_port}:{self.csrf_secret}"
            self.generation = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"EngineEndpoint(account_id={self.account_id!r}, host={self.host!r}, "
            f"http_port={self.http_port}, csrf_secret='[REDACTED]', "
            f"generation={self.generation!r}, capability_profile={self.capability_profile!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def redacted_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "account_id": self.account_id,
            "host": self.host,
            "http_port": self.http_port,
            "grpc_port": self.grpc_port,
            "csrf_secret": "[REDACTED]",
            "generation": self.generation,
            "observed_at": self.observed_at,
            "capability_profile": self.capability_profile,
        }


class RegistryManager:
    """Manages discovery, snapshot caching, generation hashing, and admission."""

    def __init__(
        self,
        registry_path: str = "/veta/app-brains/registry.json",
        max_staleness_seconds: float = 30.0,
    ) -> None:
        self.registry_path = registry_path
        self.max_staleness_seconds = max_staleness_seconds
        self._endpoints: Dict[str, EngineEndpoint] = {}
        self._preferred_account: Optional[str] = None
        self._last_valid_observed_at: Optional[float] = None
        self._last_valid_endpoint: Optional[EngineEndpoint] = None

    def _validate_port(self, raw_port: Any) -> int:
        if raw_port is None:
            raise InvalidRegistryError("Port cannot be None")
        try:
            port = int(raw_port)
        except (ValueError, TypeError):
            raise InvalidRegistryError(f"Invalid port value: {raw_port!r}")
        if port <= 0 or port > 65535:
            raise InvalidRegistryError(f"Port {port} out of range [1, 65535]")
        return port

    def _parse_data(self, data: Any, now: float) -> Tuple[Dict[str, EngineEndpoint], Optional[str]]:
        if not isinstance(data, dict):
            raise InvalidRegistryError(f"Expected dict root in registry, got {type(data)}")

        endpoints: Dict[str, EngineEndpoint] = {}
        preferred: Optional[str] = None

        if "lanes" in data:
            lanes = data.get("lanes")
            if not isinstance(lanes, dict):
                raise InvalidRegistryError("Field 'lanes' must be a dict")
            preferred = data.get("active")
            for acct_id, lane_data in lanes.items():
                if not isinstance(lane_data, dict):
                    continue
                port = self._validate_port(lane_data.get("http_port"))
                csrf = str(lane_data.get("csrf") or lane_data.get("csrf_secret") or "")
                model = str(lane_data.get("model") or "gemini-3.8-flash")
                ep = EngineEndpoint(
                    account_id=str(acct_id),
                    http_port=port,
                    csrf_secret=csrf,
                    observed_at=now,
                    capability_profile=model,
                )
                endpoints[str(acct_id)] = ep

            if not endpoints:
                raise InvalidRegistryError("No valid lanes found in registry")
            if preferred not in endpoints:
                preferred = next(iter(endpoints.keys()))
        elif "http_port" in data:
            # Flat legacy format
            port = self._validate_port(data.get("http_port"))
            csrf = str(data.get("csrf") or data.get("csrf_secret") or "")
            model = str(data.get("model") or "gemini-3.8-flash")
            preferred = "legacy"
            ep = EngineEndpoint(
                account_id="legacy",
                http_port=port,
                csrf_secret=csrf,
                observed_at=now,
                capability_profile=model,
            )
            endpoints["legacy"] = ep
        else:
            raise InvalidRegistryError("Registry missing both 'lanes' and 'http_port'")

        return endpoints, preferred

    def refresh(self, now: Optional[float] = None) -> EngineEndpoint:
        current_time = time.time() if now is None else now

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = json.loads(content)
            endpoints, preferred = self._parse_data(data, current_time)
            self._endpoints = endpoints
            self._preferred_account = preferred
            self._last_valid_observed_at = current_time
            active_ep = self._endpoints[preferred] if preferred else next(iter(self._endpoints.values()))
            self._last_valid_endpoint = active_ep
            return active_ep
        except (json.JSONDecodeError, OSError, InvalidRegistryError) as exc:
            if self._last_valid_endpoint is not None and self._last_valid_observed_at is not None:
                elapsed = current_time - self._last_valid_observed_at
                if elapsed <= self.max_staleness_seconds:
                    return self._last_valid_endpoint
                raise StaleRegistryError(
                    f"Registry corrupted and last valid snapshot expired ({elapsed:.1f}s > {self.max_staleness_seconds}s)"
                ) from exc
            if isinstance(exc, InvalidRegistryError):
                raise
            raise InvalidRegistryError(f"Failed to load registry: {exc}") from exc

    def get_preferred_account_id(self) -> Optional[str]:
        return self._preferred_account

    def get_endpoint_for_account(self, account_id: str, now: Optional[float] = None) -> EngineEndpoint:
        current_time = time.time() if now is None else now
        if not self._endpoints:
            self.refresh(now=current_time)

        if self._last_valid_observed_at is not None:
            elapsed = current_time - self._last_valid_observed_at
            if elapsed > self.max_staleness_seconds:
                raise StaleRegistryError(f"Registry snapshot is stale ({elapsed:.1f}s > {self.max_staleness_seconds}s)")

        if account_id not in self._endpoints:
            raise RegistryError(f"Account {account_id!r} not found in registry")
        return self._endpoints[account_id]

    def get_active_endpoint(self, now: Optional[float] = None) -> EngineEndpoint:
        current_time = time.time() if now is None else now
        if not self._endpoints or self._last_valid_observed_at is None:
            return self.refresh(now=current_time)

        elapsed = current_time - self._last_valid_observed_at
        if elapsed > self.max_staleness_seconds:
            raise StaleRegistryError(f"Registry snapshot is stale ({elapsed:.1f}s > {self.max_staleness_seconds}s)")

        if self._preferred_account and self._preferred_account in self._endpoints:
            return self._endpoints[self._preferred_account]
        return next(iter(self._endpoints.values()))

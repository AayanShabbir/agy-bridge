"""Supervisor interface protocol and outcome types (Section 6.3)."""
from __future__ import annotations

from typing import Protocol, Optional
from dataclasses import dataclass


@dataclass
class RestartOutcome:
    success: bool
    new_generation: Optional[str] = None
    skipped: bool = False
    error: Optional[str] = None


class EngineSupervisor(Protocol):
    """Protocol for host supervisors controlling daemon processes."""

    def restart(self, engine_id: str, expected_generation: str, reason: str) -> RestartOutcome:
        """Restart an engine instance if generation still matches."""
        ...

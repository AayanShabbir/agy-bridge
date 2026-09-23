"""Structured error domain for agy-bridge.

Maps protocol, upstream, lifecycle, and admission failures to explicit, typed
exceptions with deterministic HTTP status and OpenAI error code mappings.
"""
from typing import Optional


class AgyBridgeError(Exception):
    """Base exception for all bridge-originated errors."""

    http_status: int = 500
    error_code: str = "internal_error"
    error_type: str = "bridge_error"

    def __init__(self, message: str, *, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Render standard OpenAI-compatible error response payload."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.error_code,
                "param": None,
            }
        }


# ---------------------------------------------------------------------------
# Layer 1: Protocol Errors
# ---------------------------------------------------------------------------

class ProtocolError(AgyBridgeError):
    """Wire framing, gRPC-web, or trailer decoding failure."""

    http_status: int = 502
    error_code: str = "protocol_violation"
    error_type: str = "protocol_error"


class FrameTooLargeError(ProtocolError):
    """A single gRPC-web frame exceeded the maximum configured limit."""

    http_status: int = 413
    error_code: str = "frame_too_large"


class MalformedFrameError(ProtocolError):
    """Frame header or payload could not be parsed according to the codec."""

    http_status: int = 502
    error_code: str = "malformed_frame"


class MalformedTrailerError(ProtocolError):
    """Trailer frame was malformed, truncated, or contained conflicting status."""

    http_status: int = 502
    error_code: str = "malformed_trailer"


class TrajectoryError(AgyBridgeError):
    """Trajectory reduction, snapshot parsing, or step tracking failure."""

    http_status: int = 502
    error_code: str = "trajectory_error"
    error_type: str = "trajectory_error"


class AttributionError(TrajectoryError):
    """Output received from upstream cannot be reliably attributed to the current turn."""

    http_status: int = 502
    error_code: str = "attribution_failure"


# ---------------------------------------------------------------------------
# Layer 2: Upstream & Lifecycle Errors
# ---------------------------------------------------------------------------

class UpstreamError(AgyBridgeError):
    """Errors returned by or caused by the local Antigravity language server."""

    http_status: int = 502
    error_code: str = "upstream_error"
    error_type: str = "upstream_error"


class RateLimitExceeded(UpstreamError):
    """Upstream account quota / rate limit reached."""

    http_status: int = 429
    error_code: str = "rate_limit_exceeded"

    def __init__(self, message: str = "Upstream rate limit exceeded", *, retry_after_s: Optional[int] = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class UpstreamAuthRequired(UpstreamError):
    """Upstream Google account authentication expired or invalid."""

    http_status: int = 503
    error_code: str = "upstream_auth_required"


class UpstreamUnavailable(UpstreamError):
    """Upstream language server process is unreachable or connection refused."""

    http_status: int = 503
    error_code: str = "upstream_unavailable"


class UpstreamTimeout(UpstreamError):
    """Turn execution deadline exceeded."""

    http_status: int = 504
    error_code: str = "upstream_timeout"


class UpstreamEmptyResponse(UpstreamError):
    """Upstream explicitly completed without emitting attributable text or tool calls."""

    http_status: int = 502
    error_code: str = "upstream_empty_response"


class OutcomeUnknownError(UpstreamError):
    """Execution state unknown after bytes may have been submitted; duplicate send forbidden."""

    http_status: int = 502
    error_code: str = "upstream_outcome_unknown"


# ---------------------------------------------------------------------------
# Layer 2 & 3: Admission & Input Errors
# ---------------------------------------------------------------------------

class AdmissionError(AgyBridgeError):
    """Request could not be admitted to an engine or conversation."""

    http_status: int = 503
    error_code: str = "admission_failed"
    error_type: str = "admission_error"


class LeaseConflictError(AdmissionError):
    """A conflicting turn is already in progress on this conversation."""

    http_status: int = 409
    error_code: str = "lease_conflict"


class QueueFullError(AdmissionError):
    """Waiting queue capacity exceeded."""

    http_status: int = 503
    error_code: str = "queue_full"


class QuarantinedError(AdmissionError):
    """Engine or account is currently quarantined pending assessment or recovery."""

    http_status: int = 503
    error_code: str = "engine_quarantined"


class UnsupportedInputError(AgyBridgeError):
    """Request contains features or formats not supported by this profile."""

    http_status: int = 400
    error_code: str = "unsupported_input"
    error_type: str = "invalid_request_error"


class UnsupportedImageError(UnsupportedInputError):
    """Image or multimodal input submitted while vision policy is 'reject'."""

    http_status: int = 400
    error_code: str = "unsupported_image_input"


class ToolContractViolationError(AgyBridgeError):
    """Required tool choice not satisfied or malformed tool schema provided."""

    http_status: int = 400
    error_code: str = "tool_contract_violation"
    error_type: str = "invalid_request_error"

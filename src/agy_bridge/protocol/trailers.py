"""Layer 1: Structural gRPC-web trailer parsing.

Parses trailer frames into typed GrpcTrailers structures without naive substring
matching, with support for percent-encoded error messages and retry headers.
"""
from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, Optional

from agy_bridge.errors import MalformedTrailerError


@dataclass(frozen=True)
class GrpcTrailers:
    """Parsed gRPC status, message, and metadata headers from a trailer frame."""

    status: int
    message: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    is_success: bool = False
    retry_pushback_ms: Optional[int] = None


def parse_trailers(payload: bytes) -> GrpcTrailers:
    """Parse raw trailer bytes into structured GrpcTrailers.

    Raises MalformedTrailerError if grpc-status is missing, non-numeric,
    or contains conflicting status values.
    """
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception as ex:
        raise MalformedTrailerError(f"Trailer payload could not be decoded: {ex}") from ex

    status: Optional[int] = None
    message: Optional[str] = None
    headers: Dict[str, str] = {}
    retry_pushback_ms: Optional[int] = None

    lines = text.replace("\r\n", "\n").split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key, val = line.split(":", 1)
        k = key.strip().lower()
        v = val.strip()

        if k == "grpc-status":
            try:
                numeric_status = int(v)
            except ValueError:
                raise MalformedTrailerError(f"Invalid numeric grpc-status value: '{v}'")

            if status is not None and status != numeric_status:
                raise MalformedTrailerError(
                    f"Contradictory grpc-status in trailer: {status} vs {numeric_status}"
                )
            status = numeric_status
            headers[k] = v
        elif k == "grpc-message":
            decoded_msg = urllib.parse.unquote(v)
            message = decoded_msg
            headers[k] = decoded_msg
        else:
            headers[k] = v
            if k == "grpc-retry-pushback-ms":
                try:
                    retry_pushback_ms = int(v)
                except ValueError:
                    pass

    if status is None:
        raise MalformedTrailerError("Missing grpc-status header in trailer frame.")

    return GrpcTrailers(
        status=status,
        message=message,
        headers=headers,
        is_success=(status == 0),
        retry_pushback_ms=retry_pushback_ms,
    )

"""Layer 1: JSON message codec for gRPC-web framing.

Encodes and decodes JSON structures inside binary gRPC-web frames with
strict validation of UTF-8 encoding and JSON structure.
"""
from __future__ import annotations

import json
from typing import Any

from agy_bridge.errors import MalformedFrameError, ProtocolError
from agy_bridge.protocol.frames import Frame, encode_frame


def decode_json_frame(frame: Frame) -> Any:
    """Decode a gRPC-web data frame into a Python object via UTF-8 JSON parsing.

    Raises ProtocolError if called on a trailer frame, and MalformedFrameError
    if the payload is invalid UTF-8 or malformed JSON.
    """
    if frame.is_trailer:
        raise ProtocolError("Attempted to decode trailer frame as JSON message.")

    try:
        text = frame.payload.decode("utf-8")
    except UnicodeDecodeError as ex:
        raise MalformedFrameError(f"Frame payload is not valid UTF-8: {ex}") from ex

    try:
        return json.loads(text)
    except json.JSONDecodeError as ex:
        raise MalformedFrameError(f"Frame payload is not valid JSON: {ex}") from ex


def encode_json_frame(data: Any, *, trailer: bool = False) -> bytes:
    """Serialize a Python object to compact UTF-8 JSON and wrap in a gRPC-web frame."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return encode_frame(payload, trailer=trailer)

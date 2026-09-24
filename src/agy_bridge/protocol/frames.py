"""Layer 1: Incremental gRPC-web framing and chunked decoder.

Provides strict, bounded encoding and decoding of gRPC-web frames (5-byte header:
1 byte flags + 4 bytes big-endian length) with arbitrary chunk boundary resilience.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

from agy_bridge.errors import FrameTooLargeError, MalformedFrameError

DEFAULT_MAX_FRAME_BYTES: int = 16 * 1024 * 1024  # 16 MiB
HEADER_STRUCT: struct.Struct = struct.Struct(">BI")
HEADER_SIZE: int = HEADER_STRUCT.size  # 5 bytes

FLAG_DATA: int = 0x00
FLAG_TRAILER: int = 0x80
VALID_FLAGS: frozenset[int] = frozenset([FLAG_DATA, FLAG_TRAILER])


@dataclass(frozen=True)
class Frame:
    """Decoded gRPC-web frame."""

    flags: int
    payload: bytes
    is_trailer: bool = False


def encode_frame(payload: bytes, *, trailer: bool = False) -> bytes:
    """Encode binary payload into a 5-byte header prefixed gRPC-web frame."""
    flags = FLAG_TRAILER if trailer else FLAG_DATA
    header = HEADER_STRUCT.pack(flags, len(payload))
    return header + payload


class FrameDecoder:
    """Incremental, streaming decoder for gRPC-web frames.

    Tolerates arbitrary split chunks across headers and payloads, and detects
    unsupported compression, reserved flags, oversized payloads, and truncated EOF.
    """

    def __init__(self, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> None:
        self.max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> List[Frame]:
        """Feed an incoming byte chunk into the decoder, returning all completed frames."""
        if not chunk:
            return []

        self._buffer.extend(chunk)
        frames: List[Frame] = []

        while len(self._buffer) >= HEADER_SIZE:
            flags, length = HEADER_STRUCT.unpack_from(self._buffer, 0)

            # Validate flags: 0x00 (data), 0x80 (trailer). Reject compression / reserved bits.
            if flags not in VALID_FLAGS:
                raise MalformedFrameError(
                    f"Unsupported frame flags: 0x{flags:02x}. Only 0x00 and 0x80 are supported."
                )

            # Reject oversized frames before accumulating full payload into memory
            if length > self.max_frame_bytes:
                raise FrameTooLargeError(
                    f"Frame length {length} exceeds maximum allowed {self.max_frame_bytes} bytes."
                )

            total_frame_len = HEADER_SIZE + length
            if len(self._buffer) < total_frame_len:
                # Need more bytes to complete this frame
                break

            payload = bytes(self._buffer[HEADER_SIZE:total_frame_len])
            del self._buffer[:total_frame_len]

            frames.append(
                Frame(
                    flags=flags,
                    payload=payload,
                    is_trailer=bool(flags & FLAG_TRAILER),
                )
            )

        return frames

    def finish(self) -> None:
        """Signal end of stream. Raises MalformedFrameError if unparsed bytes remain."""
        if len(self._buffer) > 0:
            raise MalformedFrameError(
                f"Stream ended with {len(self._buffer)} unparsed trailing bytes."
            )

"""Tests for Layer 1 gRPC-web framing and chunked decoding."""
import pytest
from agy_bridge.errors import FrameTooLargeError, MalformedFrameError


def test_encode_and_decode_single_data_frame():
    from agy_bridge.protocol.frames import FrameDecoder, encode_frame

    payload = b'{"type": "step", "content": "hello world"}'
    encoded = encode_frame(payload, trailer=False)

    # Frame header is 5 bytes: 1 flag + 4 length
    assert len(encoded) == 5 + len(payload)
    assert encoded[0] == 0x00

    decoder = FrameDecoder()
    frames = decoder.feed(encoded)
    assert len(frames) == 1
    assert frames[0].flags == 0x00
    assert frames[0].payload == payload
    assert frames[0].is_trailer is False
    decoder.finish()


def test_encode_and_decode_trailer_frame():
    from agy_bridge.protocol.frames import FrameDecoder, encode_frame

    payload = b"grpc-status: 0\r\ngrpc-message: OK\r\n"
    encoded = encode_frame(payload, trailer=True)

    assert encoded[0] == 0x80

    decoder = FrameDecoder()
    frames = decoder.feed(encoded)
    assert len(frames) == 1
    assert frames[0].flags == 0x80
    assert frames[0].payload == payload
    assert frames[0].is_trailer is True
    decoder.finish()


def test_decode_arbitrary_byte_by_byte_chunking():
    from agy_bridge.protocol.frames import FrameDecoder, encode_frame

    payloads = [b"first frame", b"second frame payload", b"third"]
    wire = b"".join(encode_frame(p, trailer=(i == 2)) for i, p in enumerate(payloads))

    decoder = FrameDecoder()
    recovered = []
    # Feed 1 byte at a time
    for b in wire:
        recovered.extend(decoder.feed(bytes([b])))
    decoder.finish()

    assert len(recovered) == 3
    assert [f.payload for f in recovered] == payloads
    assert [f.is_trailer for f in recovered] == [False, False, True]


def test_decode_arbitrary_variable_chunks():
    from agy_bridge.protocol.frames import FrameDecoder, encode_frame

    payload = b"X" * 1024
    wire = encode_frame(payload)

    # Chunk sizes that misalign with 5-byte header
    chunk_sizes = [1, 2, 3, 7, 13, 29, 64, 128, 500]
    decoder = FrameDecoder()
    recovered = []
    idx = 0
    c_idx = 0
    while idx < len(wire):
        sz = chunk_sizes[c_idx % len(chunk_sizes)]
        chunk = wire[idx:idx + sz]
        recovered.extend(decoder.feed(chunk))
        idx += sz
        c_idx += 1
    decoder.finish()

    assert len(recovered) == 1
    assert recovered[0].payload == payload


def test_decode_multiple_frames_in_single_chunk():
    from agy_bridge.protocol.frames import FrameDecoder, encode_frame

    payloads = [f"item_{i}".encode("utf-8") for i in range(10)]
    wire = b"".join(encode_frame(p) for p in payloads)

    decoder = FrameDecoder()
    frames = decoder.feed(wire)
    decoder.finish()

    assert len(frames) == 10
    assert [f.payload for f in frames] == payloads


def test_reject_unsupported_flags():
    from agy_bridge.protocol.frames import FrameDecoder

    decoder = FrameDecoder()
    # Flag 0x01 (compressed flag) unsupported
    bad_frame = bytes([0x01, 0x00, 0x00, 0x00, 0x05]) + b"12345"
    with pytest.raises(MalformedFrameError, match="Unsupported frame flags"):
        decoder.feed(bad_frame)


def test_reject_oversized_frame_before_payload_accumulation():
    from agy_bridge.protocol.frames import FrameDecoder

    # Limit to 64 bytes
    decoder = FrameDecoder(max_frame_bytes=64)
    # Header claims 65 bytes payload
    header = bytes([0x00, 0x00, 0x00, 0x00, 65])
    with pytest.raises(FrameTooLargeError, match="exceeds maximum"):
        decoder.feed(header)


def test_finish_raises_on_truncated_header():
    from agy_bridge.protocol.frames import FrameDecoder

    decoder = FrameDecoder()
    # Feed only 3 header bytes (less than 5)
    decoder.feed(bytes([0x00, 0x00, 0x00]))
    with pytest.raises(MalformedFrameError, match="unparsed trailing bytes"):
        decoder.finish()


def test_finish_raises_on_truncated_payload():
    from agy_bridge.protocol.frames import FrameDecoder

    decoder = FrameDecoder()
    # Header claims 10 bytes, but only 4 bytes supplied
    header = bytes([0x00, 0x00, 0x00, 0x00, 10]) + b"1234"
    decoder.feed(header)
    with pytest.raises(MalformedFrameError, match="unparsed trailing bytes"):
        decoder.finish()


def test_default_bound_decodes_observed_large_frame_in_chunks():
    from agy_bridge.protocol.frames import DEFAULT_MAX_FRAME_BYTES, FrameDecoder, HEADER_STRUCT

    size = 14_824_067
    payload = b"x" * size
    wire = HEADER_STRUCT.pack(0, size) + payload
    decoder = FrameDecoder()
    frames = []
    for start in range(0, len(wire), 512 * 1024):
        frames.extend(decoder.feed(wire[start:start + 512 * 1024]))
    decoder.finish()
    assert DEFAULT_MAX_FRAME_BYTES == 16 * 1024 * 1024
    assert len(frames) == 1
    assert frames[0].payload == payload


def test_default_bound_rejects_over_16_mib_at_header():
    from agy_bridge.protocol.frames import FrameDecoder, HEADER_STRUCT

    decoder = FrameDecoder()
    with pytest.raises(FrameTooLargeError, match="exceeds maximum"):
        decoder.feed(HEADER_STRUCT.pack(0, 16 * 1024 * 1024 + 1))


def test_large_frame_still_rejects_unsupported_flags_and_truncated_eof():
    from agy_bridge.protocol.frames import FrameDecoder, HEADER_STRUCT

    with pytest.raises(MalformedFrameError, match="Unsupported frame flags"):
        FrameDecoder().feed(HEADER_STRUCT.pack(1, 14_824_067))
    decoder = FrameDecoder()
    decoder.feed(HEADER_STRUCT.pack(0, 14_824_067) + b"partial")
    with pytest.raises(MalformedFrameError, match="unparsed trailing bytes"):
        decoder.finish()

"""Tests for Layer 1 JSON message framing and decoding."""
import pytest
from agy_bridge.errors import MalformedFrameError, ProtocolError
from agy_bridge.protocol.frames import Frame


def test_encode_and_decode_json_message():
    from agy_bridge.protocol.frames import FrameDecoder
    from agy_bridge.protocol.messages import decode_json_frame, encode_json_frame

    original = {"type": "step", "content": "hello world", "step_index": 42}
    wire = encode_json_frame(original)

    decoder = FrameDecoder()
    frames = decoder.feed(wire)
    assert len(frames) == 1

    decoded = decode_json_frame(frames[0])
    assert decoded == original


def test_decode_rejects_trailer_frame():
    from agy_bridge.protocol.messages import decode_json_frame

    trailer_frame = Frame(flags=0x80, payload=b"grpc-status: 0\r\n", is_trailer=True)
    with pytest.raises(ProtocolError, match="trailer frame as JSON"):
        decode_json_frame(trailer_frame)


def test_decode_rejects_non_utf8():
    from agy_bridge.protocol.messages import decode_json_frame

    bad_frame = Frame(flags=0x00, payload=b"\xff\xfe\x00\x00", is_trailer=False)
    with pytest.raises(MalformedFrameError, match="not valid UTF-8"):
        decode_json_frame(bad_frame)


def test_decode_rejects_malformed_json():
    from agy_bridge.protocol.messages import decode_json_frame

    bad_frame = Frame(flags=0x00, payload=b'{"unterminated": "string', is_trailer=False)
    with pytest.raises(MalformedFrameError, match="not valid JSON"):
        decode_json_frame(bad_frame)


def test_encode_preserves_unicode():
    from agy_bridge.protocol.frames import FrameDecoder
    from agy_bridge.protocol.messages import decode_json_frame, encode_json_frame

    original = {"text": "Hello 🚀 世界 café"}
    wire = encode_json_frame(original)

    decoder = FrameDecoder()
    frames = decoder.feed(wire)
    decoded = decode_json_frame(frames[0])
    assert decoded["text"] == "Hello 🚀 世界 café"

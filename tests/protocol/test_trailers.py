"""Tests for Layer 1 gRPC-web trailer parsing."""
import pytest
from agy_bridge.errors import MalformedTrailerError


def test_parse_trailers_success_clean():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: 0\r\ngrpc-message: OK\r\n"
    trailers = parse_trailers(payload)

    assert trailers.status == 0
    assert trailers.message == "OK"
    assert trailers.is_success is True
    assert trailers.headers["grpc-status"] == "0"
    assert trailers.headers["grpc-message"] == "OK"


def test_parse_trailers_percent_encoded_message():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: 14\r\ngrpc-message: Resource%20exhausted%3A%20quota%20exceeded\r\n"
    trailers = parse_trailers(payload)

    assert trailers.status == 14
    assert trailers.message == "Resource exhausted: quota exceeded"
    assert trailers.is_success is False


def test_parse_trailers_mixed_case_and_whitespace():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"GRPC-STATUS:  0  \ngrpc-Message:  all good  \n"
    trailers = parse_trailers(payload)

    assert trailers.status == 0
    assert trailers.message == "all good"
    assert trailers.is_success is True


def test_parse_trailers_preserves_retry_pushback():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: 8\r\ngrpc-retry-pushback-ms: 5000\r\n"
    trailers = parse_trailers(payload)

    assert trailers.status == 8
    assert trailers.headers.get("grpc-retry-pushback-ms") == "5000"
    assert trailers.retry_pushback_ms == 5000


def test_reject_missing_grpc_status():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"content-type: application/grpc-web+proto\r\n"
    with pytest.raises(MalformedTrailerError, match="Missing grpc-status"):
        parse_trailers(payload)


def test_reject_non_numeric_grpc_status():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: OK\r\n"
    with pytest.raises(MalformedTrailerError, match="Invalid numeric grpc-status"):
        parse_trailers(payload)


def test_reject_contradictory_grpc_status():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: 0\r\ngrpc-status: 14\r\n"
    with pytest.raises(MalformedTrailerError, match="Contradictory grpc-status"):
        parse_trailers(payload)


def test_allow_duplicate_identical_grpc_status():
    from agy_bridge.protocol.trailers import parse_trailers

    payload = b"grpc-status: 0\r\ngrpc-status: 0\r\n"
    trailers = parse_trailers(payload)
    assert trailers.status == 0


def test_never_fooled_by_substring():
    from agy_bridge.protocol.trailers import parse_trailers

    # A custom header contains "grpc-status: 0", but the real grpc-status is 13 (INTERNAL)
    payload = b"x-custom-note: contains grpc-status: 0\r\ngrpc-status: 13\r\n"
    trailers = parse_trailers(payload)

    assert trailers.status == 13
    assert trailers.is_success is False

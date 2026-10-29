"""Tests for the HTTP service routing and the CLI."""

from __future__ import annotations

from bpe_tokenizer_library.cli import main
from bpe_tokenizer_library.service import TokenizerService, handle

_SERVICE = TokenizerService.trained(vocab_size=400)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and payload["vocab_size"] >= 256
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_encode_decode_round_trip_over_http() -> None:
    status, enc = handle("POST", "/encode", {"text": "the quick dog"}, _SERVICE)
    assert status == 200 and enc["count"] == len(enc["ids"])
    status, dec = handle("POST", "/decode", {"ids": enc["ids"]}, _SERVICE)
    assert status == 200 and dec["text"] == "the quick dog"


def test_stats_route() -> None:
    status, payload = handle("POST", "/stats", {"text": "the quick brown fox"}, _SERVICE)
    assert status == 200
    assert payload["bytes_per_token"] >= 1.0


def test_encode_validation() -> None:
    status, _ = handle("POST", "/encode", {"text": 123}, _SERVICE)
    assert status == 400
    status, _ = handle("POST", "/encode", {"text": "x" * 10_001}, _SERVICE)
    assert status == 400


def test_decode_validation() -> None:
    for body in ({"ids": "nope"}, {"ids": [-1]}, {"ids": [10**9]}, {"ids": [True]}):
        status, _ = handle("POST", "/decode", body, _SERVICE)
        assert status == 400


def test_stats_requires_nonempty_text() -> None:
    status, _ = handle("POST", "/stats", {"text": ""}, _SERVICE)
    assert status == 400


def test_unknown_route_and_metrics() -> None:
    status, _ = handle("GET", "/nope", {}, _SERVICE)
    assert status == 404
    handle("POST", "/encode", {"text": "hello"}, _SERVICE)
    assert "service_requests_total" in _SERVICE.metrics.render_prometheus()


def test_cli_train_and_encode(capsys) -> None:  # type: ignore[no-untyped-def]
    main(["--vocab-size", "400", "train"])
    main(["--vocab-size", "400", "encode", "the quick dog"])
    out = capsys.readouterr().out
    assert "bytes/token" in out
    assert "roundtrip: True" in out

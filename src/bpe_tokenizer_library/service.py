"""A dependency-free HTTP service exposing the trained tokenizer.

The service trains a tokenizer once at startup over the built-in corpus, then
encodes and decodes text. The routing logic is a pure function tested without a
socket.

Endpoints
---------
``GET  /health``   liveness and vocabulary size.
``GET  /ready``    readiness once the tokenizer is trained.
``GET  /metrics``  per-route request, error, and latency metrics (Prometheus text).
``POST /encode``   body ``{"text": str}``; returns token ids and glyph tokens.
``POST /decode``   body ``{"ids": [int]}``; returns the decoded text.
``POST /stats``    body ``{"text": str}``; returns byte length, token count, bytes per token.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from bpe_tokenizer_library.data import sample_corpus
from bpe_tokenizer_library.evaluation import compression
from bpe_tokenizer_library.observability import ServiceMetrics
from bpe_tokenizer_library.tokenizer import BPETokenizer

JsonObject = dict[str, Any]
_MAX_TEXT = 10_000
_MAX_IDS = 20_000


@dataclass(frozen=True)
class TokenizerService:
    """Holds the trained tokenizer and answers encode and decode requests."""

    tokenizer: BPETokenizer
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, vocab_size: int = 600) -> TokenizerService:
        return cls(tokenizer=BPETokenizer.train(sample_corpus(), vocab_size=vocab_size))

    def encode(self, text: str) -> JsonObject:
        ids = self.tokenizer.encode(text)
        return {
            "ids": ids,
            "tokens": [self.tokenizer._id_to_token[i] for i in ids],
            "count": len(ids),
        }

    def decode(self, ids: list[int]) -> JsonObject:
        return {"text": self.tokenizer.decode(ids)}

    def stats(self, text: str) -> JsonObject:
        report = compression(self.tokenizer, text)
        return {
            "byte_length": report.byte_length,
            "token_count": report.token_count,
            "bytes_per_token": round(report.bytes_per_token, 4),
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: TokenizerService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "vocab_size": service.tokenizer.vocab_size}
    if method == "POST" and path == "/encode":
        text = body.get("text")
        if not isinstance(text, str) or len(text) > _MAX_TEXT:
            return 400, {"error": f"'text' must be a string of at most {_MAX_TEXT} characters"}
        return 200, service.encode(text)
    if method == "POST" and path == "/stats":
        text = body.get("text")
        if not isinstance(text, str) or not text or len(text) > _MAX_TEXT:
            return 400, {
                "error": f"'text' must be a non-empty string of at most {_MAX_TEXT} characters"
            }
        return 200, service.stats(text)
    if method == "POST" and path == "/decode":
        ids = body.get("ids")
        vocab = service.tokenizer.vocab_size
        if not isinstance(ids, list) or len(ids) > _MAX_IDS:
            return 400, {"error": f"'ids' must be a list of at most {_MAX_IDS} integers"}
        for i in ids:
            if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= vocab:
                return 400, {"error": f"token ids must be integers in 0..{vocab - 1}"}
        return 200, service.decode(list(ids))
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: TokenizerService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: TokenizerService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": TokenizerService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"bpe-tokenizer-library service listening on http://{host}:{port}")
    server.serve_forever()

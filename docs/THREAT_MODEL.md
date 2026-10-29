# Threat Model - BPE Tokenizer Library

## Assets
- The HTTP service's availability and the integrity of its responses.

## Trust boundaries
- The only external input is the JSON body and path of a request. The corpus and trained merges
  are in-process and trusted.

## Entry points and mitigations

### `POST /encode` and `POST /stats`
- **Malformed JSON** is rejected with 400 before routing.
- **Validation:** the text must be a string within a length bound; `/stats` additionally requires
  it to be non-empty. Oversized or wrong-type input returns 400 with a message.

### `POST /decode`
- Token ids must be a bounded-length list of integers within the vocabulary; booleans and
  out-of-range values are rejected explicitly.

### `GET` routes
- `/health`, `/ready`, and `/metrics` take no user input and return fixed-shape payloads.
  `/metrics` exposes only aggregate counters.

## Non-threats (by construction)
- **No secrets:** the service holds no credentials and makes no outbound calls.
- **No persistence:** state is the in-memory trained tokenizer; nothing is written on the request
  path.
- **No code execution:** inputs are text and integers; no user-supplied string is evaluated.

## Supply chain
- Zero runtime dependencies beyond the standard library. Development tooling is pinned via
  `uv.lock`. CodeQL and dependency scanning run in CI.

## Residual risk
- Very large inputs are bounded by explicit length limits, so a single request cannot exhaust
  memory; a production deployment would add a rate limiter behind a reverse proxy.

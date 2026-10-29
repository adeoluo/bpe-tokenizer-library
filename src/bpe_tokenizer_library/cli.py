"""Command line interface for the byte-level BPE tokenizer."""

from __future__ import annotations

import argparse

from bpe_tokenizer_library.data import held_out_text, sample_corpus
from bpe_tokenizer_library.evaluation import compression, encode_throughput, is_lossless
from bpe_tokenizer_library.tokenizer import BPETokenizer


def _train(vocab_size: int) -> BPETokenizer:
    return BPETokenizer.train(sample_corpus(), vocab_size=vocab_size)


def _cmd_train(args: argparse.Namespace) -> None:
    tok = _train(args.vocab_size)
    text = held_out_text()
    report = compression(tok, text)
    tp = encode_throughput(tok, text, repeats=100)
    print(f"Trained a byte-level BPE tokenizer: vocab {tok.vocab_size}, {len(tok.merges)} merges.")
    print(
        f"  held-out compression: {report.bytes_per_token:.2f} bytes/token "
        f"({report.token_count} tokens)"
    )
    print(f"  lossless on the round-trip battery: {is_lossless(tok)}")
    print(f"  encode throughput: {tp.bytes_per_second:,.0f} bytes/sec")


def _cmd_encode(args: argparse.Namespace) -> None:
    tok = _train(args.vocab_size)
    ids = tok.encode(args.text)
    print("ids:", ids)
    print("tokens:", tok.tokens(args.text))
    print("roundtrip:", tok.decode(ids) == args.text)


def _cmd_serve(args: argparse.Namespace) -> None:
    from bpe_tokenizer_library.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bpe-tokenizer-library", description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=600)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="train and report compression, losslessness, and throughput")

    encode = sub.add_parser("encode", help="encode a string and show the round-trip")
    encode.add_argument("text")

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {"train": _cmd_train, "encode": _cmd_encode, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()

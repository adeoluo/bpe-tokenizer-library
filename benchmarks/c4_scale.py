"""Train on real web text (C4) at GPT-2's vocabulary size and compare with tiktoken.

1. Trainer scaling: the heap-based trainer against the previous scan-based trainer
   (an older build passed with --old-ext) on growing slices of C4, checking that both
   learn the same merges.
2. A 50,257-token vocabulary (the size of GPT-2's) trained on --train-mb of C4.
3. On held-out C4 text: compression (bytes per token) and encode throughput, single
   threaded and batched, against tiktoken's GPT-2 encoding.

    uv sync
    uv run --no-sync --with tiktoken python benchmarks/c4_scale.py \\
        --c4 c4-train.00000-of-01024.json.gz --heldout c4-train.00001-of-01024.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def c4_slice(path: Path, megabytes: float) -> list[str]:
    limit = int(megabytes * 1_000_000)
    out: list[str] = []
    size = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line)["text"]
            out.append(text)
            size += len(text.encode())
            if size >= limit:
                break
    return out


TRAIN_SNIPPET = """
import json, sys, time
from bpe_tokenizer_library.backend import rust_train
texts = json.load(open(sys.argv[1]))
start = time.perf_counter()
merges = rust_train(texts, int(sys.argv[2]))
print(json.dumps({"seconds": time.perf_counter() - start, "merges": merges}))
"""


def train_subprocess(texts_path: Path, vocab: int, ext_dir: str | None) -> dict[str, object]:
    env = dict(os.environ)
    if ext_dir:
        env["PYTHONPATH"] = ext_dir
    out = subprocess.run(
        [sys.executable, "-c", TRAIN_SNIPPET, str(texts_path), str(vocab)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(out.stdout)


def best_of(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    import tiktoken

    from bpe_tokenizer_library.tokenizer import BPETokenizer

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--c4", required=True, type=Path)
    ap.add_argument("--heldout", required=True, type=Path)
    ap.add_argument("--old-ext", help="directory holding the previous bpe_tokenizer_rs build")
    ap.add_argument("--scaling-mb", default="1,4,16")
    ap.add_argument("--scaling-vocab", type=int, default=8192)
    ap.add_argument("--train-mb", type=float, default=100)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--heldout-mb", type=float, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "c4_scale.json")
    args = ap.parse_args()
    work = ROOT / ".bench-work"
    work.mkdir(exist_ok=True)
    report: dict[str, object] = {
        "machine": f"{platform.machine()} {platform.system()}",
        "cpus": os.cpu_count(),
    }

    scaling = []
    print(f"Trainer scaling, vocabulary {args.scaling_vocab:,}:\n")
    print("| C4 text | Previous trainer | Heap trainer | Speedup | Same merges |")
    print("|---:|---:|---:|---:|:---:|")
    for mb in [float(m) for m in args.scaling_mb.split(",")]:
        path = work / f"train_{mb:g}mb.json"
        path.write_text(json.dumps(c4_slice(args.c4, mb)))
        new = train_subprocess(path, args.scaling_vocab, None)
        old = train_subprocess(path, args.scaling_vocab, args.old_ext) if args.old_ext else None
        same = old is None or old["merges"] == new["merges"]
        row = {
            "mb": mb,
            "new_s": new["seconds"],
            "old_s": old["seconds"] if old else None,
            "same": same,
        }
        scaling.append(row)
        old_cell = f"{old['seconds']:.1f} s" if old else "n/a"
        speed = f"{old['seconds'] / new['seconds']:.0f}x" if old else "n/a"
        print(
            f"| {mb:g} MB | {old_cell} | {new['seconds']:.1f} s | {speed} | "
            f"{'yes' if same else 'NO'} |"
        )
    report["scaling"] = scaling

    path = work / "train_full.json"
    texts = c4_slice(args.c4, args.train_mb)
    path.write_text(json.dumps(texts))
    full = train_subprocess(path, args.vocab, None)
    merges = [tuple(m) for m in full["merges"]]
    tok = BPETokenizer.from_merges(merges, backend="rust")
    report["full"] = {"mb": args.train_mb, "vocab": tok.vocab_size, "seconds": full["seconds"]}
    print(
        f"\nTrained a {tok.vocab_size:,}-token vocabulary on {args.train_mb:g} MB of C4 "
        f"in {full['seconds']:.1f} s"
    )

    held = c4_slice(args.heldout, args.heldout_mb)
    total_bytes = sum(len(t.encode()) for t in held)
    gpt2 = tiktoken.get_encoding("gpt2")
    ours_tokens = sum(len(ids) for ids in tok.encode_batch(held))
    gpt2_tokens = sum(
        len(ids) for ids in gpt2.encode_ordinary_batch(held, num_threads=os.cpu_count())
    )
    assert all(tok.decode(tok.encode(t)) == t for t in held[:2000]), "round trip failed"

    ours_single = best_of(lambda: [tok.encode(t) for t in held])
    gpt2_single = best_of(lambda: [gpt2.encode_ordinary(t) for t in held])
    ours_batch = best_of(lambda: tok.encode_batch(held))
    gpt2_batch = best_of(lambda: gpt2.encode_ordinary_batch(held, num_threads=os.cpu_count()))
    mb = total_bytes / 1e6
    rows = [
        {
            "tokenizer": f"bpe-tokenizer-library, trained on C4 ({tok.vocab_size:,} tokens)",
            "bytes_per_token": total_bytes / ours_tokens,
            "single_mb_s": mb / ours_single,
            "batch_mb_s": mb / ours_batch,
        },
        {
            "tokenizer": "tiktoken gpt2 (50,257 tokens)",
            "bytes_per_token": total_bytes / gpt2_tokens,
            "single_mb_s": mb / gpt2_single,
            "batch_mb_s": mb / gpt2_batch,
        },
    ]
    report["heldout"] = {"mb": mb, "documents": len(held), "rows": rows}
    print(f"\nHeld-out C4 ({mb:.1f} MB, {len(held):,} documents):\n")
    print(
        "| Tokenizer | Bytes per token | Encode, 1 thread | "
        f"Encode, batch ({os.cpu_count()} threads) |"
    )
    print("|---|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['tokenizer']} | {r['bytes_per_token']:.2f} | {r['single_mb_s']:.1f} MB/s | "
            f"{r['batch_mb_s']:.1f} MB/s |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

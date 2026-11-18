"""Compares logits written by llama_logits.cpp with the reference goldens.

    python -I compare.py --goldens goldens.gguf --logits llama.bin [--golden logits]

Prints the largest absolute difference, the relative error per position (largest |diff| over
the position's largest |logit|), how many positions have the same top token, and the first
positions where the top tokens differ.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5next"))
import gguf_io  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--logits", required=True, help="float32 [n_tokens][n_vocab] from llama_logits")
    ap.add_argument("--golden", default="logits", help="golden tensor name (default logits)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ref = gguf_io.GGUFReader(args.goldens).tensor_f32(args.golden)
    n, v = ref.shape
    got = np.fromfile(args.logits, dtype=np.float32)
    if got.size != n * v:
        raise SystemExit(f"{args.logits}: {got.size} values, expected {n} x {v}")
    got = got.reshape(n, v)
    diff = np.abs(got - ref)
    rel = diff.max(1) / np.abs(ref).max(1)
    same = got.argmax(1) == ref.argmax(1)
    first_bad = np.nonzero(~same)[0][:8].tolist()
    print(f"{args.label:24s} max|diff| {diff.max():.3e}  rel max {rel.max():.3e} median {np.median(rel):.3e}  "
          f"top-1 {same.sum()}/{n}  first mismatches {first_bad}")


if __name__ == "__main__":
    main()

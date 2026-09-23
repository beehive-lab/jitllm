#!/usr/bin/env python3
"""
report_perf.py — reformat benchmark_backends.sh results.

Reads all per-rep metrics JSON in a results dir and emits a table organised by
model-family / size / quantization / backend / configuration, with decode and
prefill throughput (mean +/- std across reps) and one-time JIT.

Usage:
  scripts/report_perf.py [results_dir]   # default: newest perf-results/*
  scripts/report_perf.py --csv out.csv --md out.md [results_dir]
"""
import json, sys, glob, os, re, statistics, argparse

BACKENDS = ["opencl", "cuda"]
QUANTS = ["Q8_0", "F16"]
# Longest first so "batch-prefill-decode-cuda-graphs" matches before "prefill-decode".
CONFIGS = [
    "batch-prefill-decode-cuda-graphs",
    "prefill-decode-cuda-graphs",
    "batch-prefill-decode",
    "prefill-decode",
    "standard",
]
CONFIG_ORDER = {c: i for i, c in enumerate(reversed(CONFIGS))}

# model name -> (family, size, size_rank)
MODELS = {
    "Qwen3-0.6B":              ("Qwen3",      "0.6B", 0.6),
    "Llama-3.2-1B-Instruct":   ("Llama-3.2",  "1B",   1.0),
    "Granite-4.0-1B":          ("Granite-4.0","1B",   1.0),
    "Qwen2.5-1.5B-Instruct":   ("Qwen2.5",    "1.5B", 1.5),
    "Granite-3.2-2B-Instruct": ("Granite-3.2","2B",   2.0),
    "Phi-3-mini-4k-instruct":  ("Phi-3-mini", "3.8B", 3.8),
    "Mistral-7B-Instruct-v0.3":("Mistral",    "7B",   7.0),
}


def parse_tag(tag):
    """metrics tag 'backend-model-quant-config' -> (backend, model, quant, config)."""
    backend = next((b for b in BACKENDS if tag.startswith(b + "-")), None)
    if not backend:
        return None
    rest = tag[len(backend) + 1:]
    config = next((c for c in CONFIGS if rest.endswith("-" + c)), None)
    if not config:
        return None
    rest = rest[: -(len(config) + 1)]
    quant = next((q for q in QUANTS if rest.endswith("-" + q)), None)
    if not quant:
        return None
    model = rest[: -(len(quant) + 1)]
    return backend, model, quant, config


def collect(results_dir):
    rows = {}  # (backend,model,quant,config) -> dict of lists
    for f in glob.glob(os.path.join(results_dir, "metrics-*-rep*.json")):
        base = os.path.basename(f)
        m = re.match(r"metrics-(.+)-rep\d+\.json$", base)
        if not m:
            continue
        parsed = parse_tag(m.group(1))
        if not parsed:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        d = rows.setdefault(parsed, {"eval": [], "prefill": [], "jit": []})
        if isinstance(data.get("eval_rate"), (int, float)):
            d["eval"].append(data["eval_rate"])
        if isinstance(data.get("prompt_eval_rate"), (int, float)):
            d["prefill"].append(data["prompt_eval_rate"])
        j = data.get("tornadovm", {}).get("jit_duration")
        if isinstance(j, (int, float)):
            d["jit"].append(j / 1e6)
    return rows


def ms(x):
    if not x:
        return "-"
    return f"{statistics.mean(x):.2f}" + (f" ± {statistics.pstdev(x):.2f}" if len(x) > 1 else "")


def build_table(rows):
    out = []
    for (backend, model, quant, config), d in rows.items():
        fam, size, rank = MODELS.get(model, (model, "?", 99))
        out.append({
            "family": fam, "size": size, "rank": rank, "quant": quant,
            "backend": backend, "config": config,
            "decode": ms(d["eval"]), "prefill": ms(d["prefill"]),
            "jit_ms": f"{statistics.mean(d['jit']):.0f}" if d["jit"] else "-",
            "decode_mean": statistics.mean(d["eval"]) if d["eval"] else 0.0,
        })
    out.sort(key=lambda r: (r["rank"], r["family"], r["quant"],
                            BACKENDS.index(r["backend"]) if r["backend"] in BACKENDS else 9,
                            CONFIG_ORDER.get(r["config"], 9)))
    return out


def build_compare(rows, config="standard"):
    """Pivot to one row per family/size/quant: decode mean per backend + CUDA
    speedup vs OpenCL (for the given config)."""
    piv = {}  # (rank,family,size,quant) -> {backend: decode_mean}
    for (backend, model, quant, cfg), d in rows.items():
        if cfg != config or not d["eval"]:
            continue
        fam, size, rank = MODELS.get(model, (model, "?", 99))
        piv.setdefault((rank, fam, size, quant), {})[backend] = statistics.mean(d["eval"])
    out = []
    for (rank, fam, size, quant), be in sorted(piv.items()):
        o, c = be.get("opencl"), be.get("cuda")
        def spd(ref):
            if c is None or not ref:
                return "-"
            pct = (c / ref - 1) * 100
            return f"{c/ref:.2f}x ({pct:+.0f}%)"
        def n(x):
            return f"{x:.1f}" if x is not None else "-"
        out.append([fam, size, quant, n(o), n(c), spd(o)])
    return out


def print_compare(rows, config="standard"):
    data = build_compare(rows, config)
    hdr = ["family", "size", "quant", "opencl", "cuda", "CUDA vs OpenCL"]
    widths = [max(len(hdr[i]), max((len(str(r[i])) for r in data), default=0)) for i in range(len(hdr))]
    fr = lambda v: "  ".join(str(x).ljust(widths[i]) for i, x in enumerate(v))
    print(f"\n## CUDA vs OpenCL — decode tok/s, config='{config}'  (>1.00x = CUDA faster)\n")
    print(fr(hdr)); print(fr(["-" * w for w in widths]))
    for r in data:
        print(fr(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    ap.add_argument("--compare", action="store_true",
                    help="Also print a CUDA-vs-OpenCL speedup table (standard config).")
    args = ap.parse_args()

    rd = args.results_dir
    if not rd:
        cands = sorted(glob.glob(os.path.expanduser("~/jitllm/perf-results/*")), reverse=True)
        rd = cands[0] if cands else "."
    raw = collect(rd)
    rows = build_table(raw)
    if not rows:
        print(f"No metrics found in {rd}", file=sys.stderr)
        sys.exit(1)

    cols = ["family", "size", "quant", "backend", "config", "decode", "prefill", "jit_ms"]
    hdr = ["family", "size", "quant", "backend", "configuration", "decode tok/s", "prefill tok/s", "jit ms (1-time)"]
    widths = [max(len(hdr[i]), max(len(str(r[cols[i]])) for r in rows)) for i in range(len(cols))]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    print(f"# Results: {rd}\n")
    print(fmt_row(hdr))
    print(fmt_row(["-" * w for w in widths]))
    last = None
    for r in rows:
        key = (r["family"], r["size"], r["quant"])
        if last is not None and key != last:
            print()  # blank line between model/quant groups
        print(fmt_row([r[c] for c in cols]))
        last = key

    if args.compare:
        print_compare(raw, "standard")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for r in rows:
                w.writerow([r[c] for c in cols])
        print(f"\nCSV  -> {args.csv}", file=sys.stderr)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("| " + " | ".join(hdr) + " |\n")
            fh.write("|" + "|".join("---" for _ in hdr) + "|\n")
            for r in rows:
                fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        print(f"MD   -> {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()

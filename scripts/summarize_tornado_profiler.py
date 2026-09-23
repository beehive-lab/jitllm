#!/usr/bin/env python3
#
# summarize_tornado_profiler.py
#
# Summarizes TornadoVM's own profiler output (concatenated JSON objects, one per
# task-graph execution) into a phase/task kernel-time breakdown. Produced when
# `jitllm` is run with `--profiler --profiler-dump-dir <dir>`; point this
# script at the resulting profiler log file.
#
# Usage:
#   scripts/summarize_tornado_profiler.py <profiler_log> [--top N]
#
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_concatenated_json(path):
    text = Path(path).read_text(errors="replace")
    dec = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        yield obj
        idx = end


def ns(value):
    try:
        return int(value)
    except Exception:
        return 0


def task_phase(graph_name):
    if graph_name.startswith("batchPrefillLayer_") or graph_name == "prefillActivation":
        return "prefill"
    if graph_name.startswith("layer_") or graph_name in {"activationUpdate", "decodeActivation", "logits"}:
        return "decode"
    return "other"


def task_short_name(task_name):
    parts = task_name.split(".")
    return parts[-1] if parts else task_name


def summarize(path):
    by_task = defaultdict(lambda: {"kernel_ns": 0, "count": 0, "methods": defaultdict(int)})
    by_phase = defaultdict(int)
    by_graph = defaultdict(int)

    for obj in load_concatenated_json(path):
        if not isinstance(obj, dict):
            continue
        for graph_name, graph in obj.items():
            if not isinstance(graph, dict):
                continue
            phase = task_phase(graph_name)
            graph_kernel = ns(graph.get("TOTAL_KERNEL_TIME"))
            by_phase[phase] += graph_kernel
            by_graph[graph_name] += graph_kernel
            for key, val in graph.items():
                if not isinstance(val, dict) or "TASK_KERNEL_TIME" not in val:
                    continue
                short = task_short_name(key)
                item = by_task[(phase, short)]
                item["kernel_ns"] += ns(val.get("TASK_KERNEL_TIME"))
                item["count"] += 1
                method = val.get("METHOD")
                if method:
                    item["methods"][method] += ns(val.get("TASK_KERNEL_TIME"))

    return by_task, by_phase, by_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiler_log")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    by_task, by_phase, _ = summarize(args.profiler_log)
    total = sum(v["kernel_ns"] for v in by_task.values())

    print(f"# {args.profiler_log}")
    print()
    print("| phase | kernel ms | pct |")
    print("|---|---:|---:|")
    for phase, t in sorted(by_phase.items(), key=lambda kv: kv[1], reverse=True):
        pct = (100 * t / total) if total else 0.0
        print(f"| {phase} | {t / 1e6:.3f} | {pct:.1f}% |")

    print()
    print("| phase | task | kernel ms | pct | calls | top method |")
    print("|---|---|---:|---:|---:|---|")
    rows = sorted(by_task.items(), key=lambda kv: kv[1]["kernel_ns"], reverse=True)
    for (phase, task), data in rows[: args.top]:
        t = data["kernel_ns"]
        pct = (100 * t / total) if total else 0.0
        method = "-"
        if data["methods"]:
            method = max(data["methods"].items(), key=lambda kv: kv[1])[0]
        print(f"| {phase} | {task} | {t / 1e6:.3f} | {pct:.1f}% | {data['count']} | {method} |")


if __name__ == "__main__":
    main()

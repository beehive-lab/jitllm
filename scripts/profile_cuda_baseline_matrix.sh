#!/usr/bin/env bash
# No `set -u`: sdkman-init.sh references optional unset shell vars.

source "$HOME/.sdkman/bin/sdkman-init.sh" >/dev/null 2>&1
sdk use java 21.0.2-open >/dev/null
source "$HOME/TornadoVM/setvars.sh" >/dev/null

export JITLLM_ROOT=/home/orion/jitllm
cd "$JITLLM_ROOT" || exit 1

RESULTS_DIR="${RESULTS_DIR:-$JITLLM_ROOT/perf-results/profile-cuda-baseline-$(date +%Y%m%d-%H%M%S)}"
PROMPT="write a matmul in java"
MAX_TOKENS=2048
GPU_MEMORY=20GB

mkdir -p "$RESULTS_DIR"

printf "results_dir=%s\n" "$RESULTS_DIR" | tee "$RESULTS_DIR/run.log"
printf "java=%s\n" "$(java -version 2>&1 | head -1)" | tee -a "$RESULTS_DIR/run.log"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tee -a "$RESULTS_DIR/run.log"
printf "fp16_kv=false\n" | tee -a "$RESULTS_DIR/run.log"
printf "ignore_eos=true\n" | tee -a "$RESULTS_DIR/run.log"

printf "model\tconfig\tstatus\teval_rate\tprompt_eval_rate\ttotal_ms\tmetrics\tprofiler\tlog\n" > "$RESULTS_DIR/summary.tsv"

models=(
  "Llama-3.2-1B-Instruct|/home/orion/LLMModels/Llama-3.2-1B-Instruct-F16.gguf"
  "Llama-3.2-3B-Instruct|/home/orion/LLMModels/Llama-3.2-3B-Instruct-F16.gguf"
  "Qwen3-0.6B|/home/orion/LLMModels/Qwen3-0.6B-f16.gguf"
  "Qwen3-1.7B|/home/orion/LLMModels/Qwen3-1.7B-f16.gguf"
  "Qwen3-4B|/home/orion/LLMModels/Qwen3-4B-f16.gguf"
)

configs=(
  "standard|"
  "batch-prefill-decode|--with-prefill-decode --batch-prefill-size 32"
)

for entry in "${models[@]}"; do
  IFS="|" read -r model path <<< "$entry"
  for cfg_entry in "${configs[@]}"; do
    IFS="|" read -r config flags <<< "$cfg_entry"
    tag="cuda-${model}-F16-${config}-baseline-profile"
    metrics="$RESULTS_DIR/metrics-${tag}.json"
    profiler="$RESULTS_DIR/profiler-${tag}.json"
    logfile="$RESULTS_DIR/run-${tag}.log"
    summary="$RESULTS_DIR/profiler-summary-${tag}.md"

    printf "[%s] profiling %s / %s / max_tokens=%s / gpu_memory=%s\n" \
      "$(date +%H:%M:%S)" "$model" "$config" "$MAX_TOKENS" "$GPU_MEMORY" | tee -a "$RESULTS_DIR/run.log"

    export JAVA_TOOL_OPTIONS="-Djitllm.bench.ignoreEos=true -Djitllm.metrics.format=json -Djitllm.metrics.output=file -Djitllm.metrics.file=$metrics"
    start_ms=$(date +%s%3N)
    # shellcheck disable=SC2086
    ./jitllm --gpu --cuda \
      --model "$path" \
      --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" \
      --seed 1 \
      --gpu-memory "$GPU_MEMORY" \
      --profiler \
      --profiler-dump-dir "$profiler" \
      $flags > "$logfile" 2>&1
    rc=$?
    end_ms=$(date +%s%3N)
    unset JAVA_TOOL_OPTIONS

    total_ms=$((end_ms - start_ms))
    if [ "$rc" -eq 0 ]; then
      status=OK
      scripts/summarize_tornado_profiler.py "$profiler" > "$summary" 2>> "$RESULTS_DIR/run.log"
    else
      status="FAIL:$rc"
    fi

    rates=$(python3 - "$metrics" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
    print(f"{data.get('eval_rate', '-')}\t{data.get('prompt_eval_rate', '-')}")
except Exception:
    print("-\t-")
PY
)

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$model" "$config" "$status" "$rates" "$total_ms" "$metrics" "$profiler" "$logfile" >> "$RESULTS_DIR/summary.tsv"

    printf "[%s] done %s / %s => %s (%sms)\n" \
      "$(date +%H:%M:%S)" "$model" "$config" "$status" "$total_ms" | tee -a "$RESULTS_DIR/run.log"
  done
done

printf "RESULTS_DIR=%s\n" "$RESULTS_DIR"

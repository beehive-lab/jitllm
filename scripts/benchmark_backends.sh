#!/usr/bin/env bash
#
# benchmark_backends.sh
#
# Measures jitllm inference performance across the three TornadoVM GPU
# backends (OpenCL, PTX, CUDA) over the full model/quantization/configuration
# matrix from .github/workflows/build-and-run.yml.
#
# For each backend it:
#   1. Rebuilds TornadoVM for that single backend only.
#   2. Sources the freshly generated TornadoVM environment.
#   3. Runs every applicable case (WARMUP discarded + REPS measured).
# Finally it prints a comparison table (eval / prompt-eval tok/s, mean ± std,
# plus one-time JIT).
#
# The cuda-graphs configurations are PTX-only (mirrors the workflow).
#
# Usage:
#   scripts/benchmark_backends.sh [backend ...]
#   BACKENDS="cuda ptx" REPS=3 scripts/benchmark_backends.sh
#
# Note: no `set -u` — sdkman-init.sh references unbound vars (ZSH_VERSION) when sourced.
set -o pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (override via environment)
# ──────────────────────────────────────────────────────────────────────────────
JAVA_SDK="${JAVA_SDK:-21.0.2-open}"
TORNADO_ROOT="${TORNADO_ROOT:-$HOME/TornadoVM}"
JITLLM_ROOT_DIR="${JITLLM_ROOT_DIR:-$HOME/jitllm}"
MODELS_DIR="${MODELS_DIR:-/opt/models}"

PROMPT="${PROMPT:-Write a long, detailed adventure story about a young explorer discovering a hidden ancient city in the jungle. Include vivid descriptions.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GPU_MEMORY="${GPU_MEMORY:-14GB}"

# Statistics: measured reps (reported as mean ± std) plus discarded warmups.
REPS="${REPS:-3}"
WARMUP="${WARMUP:-1}"

# Backends to test: CLI args > $BACKENDS env > default.
if [ "$#" -gt 0 ]; then
    BACKEND_LIST=("$@")
else
    read -r -a BACKEND_LIST <<< "${BACKENDS:-opencl ptx cuda}"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$JITLLM_ROOT_DIR/perf-results/$TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# Case matrix (from .github/workflows/build-and-run.yml)
# Fields: model_file | model | quant | config | flags | scope(all|ptx)
# ──────────────────────────────────────────────────────────────────────────────
read -r -d '' CASES <<'CASES_EOF'
Llama-3.2-1B-Instruct-F16.gguf|Llama-3.2-1B-Instruct|F16|standard||all
Llama-3.2-1B-Instruct-F16.gguf|Llama-3.2-1B-Instruct|F16|prefill-decode|--with-prefill-decode|all
Llama-3.2-1B-Instruct-F16.gguf|Llama-3.2-1B-Instruct|F16|batch-prefill-decode|--with-prefill-decode --batch-prefill-size 32|all
Llama-3.2-1B-Instruct-F16.gguf|Llama-3.2-1B-Instruct|F16|prefill-decode-cuda-graphs|--with-prefill-decode --cuda-graphs|ptx cuda
Llama-3.2-1B-Instruct-F16.gguf|Llama-3.2-1B-Instruct|F16|batch-prefill-decode-cuda-graphs|--with-prefill-decode --batch-prefill-size 32 --cuda-graphs|ptx cuda
Qwen3-0.6B-f16.gguf|Qwen3-0.6B|F16|standard||all
Qwen3-0.6B-f16.gguf|Qwen3-0.6B|F16|prefill-decode|--with-prefill-decode|all
Qwen3-0.6B-f16.gguf|Qwen3-0.6B|F16|batch-prefill-decode|--with-prefill-decode --batch-prefill-size 32|all
Qwen3-0.6B-f16.gguf|Qwen3-0.6B|F16|prefill-decode-cuda-graphs|--with-prefill-decode --cuda-graphs|ptx cuda
Qwen3-0.6B-f16.gguf|Qwen3-0.6B|F16|batch-prefill-decode-cuda-graphs|--with-prefill-decode --batch-prefill-size 32 --cuda-graphs|ptx cuda
Mistral-7B-Instruct-v0.3.fp16.gguf|Mistral-7B-Instruct-v0.3|F16|standard||all
qwen2.5-1.5b-instruct-fp16.gguf|Qwen2.5-1.5B-Instruct|F16|standard||all
Phi-3-mini-4k-instruct-fp16.gguf|Phi-3-mini-4k-instruct|F16|standard||all
granite-3.2-2b-instruct-f16.gguf|Granite-3.2-2B-Instruct|F16|standard||all
granite-4.0-1b-F16.gguf|Granite-4.0-1B|F16|standard||all
Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct|Q8_0|standard||all
Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct|Q8_0|prefill-decode|--with-prefill-decode|all
Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct|Q8_0|batch-prefill-decode|--with-prefill-decode --batch-prefill-size 32|all
Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct|Q8_0|prefill-decode-cuda-graphs|--with-prefill-decode --cuda-graphs|ptx cuda
Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct|Q8_0|batch-prefill-decode-cuda-graphs|--with-prefill-decode --batch-prefill-size 32 --cuda-graphs|ptx cuda
Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B|Q8_0|standard||all
Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B|Q8_0|prefill-decode|--with-prefill-decode|all
Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B|Q8_0|batch-prefill-decode|--with-prefill-decode --batch-prefill-size 32|all
Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B|Q8_0|prefill-decode-cuda-graphs|--with-prefill-decode --cuda-graphs|ptx cuda
Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B|Q8_0|batch-prefill-decode-cuda-graphs|--with-prefill-decode --batch-prefill-size 32 --cuda-graphs|ptx cuda
Phi-3-mini-4k-instruct-Q8_0.gguf|Phi-3-mini-4k-instruct|Q8_0|standard||all
qwen2.5-1.5b-instruct-q8_0.gguf|Qwen2.5-1.5B-Instruct|Q8_0|standard||all
Mistral-7B-Instruct-v0.3.Q8_0.gguf|Mistral-7B-Instruct-v0.3|Q8_0|standard||all
granite-3.2-2b-instruct-Q8_0.gguf|Granite-3.2-2B-Instruct|Q8_0|standard||all
granite-4.0-1b-Q8_0.gguf|Granite-4.0-1B|Q8_0|standard||all
CASES_EOF

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err()  { printf '\033[1;31m[%s] ERROR\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

# shellcheck disable=SC1090,SC1091
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk use java "$JAVA_SDK" >/dev/null 2>&1 || { err "Could not select Java $JAVA_SDK via sdkman"; exit 1; }
export JAVA_HOME="${JAVA_HOME:-$HOME/.sdkman/candidates/java/current}"
log "Java: $(java -version 2>&1 | head -1)"
log "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"
log "Backends: ${BACKEND_LIST[*]}   reps=$REPS (+$WARMUP warmup)   max_tokens=$MAX_TOKENS"
log "Results dir: $RESULTS_DIR"

build_tornadovm() {
    local backend="$1"
    log "Building TornadoVM for backend: $backend (clean, single-backend)"
    ( cd "$TORNADO_ROOT" && make BACKEND="$backend" ) \
        > "$RESULTS_DIR/tornadovm-build-$backend.log" 2>&1
    return $?
}

# Run one inference pass; writes JSON metrics to the given file. Seed varies per
# rep so repeats sample genuine run-to-run variance. $5=seed, $6=flags.
run_inference() {
    local backend="$1" model_file="$2" metrics_file="$3" run_log="$4" seed="$5" flags="$6"
    export JAVA_TOOL_OPTIONS="-Djitllm.metrics.format=json -Djitllm.metrics.output=file -Djitllm.metrics.file=$metrics_file"
    # shellcheck disable=SC2086
    ( cd "$JITLLM_ROOT_DIR" && \
      ./jitllm --gpu \
        --model "$MODELS_DIR/$model_file" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --seed "$seed" \
        --gpu-memory "$GPU_MEMORY" \
        $flags ) \
        > "$run_log" 2>&1
    local rc=$?
    unset JAVA_TOOL_OPTIONS
    return $rc
}

# ──────────────────────────────────────────────────────────────────────────────
# Main loop: one TornadoVM build per backend, then every applicable case
# ──────────────────────────────────────────────────────────────────────────────
declare -a SUMMARY_ROWS

for backend in "${BACKEND_LIST[@]}"; do
    echo
    log "════════════════════════════════════════════════════════════"
    log "BACKEND: $backend"
    log "════════════════════════════════════════════════════════════"

    if ! build_tornadovm "$backend"; then
        err "TornadoVM build failed for $backend (see $RESULTS_DIR/tornadovm-build-$backend.log)"
        SUMMARY_ROWS+=("$backend|-|-|BUILD_FAILED|-|-|-")
        continue
    fi
    log "TornadoVM build OK"

    # setvars.sh is regenerated each build and points at the new dist dir.
    # shellcheck disable=SC1091
    source "$TORNADO_ROOT/setvars.sh" >/dev/null 2>&1
    export JITLLM_ROOT="$JITLLM_ROOT_DIR"
    log "TORNADOVM_HOME=$TORNADOVM_HOME"

    while IFS='|' read -r model_file model quant config flags scope; do
        [ -z "$model_file" ] && continue
        # Scope filter: "all" runs everywhere; otherwise scope is a space-separated
        # list of backends the case applies to (cuda-graphs runs on ptx + cuda).
        if [ "$scope" != "all" ] && [[ " $scope " != *" $backend "* ]]; then
            continue
        fi
        if [ ! -f "$MODELS_DIR/$model_file" ]; then
            warn "Model not found, skipping: $model_file"
            SUMMARY_ROWS+=("$backend|$model|$quant|$config|MODEL_MISSING|-|-")
            continue
        fi

        tag="${backend}-${model}-${quant}-${config}"
        log "[$model $quant / $config]  ($WARMUP warmup + $REPS reps)"

        for w in $(seq 1 "$WARMUP"); do
            run_inference "$backend" "$model_file" \
                "$RESULTS_DIR/metrics-$tag-warmup$w.json" \
                "$RESULTS_DIR/run-$tag-warmup$w.log" "$((100 + w))" "$flags" \
                || warn "  warmup $w failed"
        done
        rep_ok=0
        for r in $(seq 1 "$REPS"); do
            if run_inference "$backend" "$model_file" \
                    "$RESULTS_DIR/metrics-$tag-rep$r.json" \
                    "$RESULTS_DIR/run-$tag-rep$r.log" "$r" "$flags"; then
                rep_ok=$((rep_ok + 1))
            else
                err "  rep $r failed (see $RESULTS_DIR/run-$tag-rep$r.log)"
            fi
        done

        if [ "$rep_ok" -eq 0 ]; then
            SUMMARY_ROWS+=("$backend|$model|$quant|$config|RUN_FAILED|-|-")
            continue
        fi

        stats=$(python3 - "$RESULTS_DIR" "$tag" <<'PY'
import json, sys, glob, statistics
results_dir, tag = sys.argv[1], sys.argv[2]
ev=[]; pr=[]; jit=[]
for f in sorted(glob.glob(f"{results_dir}/metrics-{tag}-rep*.json")):
    try:
        m=json.load(open(f))
        if isinstance(m.get("eval_rate"),(int,float)): ev.append(m["eval_rate"])
        if isinstance(m.get("prompt_eval_rate"),(int,float)): pr.append(m["prompt_eval_rate"])
        j=m.get("tornadovm",{}).get("jit_duration")
        if isinstance(j,(int,float)): jit.append(j/1e6)
    except Exception:
        pass
def fmt(x):
    if not x: return "-"
    return f"{statistics.mean(x):.2f} ± {statistics.pstdev(x) if len(x)>1 else 0.0:.2f}"
jm = f"{statistics.mean(jit):.0f}" if jit else "-"
print(f"{fmt(ev)}|{fmt(pr)}|{jm}")
PY
)
        IFS='|' read -r eval_s prompt_s jit_ms <<< "$stats"
        log "  eval=$eval_s tok/s  prompt_eval=$prompt_s tok/s  jit≈${jit_ms}ms  (n=$rep_ok)"
        SUMMARY_ROWS+=("$backend|$model|$quant|$config|$eval_s|$prompt_s|${jit_ms}ms")
    done <<< "$CASES"
done

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo
log "════════════════════════════════════════════════════════════"
log "PERFORMANCE SUMMARY   max_tokens=$MAX_TOKENS  reps=$REPS (+$WARMUP warmup)"
log "eval/prompt_eval = tok/s (mean ± std);  jit = one-time compile (mean ms)"
log "════════════════════════════════════════════════════════════"
{
    printf 'backend\tmodel\tquant\tconfig\teval_tok/s\tprompt_eval_tok/s\tjit\n'
    for row in "${SUMMARY_ROWS[@]}"; do
        printf '%s\n' "$row" | tr '|' '\t'
    done
} | column -t -s $'\t' | tee "$RESULTS_DIR/summary.txt"

echo
log "Full results in: $RESULTS_DIR"

#!/usr/bin/env bash
#
# perf-gate-run.sh — measure one tuple and gate it (M1.7 / T1.7).
#
# Runs the documented procedure for a single case: WARMUP generations discarded,
# then REPS measured generations, median eval_rate, then scripts/perf_gate.py
# compares that against the most recent gate-passing entry of the same tuple in
# docs/perf-history.jsonl.
#
# The tuple is (machine, gpu, model, quantization, backend, configuration,
# tornadovm_version). Everything not passed is derived from the environment, and
# every derived value is printed before the runs start — a tuple that silently
# changes is a comparison against the wrong baseline.
#
# Usage:
#   scripts/perf-gate-run.sh --model-file Llama-3.2-1B-Instruct-Q8_0.gguf \
#       --model Llama-3.2-1B-Instruct --quantization Q8_0 --backend cuda \
#       [--configuration standard] [--flags "--with-prefill-decode"] \
#       [--machine NAME] [--gpu NAME] [--tornadovm-version V] \
#       [--append] [--cache-cold]
#
# Exit codes are the gate's: 0 pass or record-only, 1 regression, 2 unstable
# environment, 3 usage error.
set -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$HOME/LLMModels}"
PROMPT="${PROMPT:-Write a long, detailed adventure story about a young explorer discovering a hidden ancient city in the jungle. Include vivid descriptions.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GPU_MEMORY="${GPU_MEMORY:-14GB}"
WARMUP="${WARMUP:-3}"
REPS="${REPS:-5}"

MODEL_FILE="" MODEL="" QUANT="" BACKEND="" CONFIGURATION="standard" FLAGS=""
MACHINE="" GPU="" TORNADOVM_VERSION="" CACHE_WARM="true" APPEND=""
HISTORY="$REPO_ROOT/docs/perf-history.jsonl"
TOLERANCES="$REPO_ROOT/scripts/perf-gate-tolerances.json"

die() { printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 3; }
log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --model-file)         MODEL_FILE="$2"; shift 2 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --quantization)       QUANT="$2"; shift 2 ;;
        --backend)            BACKEND="$2"; shift 2 ;;
        --configuration)      CONFIGURATION="$2"; shift 2 ;;
        --flags)              FLAGS="$2"; shift 2 ;;
        --machine)            MACHINE="$2"; shift 2 ;;
        --gpu)                GPU="$2"; shift 2 ;;
        --tornadovm-version)  TORNADOVM_VERSION="$2"; shift 2 ;;
        --history)            HISTORY="$2"; shift 2 ;;
        --tolerances)         TOLERANCES="$2"; shift 2 ;;
        --append)             APPEND="--append"; shift ;;
        --cache-cold)         CACHE_WARM="false"; shift ;;
        -h|--help)            sed -n '2,30p' "$0"; exit 0 ;;
        *)                    die "unknown argument: $1" ;;
    esac
done

[ -n "$MODEL_FILE" ] || die "--model-file is required"
[ -n "$MODEL" ]      || die "--model is required"
[ -n "$QUANT" ]      || die "--quantization is required"
[ -n "$BACKEND" ]    || die "--backend is required"
[ -f "$MODELS_DIR/$MODEL_FILE" ] || die "model not found: $MODELS_DIR/$MODEL_FILE"

# Derived tuple fields. Each is overridable, because a wrong one silently compares
# against another machine's numbers.
[ -n "$MACHINE" ] || MACHINE="${PERF_MACHINE:-$(hostname -s)}"
[ -n "$GPU" ]     || GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$GPU" ]     || die "could not detect the GPU; pass --gpu"
if [ -z "$TORNADOVM_VERSION" ]; then
    [ -n "$TORNADOVM_HOME" ] || die "TORNADOVM_HOME is unset; source setvars.sh or pass --tornadovm-version"
    # An SDK prepared by scripts/tornadovm-dev.sh carries its provenance four levels up
    # (<install>/TornadoVM/dist/<bundle>/<sdk>): the runtime id is the artifact version plus
    # the commit, because the version alone names every develop commit alike.
    PROV="$TORNADOVM_HOME/../../../../provenance.json"
    if [ -f "$PROV" ]; then
        TORNADOVM_VERSION="$(python3 -c "import json,sys;p=json.load(open(sys.argv[1]));print(p['artifact_version']+'+g'+p['ref'][:12])" "$PROV")"
    else
        # A released SDK: dist directory name tornadovm-<version>-full
        TORNADOVM_VERSION="$(basename "$TORNADOVM_HOME" | sed -E 's/^tornadovm-//; s/-full$//')"
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/perf-results/gate-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RESULTS_DIR/measured" || die "cannot create $RESULTS_DIR"

log "tuple      $MACHINE / $GPU / $MODEL / $QUANT / $BACKEND / $CONFIGURATION / $TORNADOVM_VERSION"
log "procedure  $WARMUP warm-up + $REPS measured, max_tokens=$MAX_TOKENS, cache_warm=$CACHE_WARM"
log "results    $RESULTS_DIR"

# The backend is named explicitly, never left to auto-detection: an SDK built with more
# than one backend picks its own, and the tuple would then record a backend that did not
# run. --gpu alone silently runs on the CPU.
case "$BACKEND" in
    cuda)   BACKEND_FLAG="--cuda" ;;
    opencl) BACKEND_FLAG="--opencl" ;;
    metal)  BACKEND_FLAG="--metal" ;;
    # TornadoVM folded its PTX backend into CUDA, and the launcher has no --ptx flag.
    # Rejected rather than aliased to --cuda: history holds backend=ptx rows, and silently
    # recording a cuda run under that tuple is the mislabelling the note above warns about.
    ptx)    die "backend ptx no longer exists: TornadoVM folded PTX into CUDA — use cuda" ;;
    *)      die "unknown backend: $BACKEND (expected cuda, opencl or metal)" ;;
esac

run_inference() {
    local metrics_file="$1" run_log="$2" seed="$3"
    # EXTRA_JVM_PROPS is appended, not replaced: this assignment overrides any exported
    # JAVA_TOOL_OPTIONS, so a caller trying to add a property that way silently loses it.
    JAVA_TOOL_OPTIONS="-Djitllm.metrics.format=json -Djitllm.metrics.output=file -Djitllm.metrics.file=$metrics_file ${EXTRA_JVM_PROPS:-}" \
    "$REPO_ROOT/jitllm" --gpu "$BACKEND_FLAG" \
        --model "$MODELS_DIR/$MODEL_FILE" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --seed "$seed" \
        --gpu-memory "$GPU_MEMORY" \
        $FLAGS > "$run_log" 2>&1
}

for w in $(seq 1 "$WARMUP"); do
    log "warm-up $w/$WARMUP (discarded)"
    run_inference "$RESULTS_DIR/warmup-$w.json" "$RESULTS_DIR/warmup-$w.log" "$((100 + w))" \
        || die "warm-up $w failed — see $RESULTS_DIR/warmup-$w.log"
done

for r in $(seq 1 "$REPS"); do
    log "measured $r/$REPS"
    run_inference "$RESULTS_DIR/measured/rep-$r.json" "$RESULTS_DIR/rep-$r.log" "$r" \
        || die "measured run $r failed — see $RESULTS_DIR/rep-$r.log"
done

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '')"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"

python3 "$REPO_ROOT/scripts/perf_gate.py" \
    --metrics-dir "$RESULTS_DIR/measured" \
    --machine "$MACHINE" --gpu "$GPU" --model "$MODEL" --quantization "$QUANT" \
    --backend "$BACKEND" --configuration "$CONFIGURATION" \
    --tornadovm-version "$TORNADOVM_VERSION" --cache-warm "$CACHE_WARM" \
    --history "$HISTORY" --tolerances "$TOLERANCES" \
    --warmup-runs "$WARMUP" --expected-runs "$REPS" \
    --commit "$COMMIT" --branch "$BRANCH" --workflow "perf-gate-run.sh" \
    $APPEND

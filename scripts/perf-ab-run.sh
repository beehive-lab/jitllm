#!/usr/bin/env bash
#
# perf-ab-run.sh — paired A/B benchmark of two builds in one session (T2.3).
#
# A stored baseline ages. The same laptop measured 172.5 tok/s and, two hours later,
# 167 tok/s from an unchanged build; compared against history that reads as a 3%
# regression. Only a comparison whose two sides share a thermal state says anything
# about the code, so a change-vs-change verdict is measured here, not looked up:
#
#   * the baseline ref is built in a throwaway git worktree, so the working tree is
#     never stashed or checked out from under you;
#   * runs alternate baseline, candidate, baseline, candidate — interleaving is what
#     makes drift affect both sides equally instead of accumulating on the later one;
#   * both sides go to scripts/perf_gate.py in paired mode, which consults no history.
#
# Usage:
#   scripts/perf-ab-run.sh --baseline-ref HEAD~1 \
#       --model-file Llama-3.2-1B-Instruct-Q8_0.gguf \
#       --model Llama-3.2-1B-Instruct --quantization Q8_0 --backend cuda \
#       [--pairs 5] [--configuration standard] [--flags "..."] [--machine NAME]
#
# For a flag-selected path rather than two builds, point both sides at one commit and
# differentiate with per-side properties:
#
#   BASELINE_JVM_PROPS=-Djitllm.lowering=false CANDIDATE_JVM_PROPS=-Djitllm.lowering=true \
#   scripts/perf-ab-run.sh --baseline-ref HEAD --configuration lowering ...
#
# Exit codes are the gate's: 0 pass, 1 regression, 2 unstable environment, 3 usage error.
set -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$HOME/LLMModels}"
PROMPT="${PROMPT:-Write a long, detailed adventure story about a young explorer discovering a hidden ancient city in the jungle. Include vivid descriptions.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GPU_MEMORY="${GPU_MEMORY:-14GB}"
WARMUP="${WARMUP:-2}"
PAIRS="${PAIRS:-5}"

BASELINE_REF="" MODEL_FILE="" MODEL="" QUANT="" BACKEND="" CONFIGURATION="standard" FLAGS=""
MACHINE="" GPU="" TORNADOVM_VERSION=""
TOLERANCES="$REPO_ROOT/scripts/perf-gate-tolerances.json"

die() { printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 3; }
log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --baseline-ref)      BASELINE_REF="$2"; shift 2 ;;
        --model-file)        MODEL_FILE="$2"; shift 2 ;;
        --model)             MODEL="$2"; shift 2 ;;
        --quantization)      QUANT="$2"; shift 2 ;;
        --backend)           BACKEND="$2"; shift 2 ;;
        --configuration)     CONFIGURATION="$2"; shift 2 ;;
        --flags)             FLAGS="$2"; shift 2 ;;
        --machine)           MACHINE="$2"; shift 2 ;;
        --gpu)               GPU="$2"; shift 2 ;;
        --tornadovm-version) TORNADOVM_VERSION="$2"; shift 2 ;;
        --pairs)             PAIRS="$2"; shift 2 ;;
        --tolerances)        TOLERANCES="$2"; shift 2 ;;
        -h|--help)           sed -n '2,28p' "$0"; exit 0 ;;
        *)                   die "unknown argument: $1" ;;
    esac
done

[ -n "$BASELINE_REF" ] || die "--baseline-ref is required (the build to compare against)"
[ -n "$MODEL_FILE" ]   || die "--model-file is required"
[ -n "$MODEL" ]        || die "--model is required"
[ -n "$QUANT" ]        || die "--quantization is required"
[ -n "$BACKEND" ]      || die "--backend is required"
[ -f "$MODELS_DIR/$MODEL_FILE" ] || die "model not found: $MODELS_DIR/$MODEL_FILE"
git -C "$REPO_ROOT" rev-parse --verify "$BASELINE_REF^{commit}" >/dev/null 2>&1 \
    || die "not a commit: $BASELINE_REF"

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

[ -n "$MACHINE" ] || MACHINE="${PERF_MACHINE:-$(hostname -s)}"
[ -n "$GPU" ]     || GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$GPU" ]     || die "could not detect the GPU; pass --gpu"
if [ -z "$TORNADOVM_VERSION" ]; then
    [ -n "$TORNADOVM_HOME" ] || die "TORNADOVM_HOME is unset; source setvars.sh or pass --tornadovm-version"
    TORNADOVM_VERSION="$(basename "$TORNADOVM_HOME" | sed -E 's/^tornadovm-//; s/-full$//')"
fi

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/perf-results/ab-$(date +%Y%m%d-%H%M%S)}"
WORKTREE="$RESULTS_DIR/baseline-worktree"
mkdir -p "$RESULTS_DIR/baseline" "$RESULTS_DIR/candidate" || die "cannot create $RESULTS_DIR"

cleanup() {
    if [ -d "$WORKTREE" ]; then
        git -C "$REPO_ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1
    fi
}
trap cleanup EXIT

BASELINE_SHA="$(git -C "$REPO_ROOT" rev-parse --short "$BASELINE_REF")"
log "tuple      $MACHINE / $GPU / $MODEL / $QUANT / $BACKEND / $CONFIGURATION / $TORNADOVM_VERSION"
log "comparing  candidate (working tree) vs baseline $BASELINE_REF ($BASELINE_SHA)"
log "procedure  $WARMUP warm-up per side, then $PAIRS interleaved pairs, max_tokens=$MAX_TOKENS"
log "results    $RESULTS_DIR"

log "building candidate (working tree)"
( cd "$REPO_ROOT" && mvn -o -q package -DskipTests ) > "$RESULTS_DIR/build-candidate.log" 2>&1 \
    || die "candidate build failed — see $RESULTS_DIR/build-candidate.log"

log "checking out and building baseline $BASELINE_SHA in a throwaway worktree"
git -C "$REPO_ROOT" worktree add --detach "$WORKTREE" "$BASELINE_REF" > "$RESULTS_DIR/worktree.log" 2>&1 \
    || die "could not create the baseline worktree — see $RESULTS_DIR/worktree.log"
( cd "$WORKTREE" && mvn -o -q package -DskipTests ) > "$RESULTS_DIR/build-baseline.log" 2>&1 \
    || die "baseline build failed — see $RESULTS_DIR/build-baseline.log"

# $1 = repository root to run from (this selects the build), $2 = metrics file,
# $3 = log file, $4 = seed.
# $5 = side-specific JVM properties. A/B is not always two builds: a path that is
# selected by a flag (the lowering, T13.6) has one build and two configurations, and
# without a per-side hook both legs would run the same path and the comparison would be
# a build against itself. BASELINE_JVM_PROPS/CANDIDATE_JVM_PROPS supply that, and
# EXTRA_JVM_PROPS still applies to both sides.
# The baseline side is an arbitrary older commit, which may predate the rename of
# GPULlama3.java to jitllm. Across that boundary three things differ: the launcher filename,
# the root environment variable, and the metrics system-property prefix. Comparing across
# the rename is exactly what this tool is for, so pick the launcher that exists and set
# both spellings of the other two — each build ignores the prefix it does not know.
# Returns non-zero rather than calling die(): it runs inside $(...), where an exit would
# only kill the subshell and leave the caller with an empty launcher path.
launcher_for() {
    local root="$1"
    if   [ -x "$root/jitllm" ];          then echo "$root/jitllm"
    elif [ -x "$root/llama-tornado" ]; then echo "$root/llama-tornado"
    else return 1
    fi
}

metrics_props_for() {
    local metrics_file="$1"
    printf '%s %s' \
        "-Djitllm.metrics.format=json -Djitllm.metrics.output=file -Djitllm.metrics.file=$metrics_file" \
        "-Dllama.metrics.format=json -Dllama.metrics.output=file -Dllama.metrics.file=$metrics_file"
}

run_inference() {
    local root="$1" metrics_file="$2" run_log="$3" seed="$4" side_props="$5"
    local launcher
    launcher="$(launcher_for "$root")" \
        || die "no launcher in $root (looked for jitllm and llama-tornado)"
    JITLLM_ROOT="$root" LLAMA_ROOT="$root" \
    JAVA_TOOL_OPTIONS="$(metrics_props_for "$metrics_file") ${EXTRA_JVM_PROPS:-} $side_props" \
    "$launcher" --gpu "$BACKEND_FLAG" \
        --model "$MODELS_DIR/$MODEL_FILE" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --seed "$seed" \
        --gpu-memory "$GPU_MEMORY" \
        $FLAGS > "$run_log" 2>&1
}

for w in $(seq 1 "$WARMUP"); do
    log "warm-up $w/$WARMUP (both sides, discarded)"
    run_inference "$WORKTREE"  "$RESULTS_DIR/warmup-baseline-$w.json"  "$RESULTS_DIR/warmup-baseline-$w.log"  "$((100 + w))" "${BASELINE_JVM_PROPS:-}" \
        || die "baseline warm-up $w failed — see $RESULTS_DIR/warmup-baseline-$w.log"
    run_inference "$REPO_ROOT" "$RESULTS_DIR/warmup-candidate-$w.json" "$RESULTS_DIR/warmup-candidate-$w.log" "$((100 + w))" "${CANDIDATE_JVM_PROPS:-}" \
        || die "candidate warm-up $w failed — see $RESULTS_DIR/warmup-candidate-$w.log"
done

for i in $(seq 1 "$PAIRS"); do
    log "pair $i/$PAIRS"
    run_inference "$WORKTREE"  "$RESULTS_DIR/baseline/rep-$i.json"  "$RESULTS_DIR/log-baseline-$i.log"  "$i" "${BASELINE_JVM_PROPS:-}" \
        || die "baseline run $i failed — see $RESULTS_DIR/log-baseline-$i.log"
    run_inference "$REPO_ROOT" "$RESULTS_DIR/candidate/rep-$i.json" "$RESULTS_DIR/log-candidate-$i.log" "$i" "${CANDIDATE_JVM_PROPS:-}" \
        || die "candidate run $i failed — see $RESULTS_DIR/log-candidate-$i.log"
done

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '')"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"

python3 "$REPO_ROOT/scripts/perf_gate.py" \
    --metrics-dir "$RESULTS_DIR/candidate" \
    --baseline-metrics-dir "$RESULTS_DIR/baseline" \
    --baseline-label "$BASELINE_REF ($BASELINE_SHA)" \
    --machine "$MACHINE" --gpu "$GPU" --model "$MODEL" --quantization "$QUANT" \
    --backend "$BACKEND" --configuration "$CONFIGURATION" \
    --tornadovm-version "$TORNADOVM_VERSION" --cache-warm true \
    --tolerances "$TOLERANCES" \
    --warmup-runs "$WARMUP" --expected-runs "$PAIRS" \
    --commit "$COMMIT" --branch "$BRANCH" --workflow "perf-ab-run.sh"

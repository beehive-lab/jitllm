#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: MODEL=/path/to/model.gguf $0 <langchain4j-checkout>" >&2
    exit 2
fi

langchain_dir=$(realpath "$1")
module_dir="${langchain_dir}/langchain4j-gpu-llama3"

[[ -x "${langchain_dir}/mvnw" ]] || {
    echo "LangChain4j Maven wrapper not found: ${langchain_dir}/mvnw" >&2
    exit 2
}
[[ -d "$module_dir" ]] || {
    echo "LangChain4j jitllm module not found: $module_dir" >&2
    exit 2
}
[[ -n ${MODEL:-} && -f $MODEL ]] || {
    echo "MODEL must name an existing GGUF file" >&2
    exit 2
}
[[ -n ${TORNADOVM_HOME:-} && -x ${TORNADOVM_HOME}/bin/tornado ]] || {
    echo "TORNADOVM_HOME must name a valid TornadoVM SDK" >&2
    exit 2
}

java -version
"${TORNADOVM_HOME}/bin/tornado" --version

log_file=$(mktemp)
trap 'rm -f "$log_file"' EXIT

set +e
(
    cd "$module_dir"
    ../mvnw -P run-tests
) 2>&1 | tee "$log_file"
maven_status=${PIPESTATUS[0]}
set -e

if ((maven_status != 0)); then
    echo "Maven/TornadoVM test command failed with status ${maven_status}" >&2
    exit "$maven_status"
fi

started=$(sed -n 's/^\[[[:space:]]*\([0-9][0-9]*\) tests started[[:space:]]*\]$/\1/p' "$log_file" | tail -1)
failed=$(sed -n 's/^\[[[:space:]]*\([0-9][0-9]*\) tests failed[[:space:]]*\]$/\1/p' "$log_file" | tail -1)

if [[ -z $started || -z $failed ]]; then
    echo "JUnit summary was not found; refusing to treat the run as successful" >&2
    exit 1
fi
if ((started == 0)); then
    echo "JUnit discovered no runnable tests" >&2
    exit 1
fi
if ((failed != 0)); then
    echo "JUnit reported ${failed} failed test(s)" >&2
    exit 1
fi

echo "Validated ${started} integration test(s) with zero failures"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <quarkus-langchain4j-checkout>" >&2
    exit 2
fi

quarkus_langchain4j_dir=$(realpath "$1")
runtime_module=model-providers/jitllm/runtime
deployment_module=model-providers/jitllm/deployment

[[ -x "${quarkus_langchain4j_dir}/mvnw" ]] || {
    echo "quarkus-langchain4j Maven wrapper not found: ${quarkus_langchain4j_dir}/mvnw" >&2
    exit 2
}
[[ -d "${quarkus_langchain4j_dir}/${runtime_module}" ]] || {
    echo "jitllm runtime module not found: ${quarkus_langchain4j_dir}/${runtime_module}" >&2
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
    cd "$quarkus_langchain4j_dir"
    ./mvnw test -pl "${runtime_module},${deployment_module}" -am
) 2>&1 | tee "$log_file"
maven_status=${PIPESTATUS[0]}
set -e

if ((maven_status != 0)); then
    echo "Maven test command failed with status ${maven_status}" >&2
    exit "$maven_status"
fi

# Surefire prints its per-module summary through Maven's logger, so the line carries an
# "[INFO] " prefix. Anchoring on a bare "Tests run:" found nothing and reported a passing
# run as unvalidated.
summary_pattern='^(\[INFO\] )?Tests run: [0-9]+, Failures: [0-9]+, Errors: [0-9]+, Skipped: [0-9]+$'
summary_line=$(grep -E "$summary_pattern" "$log_file" | tail -1 || true)

if [[ -z $summary_line ]]; then
    echo "Surefire summary was not found; refusing to treat the run as successful" >&2
    exit 1
fi

run=$(sed -n 's/.*Tests run: \([0-9]*\).*/\1/p' <<<"$summary_line")
failures=$(sed -n 's/.*Failures: \([0-9]*\).*/\1/p' <<<"$summary_line")
errors=$(sed -n 's/.*Errors: \([0-9]*\).*/\1/p' <<<"$summary_line")

if ((run == 0)); then
    echo "Maven discovered no runnable tests" >&2
    exit 1
fi
if ((failures != 0 || errors != 0)); then
    echo "Surefire reported ${failures} failure(s) and ${errors} error(s)" >&2
    exit 1
fi

echo "Validated ${run} test(s) with zero failures/errors"

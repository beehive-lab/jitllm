#!/usr/bin/env bash
# Wait until both published lines of a jitllm release resolve on Maven Central, then check their
# bytecode (reuses the LangChain4j skill's inspect-release.sh).
#
#   wait-for-central.sh <version> [timeout-minutes, default 90]
set -euo pipefail

if [[ $# -lt 1 || ! $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Usage: $0 <version> [timeout-minutes]" >&2
    exit 2
fi
version=$1
timeout_minutes=${2:-90}
base=https://repo1.maven.org/maven2/io/github/beehive-lab/jitllm
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

deadline=$(( $(date +%s) + timeout_minutes * 60 ))
while :; do
    missing=()
    for jdk in jdk21 jdk22plus; do
        url="$base/${version}-${jdk}/jitllm-${version}-${jdk}.pom"
        code=$(curl -s -o /dev/null -w '%{http_code}' "$url")
        [[ $code == 200 ]] || missing+=("${version}-${jdk}")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "$(date -u +%FT%TZ) both lines resolve on Maven Central"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        echo "$(date -u +%FT%TZ) still missing after ${timeout_minutes} min: ${missing[*]}" >&2
        exit 1
    fi
    echo "$(date -u +%FT%TZ) not yet: ${missing[*]}"
    sleep 60
done

"$here/../../update-langchain4j-integration/scripts/inspect-release.sh" "$version"

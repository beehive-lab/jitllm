#!/usr/bin/env bash
# set-tornadovm-release.sh X.Y.Z — record the published TornadoVM release that jitllm RELEASE
# builds (-P release) depend on, in pom.xml's tornadovm.release.version.
#
# Development builds are untouched: they compile against the SDK at TORNADOVM_HOME. The release
# profile depends on X.Y.Z-jdk21 / X.Y.Z-jdk22plus from Maven Central and nothing else.
set -euo pipefail
V=${1:-}
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 X.Y.Z" >&2; exit 2; }
POM=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pom.xml
if grep -qE '<tornadovm\.release\.version/>|<tornadovm\.release\.version>[^<]*</tornadovm\.release\.version>' "$POM"; then
  sed -i -E "s|<tornadovm\.release\.version/>|<tornadovm.release.version>$V</tornadovm.release.version>|; s|<tornadovm\.release\.version>[^<]*</tornadovm\.release\.version>|<tornadovm.release.version>$V</tornadovm.release.version>|" "$POM"
else
  echo "$POM has no tornadovm.release.version property" >&2; exit 1
fi
grep -q "<tornadovm.release.version>$V</tornadovm.release.version>" "$POM" || { echo "failed to set tornadovm.release.version" >&2; exit 1; }
echo "tornadovm.release.version = $V"

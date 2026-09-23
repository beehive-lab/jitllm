#!/usr/bin/env bash
# set-tornadovm-release.sh X.Y.Z — record the published TornadoVM release that jitllm RELEASE
# builds (-P release) depend on, in pom.xml's tornadovm.release.version.
#
# Development builds are untouched: they follow TornadoVM develop through
# scripts/tornadovm-dev.sh, and tornadovm.base.version (develop's own base number, which may
# equal a release number without meaning it) is deliberately not changed here. The release
# profile substitutes this property for the base version and drops the -dev qualifier, so a
# release build resolves X.Y.Z-jdk21 / X.Y.Z-jdk22plus and nothing else.
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

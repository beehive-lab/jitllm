#!/usr/bin/env bash
# ide-setup.sh — point the ide Maven profile at the TornadoVM SDK in TORNADOVM_HOME.
#
# The ide profile adds the SDK's jars as system dependencies, whose paths carry the SDK's version
# (tornado-api-7.1.0.jar in a released SDK, tornado-api-<base>-jdk21-dev.jar in a develop build).
# A POM cannot discover a file name, so this reads the version off the SDK's tornado-api jar and
# records it as -Dtornadovm.sdk.version in .mvn/maven.config, which Maven and the IDEs read on
# every build. Other lines in that file are kept. Run it again after switching TORNADOVM_HOME.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/.mvn/maven.config"
[ -n "${TORNADOVM_HOME:-}" ] || { echo "ide-setup: TORNADOVM_HOME is not set" >&2; exit 1; }
shopt -s nullglob
jars=("$TORNADOVM_HOME"/share/java/tornado/tornado-api-*.jar)
[ ${#jars[@]} -eq 1 ] || { echo "ide-setup: expected one tornado-api jar in $TORNADOVM_HOME/share/java/tornado, found ${#jars[@]}" >&2; exit 1; }
V=$(basename "${jars[0]}" .jar); V=${V#tornado-api-}
mkdir -p "$ROOT/.mvn"
touch "$CONFIG"
{ grep -v -- '^-Dtornadovm\.sdk\.version=' "$CONFIG" || true; echo "-Dtornadovm.sdk.version=$V"; } > "$CONFIG.tmp"
mv "$CONFIG.tmp" "$CONFIG"
echo "tornadovm.sdk.version = $V (in $CONFIG); enable the ide profile in the IDE's Maven settings"

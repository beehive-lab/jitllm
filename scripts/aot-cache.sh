#!/usr/bin/env bash
#
# aot-cache.sh — run the jitllm CLI on a JDK 25+ JVM with a Leyden AOT cache (JEP 483/514/515):
# classes loaded and linked ahead of time, and method profiles from a training run, so the JIT
# compilers start warm. The alternative to a native image (scripts/native-image.sh) that keeps the
# HotSpot JVM.
#
# Usage:
#   scripts/aot-cache.sh train <cache> [jitllm args...]   one training run that writes <cache>
#   scripts/aot-cache.sh run   <cache> [jitllm args...]   run with the cache
#
# Train with the workload you will run (same model family, GPU or CPU path): the cache records the
# classes and profiles that run used. Training and production runs must use the same JDK, the same
# jar and the same JVM options; the JVM ignores a cache that does not match (with a warning).
#
# Needs TORNADOVM_HOME (a jdk22plus SDK), JAVA_HOME (JDK 25+) and target/jitllm-*-jdk22plus.jar.
# JVM_OPTS adds JVM options (heap, -Dtornado.*, -Djitllm.*) to both modes.
#
# Example:
#   scripts/aot-cache.sh train target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Tell me a joke" --instruct
#   scripts/aot-cache.sh run   target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Write a haiku" --instruct
set -euo pipefail

: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM jdk22plus SDK}"
JAVA=${JAVA_HOME:+$JAVA_HOME/bin/}java
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# The most recently built jar, not the alphabetically first: target/ may still hold one from an older version.
JAR=$(ls -t "$ROOT"/target/jitllm-*-jdk22plus.jar 2>/dev/null | grep -v -- '-sources\|-javadoc' | head -1 || true)
[ -n "$JAR" ] || { echo "no target/jitllm-*-jdk22plus.jar: run ./mvnw -DskipTests package on JDK 22+" >&2; exit 2; }
[ $# -ge 2 ] || { sed -n '8,10p' "$0" | sed 's/^# //' >&2; exit 2; }
mode=$1 cache=$2; shift 2

# The SDK's argfile, minus its "." module path entry: an AOT cache only accepts jars on the module path.
JARS=$TORNADOVM_HOME/share/java/tornado
MODS=$(sed -n 's/^--add-modules //p' "$TORNADOVM_HOME/tornado-argfile")
# Native access for every TornadoVM module (drivers, cuBLAS, MLX make restricted FFM calls too).
TORNADO_MODS=$(tr ',' '\n' <<<"$MODS" | grep '^tornado\.' | paste -sd, -)
OPENS=$(sed -n 's/^--add-opens //p' "$TORNADOVM_HOME/tornado-argfile" | sed 's/^/--add-opens=/' | paste -sd' ' -)
LOADS=$(grep -o '^-Dtornado\.load\.[a-z.]*implementation=[A-Za-z.]*' "$TORNADOVM_HOME/tornado-argfile" | paste -sd' ' -)
# shellcheck disable=SC2206
JVM=(-Djava.library.path="$TORNADOVM_HOME/lib" --module-path="$JARS" --add-modules="$MODS,jdk.incubator.vector"
     --enable-native-access="ALL-UNNAMED,$TORNADO_MODS" $OPENS $LOADS -Dtornado.tvm.maxbytecodesize=65536
     ${JVM_OPTS:-} -cp "$JAR")

case "$mode" in
  train) exec "$JAVA" "${JVM[@]}" -XX:AOTCacheOutput="$cache" org.beehive.jitllm.JitllmApp "$@" ;;
  run)   exec "$JAVA" "${JVM[@]}" -XX:AOTCache="$cache" org.beehive.jitllm.JitllmApp "$@" ;;
  *)     sed -n '8,10p' "$0" | sed 's/^# //' >&2; exit 2 ;;
esac

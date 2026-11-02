#!/usr/bin/env bash
#
# native-image.sh — build the jitllm CLI as a GraalVM native image, and refresh the TornadoVM/JDK
# part of its native-image metadata.
#
# jitllm's own metadata ships in the jar (src/main/resources/META-INF/native-image/
# io.github.beehive-lab/jitllm/): JitllmNativeImageFeature registers the kernels and task lambdas,
# native-image.properties carries the builder flags, so `build` needs no tracing run. TornadoVM 8.0.0
# ships no metadata of its own; until it does, tornadovm/reachability-metadata.json next to it holds
# what TornadoVM and the JDK need (reflection, resources, FFM downcalls), recorded by `trace`.
#
# Usage:
#   scripts/native-image.sh build  [-o <output>]     native image of org.beehive.jitllm.JitllmApp
#   scripts/native-image.sh trace  [models-dir]       re-record tornadovm/reachability-metadata.json
#
# Needs TORNADOVM_HOME (a jdk22plus SDK) and GRAALVM_HOME (GraalVM for JDK 25 with native-image),
# and target/jitllm-*-jdk22plus.jar (./mvnw -DskipTests package on JDK 22+).
#
# Running the image: it takes the system properties the launcher would pass, plus the library path.
# Keep TORNADOVM_HOME exported: TornadoVM keeps its CUDA code cache in $TORNADOVM_HOME/var, and with
# the variable unset it writes ./null/var/cuda-codecache instead (a cold cache in every directory).
#   ./target/jitllm -Duse.tornadovm=true -Djava.library.path=$TORNADOVM_HOME/lib -Xmx20g \
#       -m model.gguf -p "Tell me a joke" --instruct
set -euo pipefail

: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM jdk22plus SDK}"
: "${GRAALVM_HOME:?set GRAALVM_HOME to a GraalVM for JDK 25 installation with native-image}"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=org.beehive.jitllm.JitllmApp
JAR=$(ls "$ROOT"/target/jitllm-*-jdk22plus.jar 2>/dev/null | grep -v -- '-sources\|-javadoc' | head -1 || true)
[ -n "$JAR" ] || { echo "no target/jitllm-*-jdk22plus.jar: run ./mvnw -DskipTests package on JDK 22+" >&2; exit 2; }
SHIM=$ROOT/src/main/resources/META-INF/native-image/io.github.beehive-lab/jitllm/tornadovm/reachability-metadata.json

# The SDK's module set, as its argfile adds it (minus the JDK modules, added below).
JARS=$TORNADOVM_HOME/share/java/tornado
MODS=$(sed -n 's/^--add-modules //p' "$TORNADOVM_HOME/tornado-argfile" | tr ',' '\n' | grep '^tornado\.' | paste -sd, -)
[ -n "$MODS" ] || { echo "cannot read --add-modules from $TORNADOVM_HOME/tornado-argfile" >&2; exit 2; }
OPENS=()
for m in tornado.runtime tornado.drivers.cuda tornado.drivers.opencl tornado.drivers.metal; do
  [[ ",$MODS," == *",$m,"* ]] && OPENS+=(--add-opens "java.base/java.lang=$m")
done
JVM=(--enable-native-access="ALL-UNNAMED,$MODS" -p "$JARS"
     --add-modules "jdk.unsupported,java.management,jdk.management,jdk.incubator.vector,$MODS" "${OPENS[@]}")

build() {
  local out=$ROOT/target/jitllm
  [ "${1:-}" = "-o" ] && out=$2
  [ -f "$SHIM" ] || { echo "missing $SHIM: run '$0 trace' first" >&2; exit 2; }
  "$GRAALVM_HOME/bin/native-image" "${JVM[@]}" -cp "$JAR" -o "$out" "$MAIN"
  echo "built $out"
}

trace() {
  local models=${1:-/opt/models} dir=$ROOT/target/native-trace
  rm -rf "$dir"
  run() { # label, extra JVM options...
    local label=$1; shift
    local model=$1; shift
    [ -f "$models/$model" ] || { echo "[trace] $label: skipped, no $models/$model" >&2; return; }
    echo "[trace] $label ($model)"
    "$GRAALVM_HOME/bin/java" -agentlib:native-image-agent=config-merge-dir="$dir" "${JVM[@]}" \
      -Djava.library.path="$TORNADOVM_HOME/lib" -Dtornado.tvm.maxbytecodesize=65536 -Xmx20g "$@" \
      -cp "$JAR" "$MAIN" -m "$models/$model" --temperature 0.0 --seed 1 --ctx-size 512 \
      --max-new-tokens 8 -p "Tell me a joke" --instruct > "$dir.$label.log" 2>&1 \
      || echo "[trace] $label failed (see $dir.$label.log); its metadata is kept" >&2
  }
  mkdir -p "$(dirname "$dir")"
  run gpu          Qwen3-0.6B-f16.gguf       -Duse.tornadovm=true
  run gpu-plain    Qwen3-0.6B-f16.gguf       -Duse.tornadovm=true -Djitllm.withPrefillDecode=false -Djitllm.cudaGraphs=false
  run gpu-q8       Llama-3.2-1B-Instruct-Q8_0.gguf -Duse.tornadovm=true
  run gpu-native   Qwen3-0.6B-f16.gguf       -Duse.tornadovm=true -Djitllm.nativeLibraries=true
  run cpu          Qwen3-0.6B-Q8_0.gguf      -Duse.tornadovm=false

  # Keep TornadoVM's and the JDK's entries; jitllm's own are the feature's and the glob's job.
  python3 - "$dir/reachability-metadata.json" "$SHIM" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
def ours(name):
    return name.startswith("org.beehive.jitllm.") or name.startswith("org/beehive/jitllm/") \
        or name.startswith("META-INF/services/org.beehive.jitllm.")
def keep_type(e):
    t = e["type"]
    return not ours(t if isinstance(t, str) else t["lambda"]["declaringClass"])
out = {"comment": "TornadoVM 8 and JDK entries for jitllm native images, recorded by scripts/native-image.sh trace. "
                  "Belongs in TornadoVM's own jars once it ships native-image metadata."}
out["reflection"] = sorted([e for e in d.get("reflection", []) if keep_type(e)], key=lambda e: json.dumps(e["type"]))
out["resources"] = [r for r in d.get("resources", []) if not ours(r.get("glob", r.get("bundle", "")))]
if "foreign" in d:
    out["foreign"] = d["foreign"]
for k in d:
    if k not in ("reflection", "resources", "foreign"):
        out[k] = d[k]
json.dump(out, open(dst, "w"), indent=2)
open(dst, "a").write("\n")
print(f"[trace] wrote {dst}: {len(out['reflection'])} reflection, {len(out['resources'])} resources, "
      f"{len(out.get('foreign', {}).get('downcalls', []))} downcalls")
EOF
}

case "${1:-}" in
  build) shift; build "$@" ;;
  trace) shift; trace "$@" ;;
  *) sed -n '12,14p' "$0" | sed 's/^# //' >&2; exit 2 ;;
esac

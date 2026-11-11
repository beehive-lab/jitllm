#!/usr/bin/env bash
#
# native-image.sh — build the jitllm CLI as a GraalVM native image, and refresh the TornadoVM/JDK
# part of its native-image metadata.
#
# jitllm's own metadata ships in the jar (src/main/resources/META-INF/native-image/
# io.github.beehive-lab/jitllm/): JitllmNativeImageFeature registers the kernels and task lambdas,
# native-image.properties carries the builder flags, so `build` needs no tracing run. TornadoVM 8
# ships no metadata of its own; until it does, tornadovm/reachability-metadata.json next to it holds
# what TornadoVM and the JDK need (reflection, resources, FFM downcalls), recorded by `trace`.
#
# Every TornadoVM backend makes its own FFM downcalls, and an image can only make the ones it was
# built with: `trace` therefore merges into the file instead of replacing it. Run it once on each
# backend the image must support (CUDA, OpenCL, Metal); delete the file first to start over.
#
# Usage:
#   scripts/native-image.sh build  [-o <output>]     native image of org.beehive.jitllm.JitllmApp
#   scripts/native-image.sh trace  [models-dir]       record this backend into tornadovm/reachability-metadata.json
#
# Needs TORNADOVM_HOME (a jdk22plus SDK) and GRAALVM_HOME (GraalVM for JDK 25 with native-image),
# and target/jitllm-*-jdk22plus.jar (./mvnw -DskipTests package on JDK 22+).
#
# Running the image: it takes the system properties the launcher would pass, plus the library path.
# Keep TORNADOVM_HOME exported: TornadoVM keeps its CUDA code cache in $TORNADOVM_HOME/var, and with
# the variable unset it writes ./null/var/cuda-codecache instead (a cold cache in every directory).
#   ./target/jitllm -Duse.tornadovm=true -Djava.library.path=$TORNADOVM_HOME/lib \
#       -Dtornado.tvm.maxbytecodesize=65536 -Xmx20g -m model.gguf -p "Tell me a joke" --instruct
set -euo pipefail

: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM jdk22plus SDK}"
: "${GRAALVM_HOME:?set GRAALVM_HOME to a GraalVM for JDK 25 installation with native-image}"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=org.beehive.jitllm.JitllmApp
# The most recently built jar, not the alphabetically first: target/ may still hold one from an older version.
JAR=$(ls -t "$ROOT"/target/jitllm-*-jdk22plus.jar 2>/dev/null | grep -v -- '-sources\|-javadoc' | head -1 || true)
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

  # Keep TornadoVM's and the JDK's entries; jitllm's own are the feature's and the glob's job. Merged
  # into what the file already holds, so the other backends' entries stay.
  python3 - "$dir/reachability-metadata.json" "$SHIM" <<'EOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
old = json.load(open(dst)) if os.path.exists(dst) else {}
def key(x):
    return json.dumps(x, sort_keys=True)
def union(a, b):
    seen, out = set(), []
    for x in a + b:
        if key(x) not in seen:
            seen.add(key(x))
            out.append(x)
    return sorted(out, key=key)
def merge_type(a, b):
    m = dict(a)
    for k, v in b.items():
        if k not in m:
            m[k] = v
        elif isinstance(v, list):
            m[k] = union(m[k], v)
        elif isinstance(v, bool):
            m[k] = m[k] or v
    return m
def ours(name):
    return name.startswith("org.beehive.jitllm.") or name.startswith("org/beehive/jitllm/") \
        or name.startswith("META-INF/services/org.beehive.jitllm.")
def keep_type(e):
    t = e["type"]
    if isinstance(t, str):
        return not ours(t)
    if "lambda" in t:
        return not ours(t["lambda"].get("declaringClass", ""))
    if "proxy" in t:
        return not all(ours(i) for i in t["proxy"])
    return True
out = {"comment": "TornadoVM 8 and JDK entries for jitllm native images, recorded by scripts/native-image.sh trace "
                  "on each backend (CUDA, OpenCL, Metal) and merged. "
                  "Belongs in TornadoVM's own jars once it ships native-image metadata."}
types = {}
for e in old.get("reflection", []) + [e for e in d.get("reflection", []) if keep_type(e)]:
    k = key(e["type"])
    types[k] = merge_type(types[k], e) if k in types else e
out["reflection"] = [types[k] for k in sorted(types)]
out["resources"] = union(old.get("resources", []),
                         [r for r in d.get("resources", []) if not ours(r.get("glob", r.get("bundle", "")))])
foreign = {}
for part in (old.get("foreign", {}), d.get("foreign", {})):
    for k, v in part.items():
        foreign[k] = union(foreign.get(k, []), v)
if foreign:
    out["foreign"] = foreign
for k in sorted(set(old) | set(d)):
    if k not in ("comment", "reflection", "resources", "foreign"):
        out[k] = union(old.get(k, []), d.get(k, []))
json.dump(out, open(dst, "w"), indent=2)
open(dst, "a").write("\n")
print(f"[trace] wrote {dst}: {len(out['reflection'])} reflection, {len(out['resources'])} resources, "
      f"{len(out.get('foreign', {}).get('downcalls', []))} downcalls")
EOF
}

case "${1:-}" in
  build) shift; build "$@" ;;
  trace) shift; trace "$@" ;;
  *) sed -n '16,18p' "$0" | sed 's/^# //' >&2; exit 2 ;;
esac

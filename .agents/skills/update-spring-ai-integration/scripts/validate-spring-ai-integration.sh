#!/usr/bin/env bash
# Runs the unit tests and real-model ITs of spring-ai-jitllm and its autoconfigure module in a
# TornadoVM JVM, on the GPU. Build the modules with the same JDK first.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <spring-ai-checkout> <java-home> <tornadovm-home> <model.gguf>" >&2
    exit 2
fi

spring_ai_dir=$(realpath "$1")
export JAVA_HOME=$(realpath "$2") TORNADOVM_HOME=$(realpath "$3") MODEL=$(realpath "$4")
export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"

[[ -x "$spring_ai_dir/mvnw" ]] || { echo "Spring AI Maven wrapper not found in $spring_ai_dir" >&2; exit 2; }
[[ -x "$TORNADOVM_HOME/bin/tornado" ]] || { echo "Not a TornadoVM SDK: $TORNADOVM_HOME" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "MODEL must name an existing GGUF file" >&2; exit 2; }

java -version
"$TORNADOVM_HOME/bin/tornado" --version

# jitllm:*-jdk21 uses java.lang.foreign, a preview API on JDK 21.
preview=""
if java -version 2>&1 | grep -q '"21\.'; then
    preview="--enable-preview"
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

status=0
for module in models/spring-ai-jitllm auto-configurations/models/spring-ai-autoconfigure-model-jitllm; do
    module_dir="$spring_ai_dir/$module"
    [[ -d "$module_dir/target/test-classes" ]] || { echo "Build $module first" >&2; exit 2; }
    (cd "$module_dir" && "$spring_ai_dir/mvnw" -B -q dependency:build-classpath \
        -Dmdep.includeScope=test -Dmdep.outputFile="$work/cp.txt" >/dev/null)
    platform=$(tr ':' '\n' < "$work/cp.txt" | sed -n 's|.*/junit-platform-engine/\([^/]*\)/.*|\1|p' | head -1)
    launcher="$work/junit-platform-console-standalone-$platform.jar"
    if [[ ! -f $launcher ]]; then
        (cd "$module_dir" && "$spring_ai_dir/mvnw" -B -q dependency:copy \
            -Dartifact="org.junit.platform:junit-platform-console-standalone:$platform" \
            -DoutputDirectory="$work" >/dev/null)
    fi
    log="$work/$(basename "$module").log"
    set +e
    "$TORNADOVM_HOME/bin/tornado" \
        --jvm="-Duse.tornadovm=true $preview --add-modules jdk.incubator.vector -Dtornado.device.memory=20GB -Xmx8g" \
        -cp "$module_dir/target/classes:$module_dir/target/test-classes:$(cat "$work/cp.txt"):$launcher" \
        --params="execute --scan-classpath $module_dir/target/test-classes --include-classname=.*Tests --include-classname=.*IT --details=tree" \
        org.junit.platform.console.ConsoleLauncher > "$log" 2>&1
    set -e
    found=$(sed -n 's/^\[[[:space:]]*\([0-9][0-9]*\) tests found[[:space:]]*\]$/\1/p' "$log" | tail -1)
    failed=$(sed -n 's/^\[[[:space:]]*\([0-9][0-9]*\) tests failed[[:space:]]*\]$/\1/p' "$log" | tail -1)
    if [[ -z $found || -z $failed || $found == 0 ]]; then
        echo "$(basename "$module"): no JUnit summary; refusing to treat the run as successful" >&2
        tail -40 "$log" >&2
        status=1
        continue
    fi
    echo "$(basename "$module"): $((found - failed))/$found passed"
    if ((failed != 0)); then
        sed 's/\x1b\[[0-9;]*m//g' "$log" | sed -n '/^Failures (/,/^Test run finished/p' | grep -E "JUnit Jupiter:|=>" >&2
        status=1
    fi
done
exit $status

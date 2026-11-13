#!/usr/bin/env bash
# Builds the Spring Boot smoke app against the locally installed Spring AI snapshot and runs it
# on the GPU through TornadoVM. Install spring-ai-bom, spring-ai-model and the jitllm modules first.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <java-home> <tornadovm-home> <model.gguf>" >&2
    exit 2
fi

export JAVA_HOME=$(realpath "$1") TORNADOVM_HOME=$(realpath "$2") MODEL=$(realpath "$3")
export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"
command -v mvn >/dev/null || { echo "mvn is required on PATH" >&2; exit 2; }

app=$(mktemp -d)
trap 'rm -rf "$app"' EXIT
cp -r "$(dirname "$0")/boot-smoke/." "$app"

profile=()
preview=""
if java -version 2>&1 | grep -q '"21\.'; then
    # The published spring-ai-jitllm POM carries jitllm's JDK 22+ artifact; swap in the JDK 21 one,
    # as the Spring AI docs tell JDK 21 users to.
    profile=(-P jdk21)
    preview="--enable-preview"
fi

mvn -B -q -f "$app/pom.xml" "${profile[@]}" package dependency:build-classpath -Dmdep.outputFile="$app/cp.txt"
tr ':' '\n' < "$app/cp.txt" | grep '/jitllm-' | xargs -n1 basename

output=$("$TORNADOVM_HOME/bin/tornado" \
    --jvm="-Duse.tornadovm=true $preview --add-modules jdk.incubator.vector -Dtornado.device.memory=20GB -Xmx8g" \
    -cp "$app/target/classes:$(cat "$app/cp.txt")" demo.DemoApplication 2>&1)
echo "$output" | grep -E '^\[(call|meta|stream|tool|tools)\]' || true

for expected in '^\[call\] .+' '^\[meta\] .*on-gpu=true' '^\[stream\] .+' '^\[tool\] getWeather' '^\[tools\] .*[Ss]unny'; do
    if ! grep -Eq "$expected" <<< "$output"; then
        echo "Smoke app output is missing: $expected" >&2
        echo "$output" | tail -40 >&2
        exit 1
    fi
done
echo "Smoke app passed"

#!/usr/bin/env bash
#
# Regenerate the committed golden logits (T1.4).
#
# Goldens are NEVER regenerated automatically on failure. Running this is a deliberate act, and
# the resulting commit must change nothing else and must say why in its message.
#
# Requires: TORNADOVM_HOME pointing at the pinned SDK, a working device, and the pinned model
# fixtures under $JITLLM_TEST_MODELS (or ~/.jitllm/test-models).
#
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${TORNADOVM_HOME:-}" ]; then
  echo "error: TORNADOVM_HOME is not set" >&2
  exit 1
fi
if [ ! -f "$TORNADOVM_HOME/tornado-argfile" ]; then
  echo "error: no tornado-argfile under $TORNADOVM_HOME" >&2
  exit 1
fi

# Refuse to run against a dirty tree: the generating commit is written into the metadata, so it
# has to actually describe the code that produced the numbers.
if ! git diff --quiet HEAD -- . ; then
  echo "error: working tree has uncommitted tracked changes; commit or stash them first" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

COMMIT=$(git rev-parse HEAD)
echo "generating goldens at commit $COMMIT"

./mvnw -q -B test-compile

CP_FILE=$(mktemp)
trap 'rm -f "$CP_FILE"' EXIT
./mvnw -q -B dependency:build-classpath -Dmdep.outputFile="$CP_FILE" -Dmdep.includeScope=test

CLASSPATH="target/classes:target/test-classes:$(cat "$CP_FILE")"

# recover.bailout=False is mandatory: with the default TRUE a failed kernel silently falls back to
# sequential Java and would produce a wrong golden instead of an error (capability C4).
# jitllm.deviceSample=false keeps the full logits row crossing to the host, which is what is hashed.
# The backend priorities pin CUDA: a multi-backend SDK defaults to OpenCL, and a golden recorded on
# one backend is not the tuple the other one runs. They are no-ops on an OpenCL-only SDK.
java "@$TORNADOVM_HOME/tornado-argfile" \
  --add-modules jdk.incubator.vector \
  -Dtornado.recover.bailout=False \
  -Djitllm.deviceSample=false \
  -Dtornado.device.memory=12GB \
  -Dtornado.cuda.priority=100 \
  -Dtornado.opencl.priority=0 \
  -Dgolden.commit="$COMMIT" \
  -cp "$CLASSPATH" \
  org.beehive.jitllm.golden.GenerateGoldens

echo
echo "Goldens written. Review the diff, then commit them on their own:"
echo "  git add src/test/resources/goldens && git commit"

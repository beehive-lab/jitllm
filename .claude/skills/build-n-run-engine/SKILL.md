---
name: build-n-run-engine
description: Build jitllm with Maven and run it on an accelerator. Use when target/ is missing or stale, after pulling changes, or when switching JDK line or backend.
license: Apache-2.0
metadata:
  author: TornadoVM Team
---

# Build and run jitllm

## When to use

| Situation | Use this skill? |
| --- | --- |
| `target/` is missing, or older than the sources | yes |
| You just switched JDK line or TornadoVM SDK | yes — always `clean` |
| You only need to run an already-built jar | no |
| `TORNADOVM_HOME` is unset or wrong | no — run `build-tornado` first |

## Environment

Two JDK lines are supported and they are not interchangeable. The build refuses anything
else at `validate`, naming both.

| Build JDK | Artifact | TornadoVM SDK it must run against |
| --- | --- | --- |
| 21 | `jitllm:<version>-jdk21` | a TornadoVM SDK built with `make BACKEND=...` (the jdk21 target) |
| 22+ (CI: 25) | `jitllm:<version>-jdk22plus` | a TornadoVM SDK built with `make jdk22plus BACKEND=...` |

There is no flag: the JDK on `JAVA_HOME` selects the profile.

Each Bash tool call is a fresh shell, so export the environment in the **same** command as
the build or run:

```bash
export JAVA_HOME="$HOME/.sdkman/candidates/java/21.0.2-open"
export TORNADOVM_HOME=/path/to/tornadovm-<version>-<backend>
export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"
java -version && tornado --devices
```

Quote every path: some contain spaces.

## Build

```bash
./mvnw clean install                 # includes the Class A gates
./mvnw clean install -DskipTests     # artifact only
make lint                            # Spotless check
make format                          # Spotless apply
```

Always `clean` when the JDK line changes. A `target/` left by the other line fails partway
through with `UnsupportedClassVersionError`, and the message names a test class rather than
the cause.

Accelerator gates are opt-in and need a device, an SDK and the pinned fixtures under
`$JITLLM_TEST_MODELS` or `~/.jitllm/test-models/`:

```bash
./mvnw clean verify -Paccel-tests
```

## Verify the build

```bash
./mvnw help:evaluate -Dexpression=project.version -q -DforceStdout   # must end -jdk21 or -jdk22plus
./jitllm --help
```

## Run

```bash
./jitllm --gpu --model <model.gguf> --prompt "..." -n 128 --seed 42
```

The backend comes from `$TORNADOVM_HOME/etc/tornado.backend`. `--opencl`, `--cuda`
and `--metal` force one when the SDK has several; they error out if the SDK does
not contain the requested backend, and are redundant on a single-backend SDK.

Both launchers take their JVM flags from `$TORNADOVM_HOME/tornado-argfile`. If it is
missing, run `tornado --devices` once to regenerate it from the template. Never add JVMCI,
module-path or preview flags by hand — they differ by TornadoVM version, JDK and backend,
and the SDK is what knows them. `jitllm4j`, the single-file Java launcher, needs JDK 25
to run itself.

### Prove what actually ran

A process that exits 0 at a plausible throughput has proved nothing. Before reporting a
result, establish all three:

```bash
# The launcher parses its own flags, so engine properties go through JAVA_TOOL_OPTIONS,
# which is what CI does too.
JAVA_TOOL_OPTIONS="-Djitllm.metrics.format=json -Djitllm.metrics.output=file -Djitllm.metrics.file=$PWD/run.json" \
  ./jitllm --gpu --model <model.gguf> \
    --prompt "What is the capital of France?" -n 64 --seed 42
```

1. **The backend resolved** — the launcher prints `Detected TornadoVM backend: <name>`.
2. **A plan was built** — `execution_path` in the metrics JSON is `LOWERED` or `LEGACY`. If
   it is absent, no accelerator plan was built and a CPU fallback looks identical to
   success.
3. **The output is right** — the generated text contains what the prompt implies. An
   accelerator computing wrong numbers exits 0 and reports normal throughput.

Add `-Dtornado.recover.bailout=False` whenever correctness matters: with the default, a
failed kernel silently falls back to sequential Java and produces a wrong answer instead of
an error.

## Flags worth knowing

| Flag | Use |
| --- | --- |
| `--model <path>` | required |
| `--prompt "..."` / `-i` | one-shot, or a chat loop |
| `--gpu` | required for acceleration; without it the run is CPU-only |
| `-n`, `--temperature`, `--top-p`, `--seed` | sampling |
| `--gpu-memory 20GB` | raise from the 14GB default for 3B/8B models |
| `--heap-max`, `--heap-min` | JVM heap, default 20g |
| `--server --port N` | the OpenAI-compatible server |
| `--bench --bench-args "..."` | the llama-bench-style harness |

Execution modes (batched prefill is default-off, deliberately):

| Flag | Meaning |
| --- | --- |
| `--with-prefill-decode` | sequential prefill, then single-token decode |
| `--batch-prefill-size N` | with the above, batch N prompt tokens per invocation |
| `--cuda-graphs` | PTX/CUDA only: capture and replay to cut launch overhead |

Diagnostics — for a specific investigation, not for normal runs:

| Flag | Use |
| --- | --- |
| `--verbose-init` | GGUF load, copy-in and JIT timings |
| `--show-command --execute-after-show` | print the exact JVM command first |
| `--print-kernel`, `--print-bytecodes`, `--print-threads`, `--full-dump` | codegen and scheduling detail |
| `--profiler --profiler-dump-dir <FILE>` | TornadoVM profiler output |

`--profiler-dump-dir` takes a **file**, not a directory. Given a directory the run bails
out to sequential Java and looks like a hang.

Run `--help` before using a flag you are not certain about. Keep profiler output, traces
and metrics files outside the repository.

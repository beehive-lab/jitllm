---
name: build-tornado
description: Build a TornadoVM SDK for jitllm. Use when TORNADOVM_HOME is unset, points at the wrong JDK line or backend, or the pinned version has moved.
license: Apache-2.0
metadata:
  author: TornadoVM Team
---

# Build TornadoVM

One SDK per (JDK line, backend). jitllm pins the TornadoVM version in `pom.xml`
as `tornadovm.base.version`; build that tag, not `develop`.

## When to use

| Situation | Use this skill? |
| --- | --- |
| `echo $TORNADOVM_HOME` is empty, or `tornado --devices` fails | yes |
| The SDK's JDK line does not match the JDK you will build jitllm with | yes |
| You need a different backend than the installed SDK has | yes |
| The SDK is fine and you only need to rebuild jitllm | no — use `build-n-run-engine` |

## 1. Decide the tuple

Ask, or read, three things. Do not guess any of them.

```bash
# The pinned version, from the consuming project:
sed -n 's:.*<tornadovm.base.version>\(.*\)</tornadovm.base.version>.*:\1:p' /path/to/jitllm/pom.xml
```

- **Backend**: `opencl`, `cuda` or `metal`. Comma-separated combinations are accepted but
  make the runtime pick between them, so prefer one.
- **JDK line**: 21 or 25. It must be the same JDK you build and run jitllm with.
- **Version**: the pinned tag above.

## 2. Check the environment

```bash
java -version                 # must match the JDK line you chose
nvidia-smi                    # CUDA/OpenCL on NVIDIA
system_profiler SPDisplaysDataType | head    # macOS/Metal
```

## 3. Get a clean source tree at the pinned tag

Use a directory per tuple, so one build does not overwrite another's `dist/`.

```bash
TVM_SRC="$HOME/tornadovm-sdks/jdk21-cuda"     # name it after the tuple
git clone --depth 1 --branch v6.0.0 https://github.com/beehive-lab/TornadoVM.git "$TVM_SRC"
```

## 4. Build

```bash
cd "$TVM_SRC"
export JAVA_HOME="$HOME/.sdkman/candidates/java/21.0.2-open"   # the chosen line
export PATH="$JAVA_HOME/bin:$PATH"

python3 -m venv venv && source venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet requests tqdm    # bin/pull_graal_jars.py imports these

rm -rf graalJars && mkdir -p graalJars         # never seed from a shared directory

make BACKEND=cuda            # JDK 21
# make jdk22plus BACKEND=cuda   # JDK 25 and anything newer
```

Four things that cost time if you get them wrong:

- **`make BACKEND=...` is the JDK 21 target.** It passes `--jdk jdk21`, and `bin/compile`
  refuses a `JAVA_HOME` that is not JDK 21 rather than compiling against one JVMCI shape
  and linking against another. Everything from JDK 22 up is the single `jdk22plus` profile.
- **`make metal` is not Metal-only** — it expands to `--backend metal,opencl`. Use
  `make BACKEND=metal`.
- **Clear `graalJars/` first.** Mixing Graal packagings makes the boot layer refuse to
  start with `LayerInstantiationException: Package org.graalvm.graphio in both module ...`,
  which reports module names and no file.
- **`CUDA_PATH`** must point at a toolkit whose `include/` has `cudnn.h`; otherwise the
  `cudnn-jni` module fails with `fatal error: cudnn.h: No such file or directory`.

## 5. Point the environment at it

```bash
export TORNADOVM_HOME="$(find "$TVM_SRC/dist" -maxdepth 3 -type d -name 'tornadovm-*-cuda' | head -1)"
export PATH="$TORNADOVM_HOME/bin:$PATH"
```

Each Bash tool call is a fresh shell. Export these in the **same** command as anything
that uses them, or source a script that does.

## 6. Verify

```bash
tornado --version
tornado --devices              # must list the device you intend to use
cat "$TORNADOVM_HOME/etc/tornado.backend"
test -f "$TORNADOVM_HOME/tornado-argfile"     # what jitllm's launchers read
```

`tornado --devices` also regenerates `tornado-argfile` from its template, so run it once
after a fresh build. The argfile is the SDK's own record of how to start it — module path,
Graal/JVMCI arrangement, per-backend export lists, and the preview flag where the JDK 21
line needs one. Never hand-write those flags.

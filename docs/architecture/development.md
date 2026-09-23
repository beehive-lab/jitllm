# Development

Building, the TornadoVM setup this project expects, how to add a model family or a backend
capability, and how releases and the downstream integrations fit together.

## JDKs and TornadoVM

Two JDK lines are supported, and each publishes its own artifact against its own TornadoVM
line.

| Build JDK | This project publishes | It compiles against | `--enable-preview` |
| --- | --- | --- | --- |
| 21 | `jitllm:<version>-jdk21` | `tornado-api`/`tornado-runtime` `<tvm>-jdk21` | yes — `java.lang.foreign` is a preview API on 21 |
| 25 | `jitllm:<version>-jdk25` | `tornado-api`/`tornado-runtime` `<tvm>-jdk22plus` | no |

TornadoVM 6.0.0 collapsed its per-version `jdk25`/`jdk26`/`jdk27` profiles into one
`jdk22plus` profile and publishes only `-jdk21` and `-jdk22plus`, so `6.0.0-jdk25` does not
exist. This project's own suffix stays `-jdk25`; the two are separate Maven properties on
purpose, because tying them together would silently rename `jitllm`'s coordinates.

Profile activation is `[21,22)` and `[25,26)`, and `maven-enforcer-plugin` rejects any
other JDK at `validate` with a message naming both lines. Nothing between or beyond the two
can quietly produce a mislabelled artifact.

The JDK 25 artifact contains **no preview class file**, so it is not pinned to one JDK
version. The JDK 21 artifact necessarily does, because the foreign-memory API it uses is
preview there.

The TornadoVM version floor is in `pom.xml` as `tornadovm.base.version`, and CI builds that
exact tag from source. Raising it is a maintainer decision, not a CI tweak.

## Building TornadoVM

```bash
git clone --depth 1 --branch <tag> https://github.com/beehive-lab/TornadoVM.git
cd TornadoVM
python3 -m venv venv && source venv/bin/activate
python -m pip install --quiet requests tqdm      # bin/pull_graal_jars.py needs these
rm -rf graalJars && mkdir -p graalJars           # never seed from a shared directory

export JAVA_HOME=<a JDK 21 or JDK 25 home>
make BACKEND=cuda            # JDK 21 — the default target passes --jdk jdk21
make jdk22plus BACKEND=cuda  # JDK 25 — everything from 22 up is one profile
```

Backends are `opencl`, `cuda` and `metal`, comma-separable. A few things repay knowing:

- **`make BACKEND=...` is the JDK 21 target.** It passes `--jdk jdk21`, and `bin/compile`
  refuses a `JAVA_HOME` that is not JDK 21 rather than compiling against one JVMCI shape
  and linking against another.
- **`make metal` is not Metal-only** — it expands to `--backend metal,opencl`. Use
  `make BACKEND=metal`.
- **Clear `graalJars/` before every build.** Seeding it from a shared directory mixes Graal
  packagings, and the boot layer then refuses to start with
  `LayerInstantiationException: Package org.graalvm.graphio in both module ...`.
- One SDK per (JDK, backend). A build overwrites the previous `dist/` for that tree.

The SDK lands in `dist/tornadovm-<version>-<backend>-<platform>/tornadovm-<version>-<backend>`.
Point `TORNADOVM_HOME` at it and put its `bin` on `PATH`.

## Building this project

```bash
export JAVA_HOME=<21 or 25>
./mvnw clean install                 # Class A gates included
./mvnw clean install -DskipTests     # just the artifact
make lint                            # Spotless check
make format                          # Spotless apply
make test-scripts                    # the Python tooling tests
```

Accelerator gates are opt-in and need a device, an SDK and the pinned fixtures under
`$JITLLM_TEST_MODELS` or `~/.jitllm/test-models/`:

```bash
export TORNADOVM_HOME=/path/to/sdk
./mvnw clean verify -Paccel-tests
```

Always `clean` when switching JDKs: a `target/` left by the other line fails with
`UnsupportedClassVersionError` partway through the suite.

## Running

```bash
./jitllm --gpu --model model.gguf --prompt "..."
```

Both launchers — `jitllm` (Python) and `jitllm4j` (a single-file Java program,
which needs JDK 25 to run itself) — start the JVM from `$TORNADOVM_HOME/tornado-argfile`.
That file is the SDK's own record of how to launch it: module path, Graal and JVMCI
arrangement, per-backend export lists, and the preview flag where the line needs one. All
of that varies by TornadoVM version, JDK and installed backends, so it is read rather than
reproduced. The launchers add only what the argfile does not supply: heap and direct-memory
sizes, `jdk.incubator.vector`, the interpreter bytecode buffer size, the `tornado.*` and
`llama.*` properties, and the backend priorities when an SDK has more than one backend.

If `tornado-argfile` is missing, run `$TORNADOVM_HOME/bin/tornado --devices` once — the
launcher regenerates it from `tornado-argfile.template`.

The backend is detected from `$TORNADOVM_HOME/etc/tornado.backend`. `--cuda`, `--opencl`
and `--metal` force one on a multi-backend SDK and error out if it is not installed.

## Adding a model family

The full workflow — inventory, the decision about whether the existing abstractions express
the model, when to stop and write a design proposal, the implementation order, and the
verification a port must pass — is the `port-model-to-jitllm` skill in
`.claude/skills/port-model-to-jitllm/`. It carries the checklist and the traps previous
families have hit. What follows is the shape of the work.

1. Add a `ModelProvider` that recognizes the GGUF and loads configuration, weights,
   tokenizer and chat format. Recognize by declared `general.architecture`; use
   `general.name` only where the format leaves no other signal.
2. Add a `ModelArchitecture` describing the layer topology and the program.
3. Compose the program from the existing operation vocabulary. If the family genuinely
   needs new arithmetic, add an **operation** — defined once, implemented by every backend
   that claims it — rather than a one-off kernel.
4. Add a `CpuForwardProvider`. The CPU path is the numerical reference, so it comes first.
5. Add a `TornadoPlanProvider`, and a `TornadoLoweringProvider` if the family is to lower.
6. Add a `KvStorageFactory` if its KV layout differs.
7. Register each in `META-INF/services`. Do not add a `switch` — the architecture rules
   forbid one, and CI counts the service files in the shaded jar.
8. Add the family to the CPU↔GPU parity suite. A family whose GPU path is claimed and whose
   parity is not gated is not verified.

## Adding a backend capability

Kernel selection branches on `DeviceCapability`, never on a backend name. Add the
capability, grant it in `TornadoDevices.capabilitiesOf`, and consume it where the kernel is
chosen. A capability that is withheld is withheld with a recorded reason — that is what
makes a divergence reviewable instead of looking like a bug.

## Release and integrations

`prepare-release.yml`, `deploy-maven-central.yml` and `finalize-release.yml` publish both
artifacts. `deploy-maven-central.yml` runs the JDK 21 and JDK 25 legs separately, because
each JDK produces exactly one of the two.

Two integrations consume the facade and are validated against it:

- **LangChain4j** — `langchain4j-gpu-llama3`, selecting `-jdk21`/`-jdk25` from JDK-specific
  profiles.
- **Quarkus LangChain4j** — `model-providers/gpu-llama3/{runtime,deployment}`, with the
  version driven by the root `gpu-llama3.version` property and its `jdk25` profile
  override.

Both must import only `api/**` types. The repository skills
`update-langchain4j-integration` and `update-quarkus-langchain4j-integration` carry the
whole-chain validation each one requires; compilation and focused tests do not complete an
integration update.

CI's `quarkus-langchain4j-integration` job clones the branch named by
`QUARKUS_LANGCHAIN4J_REF` and builds it against the artifact the run just installed, so a
facade change that breaks the extension is visible here rather than downstream.

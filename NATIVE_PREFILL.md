# Native batched prefill (Qwen3 FP16, CUDA)

Run the prompt-ingestion half of a Qwen3 FP16 session on the **vendor libraries** instead of
jllm's own generated kernels: the four projections become single cuBLAS FP16 GEMMs over stacked
weights, and the first prefill chunk's attention becomes one fused cuDNN scaled-dot-product call.

There is **no flag to turn this on.** It is selected automatically whenever the configuration
supports it, and it silently keeps the generated kernels where it does not.

---

## Using it

Nothing new. The existing controls are the whole interface:

```bash
./jllm --gpu --model /path/Qwen3-0.6B-f16.gguf \
       --batch-prefill-size 512 --fp16-kv-cache \
       --prompt "…"
```

- `--batch-prefill-size N` — the prefill chunk width, as before. A multiple of 128 keeps the
  tensor-core tiles full; a prompt shorter than one chunk is fine, a prompt longer than one is
  ingested chunk by chunk.
- `--fp16-kv-cache` — the key/value cache in half precision, as before. **The fused attention
  needs it**; see the selection table.
- `--cuda-graphs` is orthogonal and optional.

Every run prints what it selected:

```
[jllm] prefill acceleration: cuBLAS projections + cuDNN attention
[jllm] prefill attention: cuDNN SDPA for the first chunk, batched JIT paged attention for the
       rest; staging 8 MiB, one set shared by every batch-prefill graph
[jllm] prefill gate/up: cuBLAS, 28 stacked weights, 336 MiB extra resident, built in 0.53 s
[jllm] prefill QKV: cuBLAS, 28 stacked weights (2048|1024|1024 cols), 224 MiB extra resident, built in 0.75 s
[jllm] prefill layer grouping: 4 layers per graph, 7 graphs instead of 28
[jllm] prefill fallback family: 7 graphs, JIT paged attention, all buffers bound from the primary family
```

If the first line says anything other than `cuBLAS projections + cuDNN attention`, it says why.

---

## What gets selected, and when

`NativePrefillSupport` answers two questions once per process, and everything follows from them.

| requires | native projections<br>(QKV, gate/up, output, down) | fused cuDNN attention<br>(first chunk) |
| --- | :---: | :---: |
| tensor-core (CUDA) backend | ● | ● |
| cuBLAS reachable | ● | ● |
| cuDNN **and** `libtornado-cudnn` reachable | | ● |
| FP16 key/value cache for this session | | ● |
| `-Djllm.prefill.native` not set to `false` | ● | ● |

Anything not granted falls back, and the fallbacks are all paths that already existed:

| situation | what runs |
| --- | --- |
| everything above holds | cuBLAS projections; cuDNN attention on the first chunk of a sequence, batched JIT paged attention on every later chunk |
| chunk starts past position 0 | the **fallback family**: the same layer pipeline and the same cuBLAS projections over the same stacked weights, with batched JIT paged attention |
| FP32 key/value cache | cuBLAS projections, JIT attention everywhere |
| cuDNN or its SDPA shim absent | cuBLAS projections, JIT attention everywhere |
| cuBLAS absent, or a non-CUDA backend, or a non-tensor-core device | the generated MMA batch-prefill kernels, exactly as before this change |
| a model family other than Qwen3 FP16 | untouched |

The libraries are probed **lazily and behind the backend gate**, so an OpenCL run, a Metal run,
a CPU run or a single-token plan never dlopens cuBLAS or cuDNN.

### Why the first chunk is special

cuDNN's causal mask aligns query *i* to key *i*. That is the correct mask exactly when the query
block **is** the whole prefix — the first chunk of a sequence, full or partial. A later chunk has
its queries at an offset into a longer key range and needs a bottom-right aligned mask, which
cudnn-frontend has but TornadoVM's `sdpaForward` binding does not expose. Later chunks therefore
take the batched JIT paged attention, which accepts an arbitrary start position and reads the
paged cache in place.

That fallback is not a slow path. It is the same graphs, the same stacked weights and the same
chunk-at-a-time batching; only the attention step differs, and every buffer it uses is bound from
the primary family, so it adds graphs and **no allocation, upload or copy**.

### The one supported switch

```
-Djllm.prefill.native=false
```

turns the whole native path off and runs the generated kernels. It exists as an escape hatch for
a host whose vendor libraries load but misbehave, and it is what the weight-handoff regression
test flips to compare the two paths in one process. There is deliberately **no per-operation
property**: a configuration with only some of these on has never been validated end to end.

It must be set before a session's plan is built and left alone until that plan is discarded — the
batch-prefill graphs and the decode graphs that consume their weights are built in one pass, and
both read it.

---

## Setup: the TornadoVM this needs

**Required revision: `beehive-lab/TornadoVM` at commit `65f06c5d162c36022f9d5724dd38b9784d85cfcc`**
(develop, the merge of PR #1094). This is what `.github/actions/setup-tornadovm` pins and what
`tornadovm.native.version` in `pom.xml` resolves against.

It has to be a **source build**, not an SDKMAN install and not a Maven Central artifact:

```bash
git init TornadoVM && cd TornadoVM
git remote add origin https://github.com/beehive-lab/TornadoVM.git
git fetch --depth 1 origin 65f06c5d162c36022f9d5724dd38b9784d85cfcc
git checkout FETCH_HEAD

python3 -m venv venv && . venv/bin/activate
python -m pip install --quiet --upgrade pip requests tqdm
rm -rf graalJars && mkdir -p graalJars

export JAVA_HOME=…/21.0.2-open           # or a JDK 25 for `make jdk22plus`
export CUDA_PATH=/usr/local/cuda-13.0    # its include/ must have cudnn.h reachable
make BACKEND=cuda

export TORNADOVM_HOME=$PWD/dist/tornadovm-*-cuda-linux-amd64/tornadovm-*-cuda
tornado --devices                        # regenerates tornado-argfile
```

Two things that build does which nothing else will:

1. **It installs `tornado-cublas` and `tornado-cudnn` into your local Maven repository.** Neither
   module is published to Maven Central at any version, and jllm compiles against both. Without
   this step `./mvnw install` fails to resolve them.
2. **It builds `libtornado-cudnn.so`,** the small JNI shim the fused attention path needs —
   cudnn-frontend is a header-only C++ graph API, so there is no C ABI to bind through Panama.
   The shim is what `$TORNADOVM_HOME/lib` must contain; without it the fused attention is not
   selected and the JIT attention runs instead.

### Native libraries, and one performance trap

| | verified on the reference machine |
| --- | --- |
| CUDA toolkit | 13.0 (`nvcc` V13.0.88) |
| driver | 580.142 |
| cuBLAS / cuBLASLt | 13.4.1.2 |
| CUDA runtime | libcudart.so.13 → 13.0.96 |
| cuDNN | 9.24.0 |
| cudnn-frontend (vendored by the shim) | 1.25.0 |

**If a CUDA 12 cuBLAS is installed alongside a CUDA 13 one, stock TornadoVM picks the CUDA 12
one.** Its FFM lookups stop at `libcublas.so.12`, which on such a host resolves an older library
that loads and computes correctly but selects an architecture-agnostic SIMT kernel instead of a
tensor-op one. Measured on an RTX 5070 Ti (sm_120) at 4096³ FP16, identical arguments:
**~21 TFLOP/s against ~100.** Nothing fails; the projections just run at a fifth of the rate.

Check what you actually got:

```bash
ldconfig -p | grep -E 'libcublas\.so\.(12|13)'
```

If both are present, either remove the stale CUDA 12 cuBLAS or apply the three-line SONAME patch
that puts the newest name first in TornadoVM's `CuBlasNativeLib`, `CuBlasLtNativeLib` and
`CuDnnNativeLib` lookups. That patch is **not upstream yet**; the benchmark numbers below were
taken with it.

---

## Scope of the performance claims

Everything measured here is **Qwen3-0.6B FP16 prefill**, on one machine (RTX 5070 Ti, driver
580.142, CUDA 13.0), at the workloads listed, against pinned llama.cpp `e2d2c0d6a` / build
b10874 where that comparison appears. It does **not** generalize to other models, other
quantizations, other GPUs or to decode. A decode measurement is reported separately and
deliberately not folded into any prefill ratio.

<!-- PERF-TABLE -->

### Memory and setup cost

<!-- MEM-TABLE -->

---

## Limitations

- **Qwen3 FP16 only.** The other batch-prefill families are untouched. Nothing here is generic
  across architectures.
- **CUDA only**, and only on a device TornadoVM grants tensor-core MMA.
- **The fused attention covers the first chunk of a sequence only** — see "Why the first chunk is
  special". Reaching later chunks needs a bottom-right causal mask that TornadoVM's cuDNN binding
  does not expose.
- **The FP16 key/value cache is a precondition for the fused attention**, because the adapters
  read K/V out of the paged cache as FP16. An FP32 KV session keeps the native projections and
  loses only the fused attention.
- **A prefill width that is not a multiple of 128 wastes tensor-core rows** in the JIT kernels
  that still run on this path, and a prompt much shorter than the chosen width pays the full
  width in the fused attention, whose sequence lengths are fixed when the graph is built. Match
  `--batch-prefill-size` to the prompt where throughput matters.
- **Requires an unreleased TornadoVM revision**, and two of its modules are not on Maven Central.
  This is a hard build prerequisite, not just a runtime one.
- **The CUDA 13 SONAME fix is not upstream.** On a host with both cuBLAS major versions
  installed, an unpatched TornadoVM silently selects the slower library.

# Native batched prefill (Qwen3 FP16, CUDA)

Run the prompt-ingestion half of a Qwen3 FP16 session on the **vendor libraries** instead of
jllm's own generated kernels: the four projections become single cuBLAS FP16 GEMMs over stacked
weights, and the first prefill chunk's attention becomes one fused cuDNN scaled-dot-product call.

There is **no flag to turn this on.** It is selected automatically whenever the configuration
supports it, and it silently keeps the generated kernels where it does not.

Decode is a different problem and gets no vendor GEMM, but it was optimized and measured on the
same device in the same campaign, so it is recorded here too — see **[Decode on
CUDA](#decode-on-cuda)**.

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
[jllm] prefill acceleration: cuBLAS projections + cuDNN first-chunk attention
[jllm] prefill attention: cuDNN SDPA for the first chunk, batched JIT paged attention for the
       rest; staging 8 MiB, one set shared by every batch-prefill graph
[jllm] prefill gate/up: cuBLAS, 28 stacked weights, 336 MiB extra resident, built in 0.53 s
[jllm] prefill QKV: cuBLAS, 28 stacked weights (2048|1024|1024 cols), 224 MiB extra resident, built in 0.75 s
[jllm] prefill layer grouping: 4 layers per graph, 7 graphs instead of 28
[jllm] prefill fallback family: 7 graphs, JIT paged attention, all buffers bound from the primary family
```

The first line describes **this session**, not the host. It names what was selected and, when
something was not, why — an FP32 key/value cache, an unreachable library, or a fused attention
this shape cannot use:

```
[jllm] prefill acceleration: cuBLAS projections; JIT attention (FP32 key/value cache)
[jllm] prefill acceleration: cuBLAS projections; JIT attention (cuDNN fused SDPA unusable at 16 heads x 128 tokens x 128)
[jllm] prefill acceleration: off (-Djllm.prefill.native=false)
```

---

## What gets selected, and when

`NativePrefillSupport` answers two questions, and everything follows from them.

| requires | native projections<br>(QKV, gate/up, output, down) | fused cuDNN attention<br>(first chunk) |
| --- | :---: | :---: |
| tensor-core (CUDA) backend | ● | ● |
| cuBLAS reachable | ● | ● |
| cuDNN reachable | | ● |
| FP16 key/value cache for this session | | ● |
| **cuDNN builds a fused-SDPA plan for this session's shape** | | ● |
| `-Djllm.prefill.native` not set to `false` | ● | ● |

The last of those is asked by building a real plan for the exact tuple this session would use —
heads, chunk width, head dimension, scale and mask — on the device it would run on, and throwing
it away. A file check would not do: TornadoVM ships `libtornado-cudnn` on every CUDA toolkit, and
below CUDA 12.0 its entry points are stubs that load fine and return a null plan. A real
implementation can also decline a device (Ampere or newer) or a head dimension (a multiple of 8,
at most 256).

Anything not granted falls back, and the fallbacks are all paths that already existed:

| situation | what runs |
| --- | --- |
| everything above holds | cuBLAS projections; cuDNN attention on the first chunk of a sequence, batched JIT paged attention on every later chunk |
| chunk starts past position 0 | the **fallback family**: the same layer pipeline and the same cuBLAS projections over the same stacked weights, with batched JIT paged attention |
| FP32 key/value cache | cuBLAS projections, JIT attention everywhere |
| cuDNN absent, or its fused SDPA absent, stubbed or unusable for this shape | cuBLAS projections, JIT attention everywhere |
| cuBLAS absent, or a non-CUDA backend, or a non-tensor-core device | the generated MMA batch-prefill kernels, exactly as before this change |
| a model family other than Qwen3 FP16 | untouched |

The libraries are probed **lazily and behind the backend gate**, so an OpenCL run, a Metal run,
a CPU run or a single-token plan never dlopens cuBLAS or cuDNN. The fused-SDPA answer is cached
per shape, so a second session of the same width does not rebuild a plan to ask again.

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
```

Then resolve the SDK directory — a shell does **not** expand a glob on the right-hand side of an
assignment, so `export TORNADOVM_HOME=$PWD/dist/tornadovm-*-cuda` sets a literal string with an
asterisk in it and every later step fails somewhere unhelpful:

```bash
TORNADOVM_HOME=$(find "$PWD/dist" -maxdepth 3 -type d -name 'tornadovm-*-cuda' | head -n 1)
[ -d "$TORNADOVM_HOME" ] || { echo "no CUDA SDK under $PWD/dist" >&2; return 1 2>/dev/null || exit 1; }
export TORNADOVM_HOME
export PATH="$TORNADOVM_HOME/bin:$PATH"
tornado --devices                        # regenerates tornado-argfile; must list your GPU
test -f "$TORNADOVM_HOME/tornado-argfile"
test -f "$TORNADOVM_HOME/lib/libtornado-cudnn.so"
```

Two things that build does which nothing else will:

1. **It installs `tornado-cublas` and `tornado-cudnn` into your local Maven repository.** Neither
   module is published to Maven Central at any version, and jllm compiles against both. Without
   this step `./mvnw install` fails to resolve them.
2. **It builds `libtornado-cudnn.so`,** the small JNI shim the fused attention path needs —
   cudnn-frontend is a header-only C++ graph API, so there is no C ABI to bind through Panama.
   On a toolkit older than CUDA 12.0 that shim still builds and still loads, but its four entry
   points are stubs that report the feature as unavailable. jllm detects that by asking cuDNN to
   build a plan for the shape it wants and throwing it away, so a stubbed or declining
   installation selects the JIT attention instead of failing mid-request.

### Which libraries a run actually loads

`ldconfig -p` lists what the cache would offer; it is **not** evidence about a running process.
`LD_LIBRARY_PATH`, an `RPATH`, or a second copy under `/usr/local/cuda-*/lib64` all change the
answer, and on this project's reference machine they do: `ldconfig` reports cuBLAS 13.4.1.2, and
the same command run with `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64` — which is what the
benchmark harness sets — maps 13.1.1.3 instead.

Ask the process:

```bash
./jllm --gpu --model … --batch-prefill-size 512 --fp16-kv-cache --prompt hi &
JPID=$!
until grep -q libcublas /proc/$JPID/maps 2>/dev/null; do sleep 1; done
grep -oE '/[^ ]*lib(cublas|cublasLt|cudnn|cudart|tornado-cudnn)[^ ]*\.so[^ ]*' /proc/$JPID/maps | sort -u
```

On the reference machine, with `LD_LIBRARY_PATH` set to the CUDA 13.0 toolkit, that prints:

```
…/tornadovm-6.1.1-jdk21-dev-cuda/lib/libtornado-cudnn.so
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.24.0                 (+ its engine libraries)
/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcublas.so.13.1.1.3
/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcublasLt.so.13.1.1.3
/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13.0.96
```

**Every measurement in this document was taken against those libraries.** CUDA toolkit 13.0
(`nvcc` V13.0.88), driver 580.142, cudnn-frontend 1.25.0 vendored by the shim.

### The SONAME lookup, and what is upstream

Stock TornadoVM at the pinned revision asks for `libcublas.so.12` before `libcublas.so.13`:

```java
// tornado-cublas/…/provider/CuBlasNativeLib.java
FFMSupport.loadLibrary("libcublas.so.12", "libcublas.so.11", "libcublas.so", …)
```

Candidates are tried in order and the first that `dlopen`s wins, so on a host where a CUDA 12
cuBLAS is installed alongside a CUDA 13 one the 12 is selected. It loads and **computes correct
results**; what it can lose is the kernel. Measured on this machine, 4096³ FP16 with identical
arguments: **cuBLAS 12.0.2.224 — a January 2023 build whose newest cubin is sm_90 — selects
`magma_sgemmEx_kernel` at ~21 TFLOP/s on an sm_120 device, where cuBLAS 13 selects a CUTLASS
tensor-op kernel at ~100 TFLOP/s.** That figure is about **that** library. Blackwell support
arrived during the 12.x series, and a newer 12.x has the kernels; nothing here measured one.

**This is not part of the upstream requirement.** jllm builds and runs correctly against stock
`65f06c5d1`. The three-line change below only reorders the candidates, and it is what the
benchmark environment used:

```diff
--- a/tornado-cublas/src/main/java/uk/ac/manchester/tornado/cublas/provider/CuBlasNativeLib.java
+++ b/tornado-cublas/src/main/java/uk/ac/manchester/tornado/cublas/provider/CuBlasNativeLib.java
-    private static final SymbolLookup LIBCUBLAS = FFMSupport.loadLibrary("libcublas.so.12", "libcublas.so.11", "libcublas.so", "cublas64_12.dll", "libcublas.dylib");
+    private static final SymbolLookup LIBCUBLAS = FFMSupport.loadLibrary("libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so", "cublas64_13.dll", "cublas64_12.dll", "libcublas.dylib");
-    private static final SymbolLookup LIBCUDART = FFMSupport.loadLibrary("libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "cudart64_12.dll", "libcudart.dylib");
+    private static final SymbolLookup LIBCUDART = FFMSupport.loadLibrary("libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "cudart64_13.dll", "cudart64_12.dll", "libcudart.dylib");
--- a/tornado-cublas/src/main/java/uk/ac/manchester/tornado/cublas/provider/CuBlasLtNativeLib.java
+++ b/tornado-cublas/src/main/java/uk/ac/manchester/tornado/cublas/provider/CuBlasLtNativeLib.java
-    private static final SymbolLookup LIBCUBLASLT = FFMSupport.loadLibrary("libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so", "cublasLt64_12.dll", "libcublasLt.dylib");
+    private static final SymbolLookup LIBCUBLASLT = FFMSupport.loadLibrary("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so", "cublasLt64_13.dll", "cublasLt64_12.dll", "libcublasLt.dylib");
--- a/tornado-cudnn/src/main/java/uk/ac/manchester/tornado/cudnn/provider/CuDnnNativeLib.java
+++ b/tornado-cudnn/src/main/java/uk/ac/manchester/tornado/cudnn/provider/CuDnnNativeLib.java
-    private static final SymbolLookup LIBCUDART = FFMSupport.loadLibrary("libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "cudart64_12.dll", "libcudart.dylib");
+    private static final SymbolLookup LIBCUDART = FFMSupport.loadLibrary("libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "cudart64_13.dll", "cudart64_12.dll", "libcudart.dylib");
```

Apply it to the TornadoVM checkout before `make`, save it as `cuda13-soname.patch` and:

```bash
cd TornadoVM && git apply cuda13-soname.patch && make BACKEND=cuda
```

**If you would rather not patch**, point the loader at the toolkit you want and check the result
with `/proc/<pid>/maps` as above:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

That only helps if the directory has no `libcublas.so.12`, since the unpatched lookup asks for
that name first. **Do not remove a system CUDA 12 library to get there** — other software on the
host links against it, and this is a lookup-order problem rather than a reason to break them.

## Scope of the performance claims

Everything measured here is **Qwen3-0.6B FP16**, on one machine (RTX 5070 Ti, driver 580.142,
CUDA 13.0), at the workloads listed, against pinned llama.cpp `e2d2c0d6a` / build b10874 where
that comparison appears. It does **not** generalize to other models, other quantizations or other
GPUs. Prefill and decode are measured separately and neither is ever folded into the other's
ratio; this section is prefill, and [Decode on CUDA](#decode-on-cuda) is decode.

### Measured

RTX 5070 Ti (sm_120), driver 580.142, CUDA 13.0, cuBLAS/cuBLASLt 13.1.1.3, cuDNN 9.24.0, JDK
21.0.2-open, TornadoVM `65f06c5d1` with the SONAME reordering above.
Qwen3-0.6B FP16. Four rounds, variant order rotated, six full-workload repetitions per process
with the first reserved as an untimed warmup, five measured; llama.cpp gets one discarded
complete `llama-bench` invocation before each measured one. Peak foreign device occupancy was
0 MiB on all 72 invocations, so nothing was excluded. Every figure is the **median** of 20
retained samples.

**Against pinned llama.cpp `e2d2c0d6a` / build b10874**, same model file, `-fa on -ctk f16
-ctv f16`:

| workload | jllm t/s | llama.cpp t/s | ratio | per round |
| --- | ---: | ---: | ---: | --- |
| pp512, `--batch-prefill-size 512` | 55,476 | 47,131 | **1.177×** | 1.177 / 1.190 / 1.178 / 1.169 |
| pp300, `--batch-prefill-size 300` | 43,662 | 39,291 | **1.111×** | 1.114 / 1.122 / 1.099 / 1.116 |
| pp300, `--batch-prefill-size 512` | 33,369 | 39,317 | 0.849× | 0.848 / 0.849 / 0.851 / 0.848 |

The third row is the padding cost made explicit: a 512-wide chunk does 512 rows of work for 300
tokens. Match the width to the prompt.

llama.cpp's spread on these rows is 10–14%, and it is **entirely its first measured repetition**,
which lands ~24% low in every round despite the discarded invocation before it. jllm's warmup is
in-process, so all five of its samples are warm. Medians are used for that reason, and they are
the conservative choice: llama.cpp's warm-only median differs from its all-sample median by 0.2%.

**Multi-chunk prompts, against `-Djllm.prefill.native=false`** — the generated-kernel path this
replaces, with both the JIT projections and the JIT attention:

| workload | native t/s | generated-kernel t/s | ratio | per round |
| --- | ---: | ---: | ---: | --- |
| pp360, width 128 (1 native chunk of 3) | 13,622 | 6,001 | **2.270×** | 2.283 / 2.239 / 2.253 / 2.277 |
| pp700, width 256 (1 of 3) | 9,496 | 6,308 | **1.505×** | 1.506 / 1.505 / 1.509 / 1.507 |
| pp900, width 128 (1 of 8) | 6,226 | 3,901 | **1.596×** | 1.593 / 1.593 / 1.596 / 1.597 |

**Decode is measured separately and never folded into a prefill ratio.** Enabling the native
prefill path does not change it — 320.6 t/s against 317.0 with `-Djllm.prefill.native=false`, a
1.1% difference whose sign is consistent across four rounds but which sits within ~1.5 standard
deviations of this machine's spread and is **not** claimed as an improvement. The decode path was
then optimized in its own right; those numbers are under [Decode on CUDA](#decode-on-cuda).

### Memory and setup cost

Peak device occupancy, maximum over all rounds, and the same workload with the native path
switched off:

| workload | native | generated kernels | delta |
| --- | ---: | ---: | ---: |
| pp360, width 128 | 2,548 MiB | 1,824 MiB | +724 MiB |
| pp700, width 256 | 2,568 MiB | 1,870 MiB | +698 MiB |
| pp900, width 128 | 2,604 MiB | 1,880 MiB | +724 MiB |
| tg128, width 512 | 2,538 MiB | 1,832 MiB | +706 MiB |

For reference, llama.cpp peaks at 1,360 MiB on pp512 and 1,490 MiB on tg128.

The cost is dominated by the two stacked weight sets, which are **copies**, not views: cuBLAS
takes one B operand and the binding has no operand offset, so `[q|k|v]` and `[gate|up]` have to
be contiguous. The originals stay resident because decode and the generated-kernel path still
read them.

| | Qwen3-0.6B, 28 layers |
| --- | ---: |
| stacked `[gate\|up]` | 336 MiB |
| stacked `[q\|k\|v]` | 224 MiB |
| cuDNN staging quartet | 8 MiB at width 512, 2 MiB at width 128 |
| fallback family | +4 MiB — 7 graphs, no weight, workspace or cache duplicated |
| **logical total** | **≈ 572 MiB** |

The rest of the ~700 MiB is TornadoVM's per-buffer reservation granularity.

**Setup cost: ≈ 1.0–1.3 s, once per execution plan** — 0.44–0.53 s to stack gate/up and
0.52–0.79 s to stack QKV, both printed by the banner. Not per request and not per chunk. The
fallback family adds none: it is built from arrays that already exist.

The capability probe adds to that, measured on the same machine: **56 ms** the first time a
process asks about a fused-attention shape — that call also loads the JNI shim and creates the
cuDNN handle — and **18 ms** for each further distinct chunk width. A width already asked about is
answered from cache in 0.02 ms, so a second plan at the same width costs nothing. The cuBLAS
availability question is answered during TornadoVM's runtime initialization, which a plan build
performs regardless.

Verified from `-Dtornado.print.bytecodes`: 243 distinct (graph, object) allocations across the
seven primary graphs, **none duplicated**, and the fallback family allocates three small objects
and copies nothing.

---

## Decode on CUDA

Decode emits one token at a time, so every projection is a matrix-vector product. A Qwen3-0.6B
FP16 token streams **1151 MiB** of weights, which at this card's ~896 GB/s is a 1.286 ms floor.
No vendor GEMM applies at one column; what is left to win is how the kernels reduce and how often
the host waits for the device. Three changes, all selected automatically:

- **Four transformer layers to a decode graph.** Every `TaskGraph.execute()` ends in a device
  wait, and a profile put **29.8 gaps per token at a median of 22.8 µs — 97% of all device idle**,
  one per graph submission. Twenty-eight layer graphs became seven, and a token's submissions went
  from thirty to nine. Four rather than more: `TornadoTaskGraph` holds a fixed `byte[8192]` of
  high-level bytecode per graph — *not* the buffer `tornado.tvm.maxbytecodesize` sizes — and
  fourteen layers overflow it with a hard throw at plan construction. Seven fits and is 1.2%
  faster; four keeps headroom for a family with a slightly richer layer.
- **A warp-butterfly reduction for four layer matrix-vector kernels**, under a new
  `DeviceCapability.WARP_SHUFFLE_GEMV_FP16`. The shuffle-reducing twins already existed but were
  reachable only through `WARP_SHUFFLE`, which asserts shuffle *correctness*, is branched on by
  unrelated call sites, is miscompiled on the OpenCL backend, and carries a contrary measurement
  from a different GPU on a different model. The new grant is the narrow claim actually measured:
  *these five FP16 GEMV kernels reduce faster with a 32-lane butterfly on this device class.*
- **The vocabulary projection joins it.** Once the layer kernels moved it was the largest single
  kernel in a token — 502 µs, 20.6% of GPU time, 311 MiB of the 1151 — and its shuffle-reducing
  twin existed with a matching worker grid but was gated to Metal. It now runs at 842 GB/s,
  against 619 before.

None of the three allocates anything: same buffers, same tasks, fewer graphs, a different
reduction. Peak device memory is identical to the MiB before and after them.

### Measured

Same machine and protocol as above. Four interleaved rounds in rotated order, five timed
repetitions after an untimed full-workload warm-up, medians over twenty samples; all 24
workload-rounds passed the foreign-occupancy exclusion.

| tg128, batch 512 | before | after | llama.cpp b10874 | vs. before | vs. llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth 0 | 322.2 | **412.5** | 500.9 | **1.28×** | 0.82× |
| depth 512 | 274.0 | **337.6** | 483.8 | **1.23×** | 0.70× |
| depth 2048 | 191.3 | **221.5** | 436.6 | **1.16×** | 0.51× |

Sample spread was ±0.1–0.3%, and per-round ratios against llama.cpp agree to the third decimal.

**Decode is still slower than llama.cpp at every depth.** This took 0.68 ms off a 3.10 ms token at
depth zero; matching llama.cpp needed 1.11 ms. What remains is concentrated in attention — 374 µs
against llama.cpp's 200 for `flash_attn_ext_vec` plus its combine — which is why the shortfall
grows with context: 0.43 ms/token at depth 0, 0.90 at 512, 2.22 at 2048. It is the kernel and not
the partitioning policy: the existing eight split-KV partitions measured best at every depth
tried, so closing the rest means a different attention kernel.

### Correctness

Grouping is **bit-identical** over 64 teacher-forced decode-step logit vectors on two shapes,
against a build with grouping set back to one layer per graph. The two reduction changes land at
relative L2 3.3e-04 to 4.5e-04, which is the ordinary consequence of summing in a different order.
`Qwen3DecodeDispatchAccelTest` asserts that a built plan actually grouped its layer graphs and
actually selected the shuffle-reducing kernels, reading both off the plan's own grid scheduler, so
a silent fallback fails instead of being measured as though it had not happened.

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
- **The decode changes are narrower still.** Layer grouping is applied to the Qwen3 FP16 decode
  families only, and four layers per graph is a headroom choice against a fixed per-graph bytecode
  buffer rather than a tuned optimum. The warp-butterfly reduction is granted on CUDA because that
  is where it was measured; it is deliberately not the broader `WARP_SHUFFLE` grant, and an
  OpenCL or Metal device reaches these kernels, or does not, through its own capabilities.
- **Decode remains slower than llama.cpp** at every depth measured, and the shortfall grows with
  context. See [Decode on CUDA](#decode-on-cuda).
- **Requires an unreleased TornadoVM revision**, and two of its modules are not on Maven Central.
  This is a hard build prerequisite, not just a runtime one.
- **The SONAME reordering is not upstream.** jllm builds and runs correctly without it; on a host
  that also has a CUDA 12 cuBLAS installed, an unpatched lookup selects that one, which is
  correct but can be much slower on a recent device. Check what your process actually mapped
  rather than what `ldconfig` advertises.

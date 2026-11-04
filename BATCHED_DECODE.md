# Static batched decode (LLaMA / Qwen3 FP16)

Decode **B independent sequences at once**, one token per step, each attending its
**own** KV cache. Turns the bandwidth-bound single-token matvecs of decode into
compute-bound tensor-core GEMMs (one weight read amortized across B tokens), so
aggregate throughput scales ~linearly with B until the 128-row MMA tile fills.

Measured up to **41× aggregate throughput** vs single-stream decode on an RTX 4090
(Llama-3.2-1B FP16), output verified coherent + bit-exact against the single-stream
greedy reference. Runs on the TornadoVM **CUDA backend** (tensor-core MMA).
Both **LLaMA** and **Qwen3** are supported (Qwen3 adds per-head Q/K RMS norm + split-
half RoPE).

Two ways to use it:

- **`jitllm serve --continuous-batching SLOTS`** (experimental) decodes up to `SLOTS`
  HTTP requests in one batch — see [Reproduce](#reproduce).
- **`BatchedDecodeEngine`** (`org.beehive.jitllm.backend.tornado.bench`), the benchmark
  harness every number below comes from, configured with a `BatchDecodeOptions`.

The measurements in this document were taken with that harness on TornadoVM 5.0.1 (JDK 21,
RTX 4090), before the move to TornadoVM 7.3; they are kept as recorded.

---

## Why batching wins decode

Single-token decode is **memory-bound**: every projection reads the full weight
matrix to produce one output vector (arithmetic intensity ~1). The GPU stalls on
HBM bandwidth; tensor cores sit idle.

Batch B tokens that share the same weights and each weight row is read **once** and
applied to **all B** tokens — arithmetic intensity ~B, i.e. a GEMM. Past B≈16 the
projections become compute-bound and the tensor cores engage. Attention does *not*
batch this way (each sequence has its own KV), but it is a small fraction of the
decode FLOPs, so the projection win dominates.

```
             single-token decode              static batched decode (B)
             ────────────────────             ─────────────────────────
  weight W    read once  ─┐                     read once ─┐
  token x0    ───────────┼─> y0                 ──────────┼─> y0
  token x1    (next step: read W again)         ──────────┼─> y1     one GEMM,
  token x2    (read W again) ...                ──────────┼─> y2     W read once
   ...                                          ──────────┘  ...
  intensity ~1  (HBM-bound)                     intensity ~B (compute-bound)
```

---

## Design

### Engine loop (`backend/tornado/bench/BatchedDecodeEngine.java`)

One step routine serves **both** phases. Prompt tokens are fed with logits
discarded — this fills every slot's KV region with the *same* RoPE the decode step
uses, so there is no prefill/decode cache mismatch. Then generated tokens feed back.

```
  prompt "…paint."   ┌──────────────────────────────────────────────┐
        │            │  for each step:                              │
        ▼            │    host: write B token embeddings → embXBatch │
  ┌───────────┐      │          write positions → seqPositions[B]    │
  │  PREFILL  │      │    GPU:  activation → L0 → L1 → … → L15 → logit│
  │ P tokens  │──────│    host: sample B rows of logitsBatch         │
  │(no logits)│      └──────────────────────────────────────────────┘
  └───────────┘                        │
        │                              ▼
        ▼                      ┌────────────────┐
  ┌───────────┐   per step     │  B next tokens │  greedy → all B identical
  │  DECODE   │───────────────>│  (one / slot)  │  temp>0 → B divergent streams
  │ N steps   │                └────────────────┘
  └───────────┘
```

### Per-step task-graph pipeline (2 + N graphs)

```
 embXBatch(FP16 B×dim)
      │  [0] prefillActivation      FP16 → FP32  wrapXBatch
      ▼
 ┌─────────────────  [1..N] batchDecodeLayer_i  (12 tasks, all GEMMs on MMA) ─────────────────┐
 │  batch_attn_rms → batch_attn_rms_apply → qkvProj(MMA) → batch_rope_kv* → batch_attention*  │
 │  → woProj(MMA) → batch_ffn_rms → batch_ffn_rms_apply → gateUpProj(MMA) → swiglu            │
 │  → w2Proj(MMA) → w2Resid                                                                    │
 │      (*) the only decode-specific kernels: per-slot position + per-slot KV base            │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
      │  wrapXBatch (B×dim, persists to next layer)
      ▼
 [N+1] batchLogits    final RMS → gemmMMA over vocab(128256) → logitsBatch (B×vocab, FP32)
```

The 12-task layer pipeline is **identical** to the existing batched-prefill MMA
layer (`LlamaFP16LayersBatchPrefillMMA`). Only two kernels change, and only in KV
addressing:

| kernel | prefill (one sequence) | decode (B sequences) |
|--------|------------------------|----------------------|
| RoPE + KV write | `batchedRopeWithKVCachePacked` — `pos = start + b`, one shared cache | `batchedDecodeRopeWithKVCachePacked` — `pos = seqPositions[b]`, per-slot base |
| flash attention | `batchedFlashAttentionFP16Out` — shared causal KV | `batchedDecodeAttentionFP16Out` — own KV region per slot |

The math (RoPE rotation, online-softmax flash attention, register-partitioned P·V)
is untouched; both forks only change the index into the KV cache.

**Qwen3** reuses the same pattern: `Qwen3FP16LayersBatchDecodeMMA` keeps the per-head
Q/K RMS-norm task (`batchedFusedQKRmsNormPacked`, per-token — reused as-is) and swaps in
`batchedDecodeRopeWithKVCacheQwen3Packed` (split-half RoPE, per-slot) +
`batchedDecodeAttentionFP16Out` (the same per-slot flash kernel, parameterized by
`qDim`/`nEmbdHead`/`gqa`). The engine dispatches on model type; embeddings and the
batched-logits GEMM are model-agnostic.

### KV cache layout (B-sized)

Each slot owns a contiguous region; `batchIdx` stride = `numLayers·ctx·kvDim`.

```
 keyCacheBatch  [ slot0 | slot1 | … | slot B-1 ]      size = B · numLayers · ctx · kvDim
                    │
                    └─ [ layer0 | layer1 | … ]         stride numLayers·ctx·kvDim
                          │
                          └─ [ pos0 | pos1 | … ]       row = pos·kvDim, written at seqPositions[slot]
```

Because the engine (not `State`) owns this buffer, `ctx` is capped independently
(default 512) to keep `B·L·ctx·kvDim` in VRAM.

### Static batching

All slots advance one position per step, so `seqPositions` is the same scalar for
every slot at every step. The load-bearing part is the per-slot KV **base** address,
not the position — the `IntArray seqPositions` keeps the door open for ragged /
continuous batching below.

---

## Correctness

Greedy sampling with B copies of the same prompt → **all B streams are bit-exact
identical AND equal to the single-stream greedy reference**. Each slot independently
reproduces the reference, so the per-slot forward (including per-slot KV) is correct.

```
[verify] mode=greedy: all 128 streams identical (== single-stream greedy ref): true
```

Temperature sampling with a per-slot RNG → **B distinct, individually coherent**
continuations, proving the KV regions are genuinely independent (real concurrent
requests, not replication):

```
[verify] mode=sample temp=0.70: 128/128 streams distinct (independent per-slot KV)
  slot 0: … a robot named Zeta stood at a workbench …
  slot 1: … Zeta sat in the studio, its mechanical arms splayed …
  slot 2: … a robot named Zeta whirred to life …
```

---

## Performance

RTX 4090, TornadoVM 5.0.1 **CUDA backend**, FP16, `ctx=512`, CUDA graphs on, steady-state
(capture steps excluded).

**Llama-3.2-1B** (16 layers, vocab 128256):

| config | aggregate tok/s | vs single-stream | per-step latency |
|-------:|----------------:|-----------------:|-----------------:|
| single-stream (stock decode) | 101 | 1.0× | — |
| batched B=32  | 1085 | 10.7× | 29.5 ms |
| batched B=64  | 2167 | 21.5× | 29.5 ms |
| batched B=128 | **4175** | **41.3×** | 30.7 ms |

CUDA graphs (`cudaGraphs`, default on) cut a uniform ~6% off per-step
(dispatch overhead across the 2+N graphs).

**Across models** (largest B that fits 24 GB; deeper/wider → higher per-step, so B is
capped lower):

| model | layers · dim · vocab | single-stream | batched (B) | aggregate | speedup |
|-------|----------------------|--------------:|:-----------:|----------:|--------:|
| Llama-3.2-1B | 16 · 2048 · 128256 | 101 tok/s | 128 | 4175 tok/s | 41× |
| Qwen3-1.7B   | 28 · 2048 · 151936 |  48 tok/s |  64 | 1433 tok/s | 30× |
| Qwen3-4B     | 36 · 2560 · 151936 |  39 tok/s |  32 |  405 tok/s | 10× |

All bit-exact vs the single-stream greedy reference (`all B streams identical: true`)
and coherent.

**Operating point.** Per-step latency is ~flat (29–31 ms) from B=8 to B=128: every
step runs at `paddedBatch = 128` (the MMA tile is 128 rows) and the logits GEMM spans
the full 128256 vocab regardless of B. So B<128 does the same GPU work as B=128
(wasted pad); aggregate scales ~linearly up to B=128, then per-step grows.
**B=128 is the efficient operating point.** The residual per-step cost is real GEMM
compute — full-vocab logits (~33 GFLOP) + 16 layer projections at 128 rows — not
launch overhead, which is why CUDA graphs help only marginally.

---

## Continuous batching (iteration-level scheduling)

`continuous` runs an Orca-style scheduler over the same B slots:
each slot is independently either **prefilling** its prompt (token-by-token, logits
ignored) or **decoding**; a slot that hits a stop token / its max-gen is evicted and
**immediately refilled** from a pending queue — new requests join a partly-decoded
batch mid-flight, so no slot waits for the slowest in a wave. This works with zero
kernel changes because the per-step forward already feeds one token per slot at its own
`seqPositions[b]` with its own KV region — prefill and decode are the same op.

Same 512-request workload (Llama-1B, B=128, prompt 22 tok, max-gen ∈ [8,64], greedy —
so identical token counts and all outputs mutually prefix-consistent):

| scheduling | steps | gen tok/s | requests/s | slot utilization |
|------------|------:|----------:|-----------:|-----------------:|
| static wave (no refill) | 336 | 1645 | 46.9 | 66.4% |
| **continuous** (refill) | **272** | **1972** | **56.2** | **82.2%** |

**+20% throughput / +24% relative utilization** by refilling freed slots instead of
draining each wave to its longest request. The gap widens with more length variance.
Correctness under scheduling = all completed outputs are mutually prefix-consistent.

## PagedAttention (`paged`, LLaMA + Qwen3)

The contiguous KV cache reserves the full `ctx` per slot — `B·ctx` token-slots even
when most sequences are short. Paging stores KV in a **shared pool of fixed-size blocks**
(`blockSize` positions each) addressed through a per-slot **block table**; the two decode
KV kernels index `blockTable[b][pos/blockSize]` instead of `pos·kvDim`. Blocks are
allocated on demand as a sequence grows and returned to the pool on eviction, so the pool
is sized to **actual concurrent demand**, not the worst case.

Same 512-request continuous workload (Llama-1B, B=128, ctx=512, blockSize=16):

| KV cache | pool | peak used | throughput | correct |
|----------|-----:|----------:|-----------:|:-------:|
| contiguous | 4096 MB (=B·ctx reservation) | — | 1972 tok/s | ✓ |
| paged, 768-block pool | 768 MB | 372 blk (48%) | 1971 tok/s | ✓ |
| paged, 384-block pool | 384 MB | 372 blk (97%) | 1939 tok/s | ✓ |

**~10.7× less KV memory** (384 MB vs 4096 MB) at ~1% throughput overhead, output still
bit-exact. The pool floor is the peak concurrent block demand (~372 here); undersizing it
throws a clear `KV block pool exhausted` (the point where admission control / backpressure
takes over — the vLLM behavior). One reserved *scratch* block absorbs the KV writes of
inactive slots (they still execute the kernels each step) so they never corrupt a live
block. Paging is the prerequisite for prefix caching (share a prompt's blocks across
requests via the block table + refcounting).

## Prefix caching (`prefixCache`, needs `paged`)

When requests share a common prompt prefix (a system prompt, a few-shot preamble), its KV
is identical across all of them. Prefix caching prefills the block-aligned shared prefix
**once** into pinned blocks; every request points its block-table prefix rows at those
shared blocks and starts decoding at `pos = sharedPrefixLen`, skipping the prefix's prefill
entirely. The attention kernel walks the block table, so it reads shared-then-private blocks
transparently — **no kernel change**, just scheduling + block-table setup.

Same 512-request continuous+paged workload, 48-token shared prompt (Llama-1B, B=128):

| | steps | gen tok/s | requests/s | prefill tokens |
|--|------:|----------:|-----------:|---------------:|
| no prefix cache   | 419 | 1307 | 37.2 | 28672 |
| **prefix cache**  | **211** | **2422** | **69.0** | 4096 (**85.7% saved**) |

**~2× fewer steps, +85% throughput** — the 48-token prefix is prefilled once (3 pinned
blocks) instead of 512×, and its KV blocks are shared. Output still bit-exact (all completed
requests mutually prefix-consistent). The win scales with prefix length / request count. The
prefix is block-aligned down (`sharedPrefixLen = ⌊(promptLen-1)/blockSize⌋·blockSize`) so a
request only ever writes into its own private blocks; shared blocks stay read-only.

The full stack (batched decode → continuous → paging → prefix caching) runs on **Qwen3**
as well (Qwen3 adds a split-half paged RoPE kernel; the paged flash kernel is shared since
`qDim == nHeads·nEmbdHead`). Qwen3-1.7B, 256-request continuous+paged workload, 48-tok
shared prompt:

| Qwen3-1.7B | steps | gen tok/s | requests/s | peak KV blocks |
|--|------:|----------:|-----------:|---------------:|
| paged, no prefix cache | 405 | 465 | 13.6 | 296 |
| **paged + prefix cache** | **193** | **949** | **27.7** | **151** |

+104% throughput and prefix sharing also halves peak KV blocks (296→151), all bit-exact.

## On-device sampling (`deviceSample`, default on)

Profiling (nsys) showed the engine was **host-bound**: each step copied the full
`paddedB×vocab` logits tensor D2H (~65 MB on Llama, ~78 MB on Qwen) and ran a CPU argmax over
it, stalling the GPU between steps. `batchedArgmaxLogits` (one workgroup per row) does the
greedy argmax **on the GPU**; the logits stay device-side and only **B token ids** cross to the
host.

Llama-1B, B=128, 512-request continuous+paged+prefix, identical greedy output:

| sampling | D2H/step | wall | gen tok/s |
|----------|---------:|-----:|----------:|
| host | ~65 MB | 7.66 s | 2344 |
| **on-device** | ~0.5 KB | 5.88 s | **3057** |

**+30% throughput** — the D2H copy collapses (115 ms → 0.03 ms in the trace) and the CPU scan
is gone; GPU kernel work per step is unchanged, so the gain is pure host-stall removal. Greedy
only: temperature sampling still needs logits on the host, so a configuration with a temperature
sets `deviceSample` to `false`.

## Reproduce

### Build

You need a TornadoVM SDK with the **CUDA backend**, one with `TaskGraph.consumeFromUnifiedMemory` (beehive-lab/TornadoVM#1315), at
`TORNADOVM_HOME` (its JDK line must match the JDK you build with; see the README), and an NVIDIA
GPU with tensor cores. Then:

```bash
export TORNADOVM_HOME=/path/to/tornadovm-sdk
export PATH="$TORNADOVM_HOME/bin:$PATH"
./mvnw clean package -DskipTests
```

### Serve: continuous batching over HTTP

```bash
./jitllm serve --gpu -m Llama-3.2-1B-Instruct-f16.gguf --continuous-batching 16
```

`--continuous-batching SLOTS` (at least 2) decodes up to `SLOTS` requests together;
`--max-queued-requests N` (default 64) bounds how many wait for a slot, and
`--prefix-cache-entries N` (default 0, off) shares cached prompt prefixes between requests. It
is experimental and narrower than the regular path: CUDA tensor-core devices, FP16 Llama/Qwen3
weights and greedy requests (temperature 0) only, and it refuses `--with-prefill-decode`,
`--batch-prefill-size`, `--cuda-graphs` and `--with-native-libraries`.

### Benchmark: `BatchedDecodeEngine`

The harness has no `main`; call it from Java with the model loaded on TornadoVM:

```java
Model model = ModelLoader.loadModel(Path.of("Qwen3-1.7B-f16.gguf"), 512, true, true);

// A) static batch: all B streams bit-exact vs single-stream greedy
var report = BatchedDecodeEngine.run(model, "What is the capital of France?",
        BatchDecodeOptions.of(32));                // batch 32, context 512, 64 tokens

// B) continuous multi-request serving, paged KV + prefix cache
var options = new BatchDecodeOptions(
        16, 512, 64, true,      // batch, context, tokens, CUDA graphs
        true, 16, 0,            // paged, block size, blocks (0 = derive)
        true, 128, 32,          // continuous, requests, minimum tokens
        true, 0.0f, true);      // prefix cache, temperature, device sampling
var continuous = BatchedDecodeEngine.run(model, "What is the capital of France?", options);
```

Run it on the JVM command `./jitllm --gpu --show-command …` prints, with your class on the
classpath and `-Djitllm.prefillBatchSize` set to the batch size (32 for A, 16 for B): the prefill
plan is sized for it, and `run` refuses to start otherwise. `run` returns a report instead of
printing: a `StaticReport` (`allStreamsIdentical`, `distinctStreams`, throughput) or a
`ContinuousReport` (`prefixConsistent`, throughput, slot utilization, peak blocks, prefill tokens).

`BatchDecodeOptions` fields and the `of(B)` defaults:

| field | default | meaning |
|-------|---------|---------|
| `batchSize` | B | slots decoded together |
| `decodeContext` | 512 | context each slot gets (VRAM) |
| `decodeTokens` | 64 | tokens decoded per slot |
| `cudaGraphs` | true | capture and replay the plan as a CUDA graph |
| `paged` | false | block-addressed KV storage |
| `blockSize` | 16 | tokens per block, when paged |
| `blocks` | 0 | blocks in the pool, when paged; 0 derives `B·ctx/blockSize` |
| `continuous` | false | refill finished slots while others run |
| `requests` | 4·B | requests served in continuous mode |
| `minDecodeTokens` | 32 | shortest a refilled request may be |
| `prefixCache` | false | share a prompt prefix across requests; needs `paged` |
| `temperature` | 0.0 | 0 is greedy; > 0 gives divergent streams |
| `deviceSample` | true | argmax on the device; greedy only |

### Results as recorded

Measured with the harness on TornadoVM 5.0.1-jdk21-dev, JDK 21, RTX 4090 — every row bit-exact /
prefix-consistent:

| model | mode | throughput | correctness |
|-------|------|-----------:|-------------|
| Qwen3-1.7B | static B=8 | 198 tok/s | all streams identical |
| Qwen3-1.7B | static B=16 | 402 tok/s | all streams identical |
| Qwen3-1.7B | static B=32 | **801 tok/s** (32×) | all streams identical |
| Qwen3-1.7B | continuous B=16, R=128, paged+prefix | 301 tok/s · 6.2 req/s · 95.8% util | all prefix-consistent |
| Qwen3-4B | static B=16 | 217 tok/s | all streams identical |
| Qwen3-4B | static B=32 | **424 tok/s** (32×) | all streams identical |

Per-stream tok/s stays flat as B grows (1.7B ~25, 4B ~13.5): the MMA tile runs 128
rows regardless of B, so batching is effectively free until B=128.

### Exact prompts + configuration per result

Prompts (verbatim):
- `P_SHORT` = `Tell me a short story about a robot learning to paint.`
- `P_LONG` (48-token shared prefix) = `You are a helpful and knowledgeable assistant. Please write a detailed, vivid and engaging short story about a small curious robot named Zeta who lives in an old cluttered workshop and slowly learns the delicate art of painting with oil colors.`

Every row ran with `-Djitllm.prefillBatchSize` equal to `batchSize`.

| result | model | prompt | `BatchDecodeOptions` |
|--------|-------|--------|----------------------|
| static B=128 (41×) | llama-1b-fp16 | P_SHORT | `batchSize=128, decodeContext=512, decodeTokens=64` |
| divergent streams | llama-1b-fp16 | P_SHORT | above + `temperature=0.7, deviceSample=false` |
| continuous (vs static wave) | llama-1b-fp16 | P_SHORT | `batchSize=128, continuous, requests=512, minDecodeTokens=8` |
| paged (10.7× less KV) | llama-1b-fp16 | P_SHORT | `…continuous, paged, blocks=384, requests=512, minDecodeTokens=8` |
| prefix cache (+85%) | llama-1b-fp16 | P_LONG | `…continuous, paged, blocks=1024, prefixCache, requests=512, minDecodeTokens=8` (off: no `prefixCache`) |
| Qwen3 paged+prefix (+104%) | Qwen3-1.7B-f16 | P_LONG | `-Dtornado.device.memory=20GB`; `batchSize=64, decodeContext=512, decodeTokens=64, continuous, paged, blocks=768, prefixCache, requests=256, minDecodeTokens=8` |

The static-wave baseline of the continuous comparison refilled no slots, a mode the harness no
longer offers.

---

## Limitations / next

- **FP16 LLaMA + Qwen3, greedy, CUDA tensor cores.** Q8_0 and other architectures reuse
  the same pattern but need their own decode layer graph. On the toolchain the results
  above were recorded with, Qwen3-8B produced garbage in the **stock** decode path
  (independent of batched decode — the engine faithfully reproduces the stock output);
  Qwen3-1.7B/4B were fine.
- **One prompt in the harness.** `run` replicates its prompt across slots; continuous mode
  varies each request's generated length. `jitllm serve --continuous-batching` takes
  independent requests.
- **Logits GEMM over the full vocab every step** is ~half the per-step cost; only
  needed for slots actually sampling.

# Memory, ownership and concurrency

Who owns what, what `close()` releases, what may be shared, and what serializes. This is
the single place those answers live; API Javadoc agrees with it.

## Ownership

"Owner" is the object whose `close()` releases the resource. "Borrowers" hold a reference
and must not release it.

| Resource | Owner | Borrowers | Released at | Sharing |
| --- | --- | --- | --- | --- |
| Model weights, host and device | `LocalModel` | compiled programs, sessions, engine | `LocalModel.close()` | any number of sessions and programs of the same model |
| Compiled program and its fixed device workspace | its entry in the model's program cache | sessions and the engine that invoke it | model close | every session in the same binding domain |
| Compiled-program cache | `LocalModel` | none — internal | model close | n/a; no public compile entry point and no eviction |
| Backend and device selection | `LocalModel` | model, programs, engine | model close | resolved once at load, immutable after |
| `KvCacheManager` and its `BlockPool` | the engine, for engine-created sessions; otherwise the session runtime the `LocalModel` handle owns | sessions via leases, scheduler, prefix cache | engine close, or model close on the engineless path | one pool per (owner, device) |
| KV lease and its block table | the session holding it | attention kernels during an invocation | `close()`, or `reset()` for private blocks | the *blocks* may be shared and refcounted; the lease never is |
| Shared prefix blocks | `BlockPool` storage, accounted by `PrefixCache` | any lease whose table references them | refcount zero **and** not pinned | any lease of the same (model, dtype, position offset) |
| Session logical state — position, lease handle, sampler, conversation | the session | the invocation that writes it into control arrays | session close | never shared |
| Engine batched invocation buffers | the engine | the compiled program during a step | engine close | one set per engine; slots are positions in them, not per-session allocations |
| Session | the caller, or the engine for internal sessions | the engine schedules it, never owns it | `close()`, idempotent | many sessions per model |
| Engine | the caller | none | `engine.close()` — **never closes the model it borrows** | one engine per (model, device, execution configuration) |
| Metrics sink | the composition root that installs it | backends write, engine and API read | never device-owning | one per process is typical; per-engine is permitted. Must be thread-safe |

A live session makes `LocalModel.close()` throw, naming the sessions, and the model stays
open. Closing a model while kernels can still reach its weights is never a silent free.

## The session/program split

Before compiled programs were shared, a session owned everything it touched — its
activation buffers, staging arrays and control arrays — which is why two GPU sessions cost
two device copies of the weights. The split is what removed that:

- **A session owns its logical invocation state**: position, lease, sampler, conversation,
  budget. Per session, never shared, dies with the session.
- **A compiled program owns its fixed device workspace**: every device array bound into a
  captured graph — weights, KV pool, block table, activation and attention workspace, batch
  staging arrays, control arrays, and the arrays results are read from. Input and output
  arrays are not exempt; an output array is a device array in a captured graph.

**An invocation moves values, never arrays.** The token, the position, the block-table
slot, the active-request count and the sampling parameters are written into persistent
control and staging arrays at declared offsets. Nothing is rebound, so there is no rebind
operation to get wrong.

**Sharing a program means sharing that workspace**, which is safe because invocations are
serialized.

## Binding domains

The physical KV pool, block table and captured workspace together form a **binding
domain**, and its identity is part of the compiled-program cache key. Standalone sessions
of one model sharing one session runtime are in one domain and may share a program. An
engine has its own pool and its own domain, so an engine and a standalone session never
share a program, and neither do two engines. Equal shapes are not the same domain.

## KV storage

Storage is owned by a cache manager, never by a session and never by a loaded model. A
session holds a **lease**: a claim on blocks it does not own, released on close. Blocks
under a live lease are pinned against eviction. This wording matters, because blocks may
outlive a sequence and be shared between sequences, which is what makes prefix reuse and
paged attention possible.

The **block pool is one persistent device array**, indexed in-kernel — not a set of
separately allocated buffers. The **block table is one persistent device array** too, walked
in-kernel; host code updates its contents and never replaces the array. Host writes are
validated: an entry must name a block that exists and is leased to the writing session.
Growing the pool or the table invalidates captured graphs, so a capacity change is explicit
and off the hot path.

These are invariants, not implementation choices. Captured-graph replay bakes device
addresses, so re-pointing a slot's buffer between replays breaks replay — and with
TornadoVM's default `recover.bailout=true` a broken kernel silently falls back to
sequential Java, turning the break into wrong output rather than an error. That is why
every accelerator gate runs with `-Dtornado.recover.bailout=False`.

### Which topology gets a shared pool

A manager exists on both paths; only one shares a pool, and the choice follows execution
topology rather than a global switch.

| Topology | KV storage | Why |
| --- | --- | --- |
| Standalone session with a private plan | **private, by default** | TornadoVM allocates device buffers per task graph, so a pool sized for N sequences is copied per plan. Measured: two sessions cost 3808 MiB shared against 3360 MiB private |
| Engine with its batched shared plan | **one shared manager, store and pool** | one plan, one device copy, slots inside it — what the pool was designed for |

**Silent fallback is forbidden.** Engine construction *fails* for a family that cannot use
shared storage — engine execution is the shared pool, and there is no engine that quietly
runs on private per-session storage. An **explicit** standalone request for shared storage
is honoured or refused, never dropped. The standalone default asks the model nothing,
because private storage is what it already had.

A family qualifies when its layer graphs consume the block table from a *named*
predecessor. One that re-uploads its own copy would read a stale mapping from a shared,
mutable table.

## Concurrency

**A session is not thread-safe.** One thread at a time. A loaded model is thread-safe;
so is `addRequest` on the engine.

**`invoke(...)` is not thread-safe either, and does not need to be.** The program's cache
entry serializes the whole operation — bind, execute, complete, read back — which is what
makes several live sessions safe without per-session copies of the program. Two sessions
sharing a program take turns in the workspace, and each reads its results before the next
begins. In batched mode the engine is the sole invoker.

True concurrent invocation of one program is not planned: device concurrency comes from
batching, so there is no consumer for it.

**User callbacks run outside the lock.** A token callback is arbitrary user code; holding
the program lock across it would let one caller stall every other session, and re-entering
the session from a callback would deadlock.

## Engine buffers

Batched invocation buffers are allocated once at engine construction, sized by the maximum
batch width, and never reallocated while a captured graph exists. The width is fixed at
construction: changing it means reconstruction and recapture, which is a policy event, not
an allocation. A step with fewer active requests leaves the spare slots inactive; it
resizes nothing.

## Memory planning

`LocalModels.preflight(...)` predicts device memory before anything is allocated, and a
load known to exceed capacity throws `GPUL-MEM-001` before the first device buffer rather
than dying part-allocated.

A plan carries a **confidence level**, and admission acts on it. `EXACT` is claimed only
where the model has been bisected against measurement on that backend; elsewhere the plan
is capped at `CONSERVATIVE`. The model multiplies per-layer weight bytes by the number of
layer-graph families, because the TornadoVM runtime holds object state per task graph and
so allocates a device buffer per graph that binds an array — getting that count wrong
under-predicts by roughly the size of the model, and an under-prediction admits a load that
then dies part-allocated. `GraphTopologyConsistencyTest` pins each layout's declared family
count against its graph indices.

The tensor-core batched prefill layers add a per-session workspace — tensor-core attention
operands, split-K partial products, and the FP16 projections a Q8_0 (or native F16) prefill
decodes — which each layer class sizes with the arithmetic its constructor allocates by
(`TornadoPlanProvider.batchPrefillWorkspace`). Each session builds its own plan, so the
workspace is charged once per session. Whether the decoded projections stay resident is
decided against the budget at load (`MemoryPreflight.resolveDecodedProjections`) and carried
to the layers in the execution policy, so the plan builds what was predicted; when not even
the per-chunk scratch fits beside an exactly predicted remainder, the load is refused. These
components are not bisected against measurement, so a plan that carries them is
`CONSERVATIVE`.

The MLX prefill on Metal is accounted the same way (`MlxPrefill.workspace`, through each
provider's `batchPrefillWorkspace`): the stacked F16 copies, or the resident FP16 copies or
affine repack of Q8_0 and Q4_0 projections, the FP16 GEMM staging, and the attention's
contiguous key/value buffers sized for the whole context. The model's projection weights are
not charged twice: the prefill graphs stop binding the ones MLX replaced and the decode graphs
upload those themselves, binding the rest from the prefill graphs. Measured on Qwen3-0.6B at
width 128 and context 4096, the prediction is within 2 MiB above the allocation in every
representation.

A `CONSERVATIVE` plan is report-only, with one exception: when its **logical** bytes — every
buffer counted once, no duplication or headers — already exceed the configured budget, the
load is refused before the first device buffer (`MemoryPlan.exceedsConfiguredBudgetAtLowerBound`).
That is a lower bound, so it cannot refuse a load that would have run. It is what refuses
Mistral-7B Q8_0 with an MLX prefill of 2048 under 14 GB, which otherwise died part-allocated.

## A note on device memory and test forks

Device memory a closed session frees returns to TornadoVM's buffer provider but not to the
driver, and the provider recycles it only under budget pressure. Measured: five sequential
load/session/close cycles in one JVM each grow resident device memory by about 2.1 GiB and
never shrink. Process exit is what releases it. That is why the accelerator suite forks one
JVM per test class — a shared JVM exhausts the device after a handful of classes, whichever
they are, and the test that fails is the last one to run rather than the one that is wrong.

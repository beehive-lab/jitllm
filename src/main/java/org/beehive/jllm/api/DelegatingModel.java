package org.beehive.jllm.api;

import java.nio.file.Path;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.stream.Collectors;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.kv.KvLease;
import org.beehive.jllm.runtime.kv.KvStorageFactories;
import org.beehive.jllm.runtime.kv.KvStorageFactory;
import org.beehive.jllm.runtime.kv.KvStorageRequest;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/**
 * The facade over today's {@code Model}: everything a user holds, nothing a user should not.
 *
 * <p>Package-private on purpose — it is reached only as {@link LocalModel} or {@link
 * TextGenerationModel}, so no accessor of the internal model can leak through the facade.
 *
 * <p>Thread-safe, as the interface promises: the delegate is immutable after load, and the only
 * mutable state here is the set of open sessions, which is synchronized.
 */
final class DelegatingModel implements TextGenerationModel {

    private final Model delegate;
    private final ModelInfo info;

    /**
     * This model's default execution policy, resolved once at load.
     *
     * <p>Sessions override it field by field; nothing here re-reads a system property afterwards.
     */
    private final ExecutionPolicy executionPolicy;

    /** The default reasoning mode every session inherits unless it overrides it. */
    private final ThinkingMode thinkingMode;

    /**
     * How this model's key/value storage is shaped, resolved once at load.
     *
     * <p>Not policy: it types and sizes the pool every session addresses, so no session overrides
     * it.
     */
    private final org.beehive.jllm.runtime.policy.StorageOptions storageOptions;

    private final KvCacheManager sessions;

    /**
     * The handle's internal compiled-program cache, and the binding domains programs are keyed
     * against.
     *
     * <p>Internal in v1: nothing about it reaches a public signature. Released at model close, when
     * guarantees no session is live.
     */
    private final org.beehive.jllm.backend.tornado.lowering.CompiledProgramCache programs =
            new org.beehive.jllm.backend.tornado.lowering.CompiledProgramCache();

    private final ModelConfiguration configuration;
    private final boolean gpu;

    /** Identity-keyed: two sessions are the same session only if they are the same object. */
    private final Set<DelegatingSession> openSessions =
            Collections.newSetFromMap(new IdentityHashMap<>());

    private boolean closed;

    private static final int BLOCK_SIZE_TOKENS = 16;

    /** Rule 16: library code routes its output through the platform logger. */
    private static final System.Logger LOGGER = System.getLogger(DelegatingModel.class.getName());

    /**
     * How many sessions may be open at once — {@link ModelOptions#maxConcurrentSessions()}. It
     * sizes the admission accounting and, when one is attached, the shared key/value pool.
     */
    private final int maxConcurrentSessions;

    DelegatingModel(Model delegate, Path source, boolean gpu) {
        this(
                delegate,
                source,
                gpu,
                ExecutionPolicy.fromSystemProperties(),
                org.beehive.jllm.runtime.policy.StorageOptions.fromSystemProperties(),
                ThinkingMode.DEFAULT);
    }

    DelegatingModel(
            Model delegate,
            Path source,
            boolean gpu,
            ExecutionPolicy executionPolicy,
            org.beehive.jllm.runtime.policy.StorageOptions storageOptions) {
        this(delegate, source, gpu, executionPolicy, storageOptions, ThinkingMode.DEFAULT);
    }

    DelegatingModel(
            Model delegate,
            Path source,
            boolean gpu,
            ExecutionPolicy executionPolicy,
            org.beehive.jllm.runtime.policy.StorageOptions storageOptions,
            ThinkingMode thinkingMode) {
        this(delegate, source, gpu, executionPolicy, storageOptions, thinkingMode, 1);
    }

    DelegatingModel(
            Model delegate,
            Path source,
            boolean gpu,
            ExecutionPolicy executionPolicy,
            org.beehive.jllm.runtime.policy.StorageOptions storageOptions,
            ThinkingMode thinkingMode,
            int maxConcurrentSessions) {
        if (maxConcurrentSessions < 1) {
            throw new IllegalArgumentException(
                    "maxConcurrentSessions must be at least 1: " + maxConcurrentSessions);
        }
        this.maxConcurrentSessions = maxConcurrentSessions;
        this.delegate = delegate;
        this.gpu = gpu;
        this.thinkingMode = thinkingMode;
        this.executionPolicy = executionPolicy;
        this.storageOptions = storageOptions;
        this.configuration = new ConfigurationView(delegate.configuration());
        // One weight representation is all today's Weights can report: it carries a single
        // materialized type for the whole set. Per-tensor descriptors make a genuinely
        // mixed answer possible, and ModelInfo can already express it — see weightTypes().
        // Sized for the caller's concurrency at the model's load-time context length. The
        // bytes-per-block figure is what a block costs across every layer, which is what
        // admission has to reason about (D5).
        int contextLength = delegate.configuration().contextLength();
        this.sessions =
                KvCacheManager.sizedFor(
                        maxConcurrentSessions,
                        contextLength,
                        BLOCK_SIZE_TOKENS,
                        bytesPerBlock(delegate, storageOptions));
        attachStorage(delegate, contextLength);
        this.info =
                new ModelInfo(
                        delegate.getModelType().name(),
                        delegate.getModelType().name().toLowerCase(),
                        delegate.configuration().contextLength(),
                        source,
                        java.util.Set.of(delegate.weights().dataType()),
                        delegate.configuration().activationType());
    }

    /**
     * Gives the session runtime its device storage, so leases address one shared pool instead of
     * every session allocating a copy of the cache.
     *
     * <p>GPU only, and only for families whose kernels can read the block-major layout — Llama
     * today. Everything else keeps the retained per-state path, which is what makes this migration
     * family-by-family rather than a flag day. A failure to allocate is not fatal here: the runtime
     * falls back to per-state storage, which is the behaviour of the release before this one.
     */
    private void attachStorage(Model delegate, int contextLength) {
        // Shared key/value storage is optional for the legacy path and keeps its default, and is a
        // construction invariant of a shareable binding domain: a shared workspace and program over
        // session-private key/value arrays is the combination that silently gives one session
        // another's cache (option 1).
        if (!gpu
                || !attachesSharedPool(
                        delegate,
                        storageOptions,
                        org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection.mayHandle(
                                delegate, executionPolicy))) {
            return;
        }
        var config = delegate.configuration();
        int kvDim = delegate.kvCacheDim();
        int blocksPerSlot = (contextLength + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
        var request =
                new KvStorageRequest(
                        KvCacheManager.totalBlocks(maxConcurrentSessions, blocksPerSlot),
                        blocksPerSlot,
                        maxConcurrentSessions,
                        BLOCK_SIZE_TOKENS,
                        config.numberOfLayers(),
                        kvDim,
                        storageOptions.usesFp16KeyValueCache());

        // Resolved OUTSIDE the try, deliberately. The catch below is for a device that
        // cannot fit the pool, which is a capacity answer and a reasonable thing to fall back from.
        // "No backend can allocate a pool at all" is a configuration error — a lost service file in
        // a shaded jar, most likely — and swallowing it here would turn every shared-pool
        // deployment into a silent per-session one.
        KvStorageFactory factory = KvStorageFactories.single();
        try {
            sessions.attach(factory.create(request));
        } catch (RuntimeException | OutOfMemoryError e) {
            LOGGER.log(
                    System.Logger.Level.INFO,
                    "shared KV pool not attached (" + e + "); sessions keep their own cache");
        }
    }

    /**
     * Whether a GPU load of this model attaches the shared key/value pool — the one rule, so the
     * preflight predicts the reservation the load makes.
     *
     * @param lowerable whether a session of this model can take the lowered path, which requires
     *     the pool
     */
    static boolean attachesSharedPool(
            Model model,
            org.beehive.jllm.runtime.policy.StorageOptions storageOptions,
            boolean lowerable) {
        return model.supportsSharedKvStorage()
                && (storageOptions.sharedKeyValuePool() || lowerable);
    }

    /**
     * Rejects an explicit reasoning mode this model's format cannot express.
     *
     * <p>{@link ThinkingMode#DEFAULT} always passes: it asks for nothing, so there is nothing a
     * family could fail to represent. {@code ENABLED} and {@code DISABLED} are requests, and a
     * request that cannot be honoured is an error — silently ignoring it would leave a caller who
     * turned thinking off paying for tokens they asked not to generate, with nothing to tell them.
     */
    private ThinkingMode requireRepresentable(ThinkingMode mode) {
        if (mode.isExplicit() && !delegate.chatFormat().supportsThinking()) {
            throw new IllegalArgumentException(
                    "thinkingMode("
                            + mode
                            + ") was requested, but "
                            + info.name()
                            + " has no reasoning phase to control."
                            + " Use ThinkingMode.DEFAULT for a model without one");
        }
        return mode;
    }

    /** The session runtime's capacity — total and free blocks, and what a block costs. */
    KvCacheManager sessionRuntime() {
        return sessions;
    }

    @Override
    public ModelInfo info() {
        return info;
    }

    @Override
    public ModelConfiguration configuration() {
        return configuration;
    }

    /**
     * What one block of KV costs on the device, across every layer: two caches (keys and values),
     * {@code kvDim} values per token, {@code BLOCK_SIZE_TOKENS} tokens, every layer.
     */
    private static long bytesPerBlock(
            Model model, org.beehive.jllm.runtime.policy.StorageOptions storage) {
        var config = model.configuration();
        long kvDim = (long) config.dim() * config.numberOfKeyValueHeads() / config.numberOfHeads();
        // Taken from the resolved storage options rather than assumed FP32: the accounting a
        // caller sees should describe the pool that was actually built, and an FP16 pool costs
        // half. It read Float.BYTES unconditionally with a comment admitting the gap.
        long bytesPerValue = storage.usesFp16KeyValueCache() ? Short.BYTES : Float.BYTES;
        return 2L * kvDim * BLOCK_SIZE_TOKENS * config.numberOfLayers() * bytesPerValue;
    }

    @Override
    public GenerationSession newSession() {
        return newSession(SessionOptions.defaults());
    }

    @Override
    public GenerationSession newSession(SessionOptions options) {
        int modelContext = delegate.configuration().contextLength();
        int requested = options.contextLength();
        if (requested > modelContext) {
            throw new IllegalArgumentException(
                    DiagnosticCode.CONTEXT_LENGTH_EXCEEDED.message(
                            "session context length "
                                    + requested
                                    + " exceeds the model's "
                                    + modelContext));
        }
        // Validated before anything is acquired. Throwing after the lease would leak it, and the
        // model could then never close — which is exactly what the first version of this did, and
        // what ThinkingModeAccelTest caught.
        ThinkingMode thinking = requireRepresentable(options.resolveThinkingMode(thinkingMode));
        DelegatingSession session;
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException(
                        DiagnosticCode.USED_AFTER_CLOSE.message("model is closed: " + info.name()));
            }
            if (openSessions.size() >= maxConcurrentSessions) {
                throw new IllegalStateException(
                        DiagnosticCode.KV_POOL_EXHAUSTED.message(
                                (maxConcurrentSessions == 1
                                                ? "the one session "
                                                : "all " + maxConcurrentSessions + " sessions ")
                                        + info.name()
                                        + " was loaded for "
                                        + (maxConcurrentSessions == 1 ? "is" : "are")
                                        + " open; close one, or load the model with"
                                        + " ModelOptions.builder().maxConcurrentSessions(n)"));
            }
            int contextLength = requested > 0 ? requested : modelContext;
            KvLease lease = sessions.acquire(contextLength);
            // Resolved once, here, and carried by the session. Nothing reads
            // it per token, and nothing re-reads a system property after this point.
            ExecutionPolicy policy = options.executionPolicy().applyTo(executionPolicy);
            session =
                    new DelegatingSession(
                            this, delegate, gpu, contextLength, lease, policy, thinking);
            openSessions.add(session);
        }
        return session;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Throws while sessions are open and releases nothing in that case — the model stays usable,
     * so the caller can close the sessions and try again. Force-closing them instead would
     * invalidate another thread's session mid-generation, which is the worse failure.
     */
    @Override
    public void close() {
        synchronized (this) {
            if (closed) {
                return; // idempotent
            }
            if (!openSessions.isEmpty()) {
                String live =
                        openSessions.stream()
                                .map(
                                        s ->
                                                "session@"
                                                        + Integer.toHexString(
                                                                System.identityHashCode(s)))
                                .collect(Collectors.joining(", "));
                throw new IllegalStateException(
                        DiagnosticCode.CLOSE_WITH_LIVE_DEPENDENTS.message(
                                "cannot close "
                                        + info.name()
                                        + " while "
                                        + openSessions.size()
                                        + " session(s) are open: "
                                        + live
                                        + ". Close them first; the model is untouched."));
            }
            closed = true;
            // Every lease is a session, and sessions closed above, so this cannot throw here.
            sessions.close();
            // Released exactly once, here, after has guaranteed no borrower is left.
            if (loweredProgram != null) {
                loweredProgram.freeTornadoExecutionPlan();
                loweredProgram = null;
            }
            loweredWorkspace = null;
            loweredDomain = null;
            programs.close();
        }
        // Nothing else to release here: the weights are memory-mapped and the sessions that owned
        // device memory have already released it. This is the seam where an explicit weight
        // release lands once the model owns its device buffers.
    }

    /**
     * The one binding domain lowered sessions share, and the workspace it owns.
     *
     * <p>One domain, one fixed device workspace, one key/value pool, one block table — allocated
     * <b>once per domain</b>, not once per session. Every lowered session borrows them and brings
     * only its own lease and slot.
     *
     * <p>Transitionally the workspace is an ordinary {@code State} instance. Its host-side tensors
     * go unused on this path; extracting a purpose-built backend workspace type is follow-up work
     * for when the remaining families migrate, and making it a prerequisite would mean redesigning
     * nine states to prove one vertical slice.
     */
    private org.beehive.jllm.backend.tornado.lowering.BindingDomain loweredDomain;

    private org.beehive.jllm.inference.state.State loweredWorkspace;

    private org.beehive.jllm.backend.tornado.TornadoVMMasterPlan loweredProgram;

    /**
     * Builds this session's runtime: legacy when it owns its state, lowered when it borrows the
     * domain's.
     *
     * <p>A model with no shared key/value storage takes the legacy path. What it must never take is
     * the lowered one over session-private key/value arrays: that is the topology in which one
     * session executes against another's cache.
     */
    synchronized SessionRuntime newRuntime(
            org.beehive.jllm.model.Model delegate,
            org.beehive.jllm.runtime.kv.KvLease lease,
            ExecutionPolicy policy) {
        if (!gpu
                || !org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection.mayHandle(
                        delegate, executionPolicy)) {
            return new LegacySessionRuntime(delegate, lease, policy, storageOptions);
        }
        if (!policy.equals(executionPolicy)) {
            // The domain's workspace is built once, for the model's policy. A session that
            // overrode it needs a program built for its own, and it cannot get one out of a
            // workspace bound to another — so it runs standalone rather than borrowing a program
            // that answers a different question.
            return new LegacySessionRuntime(delegate, lease, policy, storageOptions);
        }
        if (loweredWorkspace == null) {
            if (!sessions.hasStorage()) {
                // No shared key/value storage: this family's kernels do not address the shared
                // pool, or attaching it failed. Take the legacy path.
                LOGGER.log(
                        System.Logger.Level.DEBUG,
                        () ->
                                "lowering requested but "
                                        + info.name()
                                        + " has no shared key/value storage; using the legacy path");
                return new LegacySessionRuntime(delegate, lease, policy, storageOptions);
            }
            // Allocated once per domain. A second lowered session reaches neither of these lines.
            loweredWorkspace =
                    org.beehive.jllm.inference.state.State.withStorageOptions(
                            storageOptions, () -> delegate.createNewState(lease));
            // The domain's workspace carries the model's policy: every session that shares it
            // shares that policy, which is why a session that overrode it took the legacy path
            // above rather than reaching this line.
            loweredWorkspace.resolveExecutionPolicy(executionPolicy);
            loweredDomain =
                    org.beehive.jllm.backend.tornado.lowering.BindingDomain.shareable(
                            "runtime@" + Integer.toHexString(System.identityHashCode(this)),
                            loweredWorkspace);
        }
        if (!org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection.handles(
                delegate, loweredWorkspace)) {
            return new LegacySessionRuntime(delegate, lease, policy, storageOptions);
        }
        var key =
                org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection.key(
                        delegate,
                        loweredDomain,
                        loweredWorkspace.executionPolicy(),
                        loweredWorkspace.usesFp16KeyValueCache()
                                ? org.beehive.jllm.runtime.tensor.DataType.F16
                                : org.beehive.jllm.runtime.tensor.DataType.F32);
        loweredProgram =
                programs.acquire(
                        key,
                        () ->
                                org.beehive.jllm.backend.tornado.lowering.LoweredPlanSelection
                                        .lower(
                                                delegate,
                                                loweredWorkspace,
                                                org.beehive.jllm.auxiliary.metrics.RunMetricsSink
                                                        .installedOrDisabled()));
        // After the acquire, not inside the supplier: on a cache hit the supplier never runs, and
        // this session took the lowered path just the same. Without it a run through the facade
        // reports no execution_path at all.
        org.beehive.jllm.backend.tornado.TornadoVMMasterPlan.reportLoweredPath(
                delegate, loweredWorkspace);
        var perSession =
                new org.beehive.jllm.backend.tornado.lowering.SharedWorkspacePlan(
                        loweredProgram,
                        loweredDomain.invocationLock(),
                        loweredWorkspace.workspace.positionHolder,
                        lease.slot(),
                        delegate.configuration().vocabularySize(),
                        token ->
                                org.beehive.jllm.backend.tornado.lowering.EmbeddingStaging.stage(
                                        delegate, loweredWorkspace, token),
                        false,
                        null);
        // The family's initial seed, taken from a freshly built state rather than assumed: it is
        // what createNewState puts there, and for Llama that is <|begin_of_text|>, not -1.
        return new LoweredSessionRuntime(
                loweredWorkspace, lease, perSession, loweredWorkspace.latestToken);
    }

    /**
     * The domain's workspace, or {@code null} before the first lowered session. Internal; tests.
     */
    org.beehive.jllm.inference.state.State loweredWorkspace() {
        return loweredWorkspace;
    }

    /** The domain, or {@code null} before the first lowered session. Internal; tests. */
    org.beehive.jllm.backend.tornado.lowering.BindingDomain loweredDomain() {
        return loweredDomain;
    }

    /** The shared compiled program, or {@code null} before the first lowered session. Internal. */
    org.beehive.jllm.backend.tornado.TornadoVMMasterPlan loweredProgram() {
        return loweredProgram;
    }

    /** The compiled-program cache's keys, for diagnostics. Internal; for tests. */
    java.util.List<String> compiledProgramKeys() {
        return programs.describeKeys();
    }

    /** How many distinct compiled programs this handle holds. Internal; for tests. */
    int compiledProgramCount() {
        return programs.size();
    }

    synchronized boolean isClosed() {
        return closed;
    }

    synchronized void sessionClosed(DelegatingSession session) {
        openSessions.remove(session);
    }
}

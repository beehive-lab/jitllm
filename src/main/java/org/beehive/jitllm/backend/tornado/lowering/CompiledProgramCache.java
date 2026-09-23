package org.beehive.jitllm.backend.tornado.lowering;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;

/**
 * The model handle's internal compiled-program cache.
 *
 * <p>Internal in v1: no type, key or statistic of it appears in a public signature. Populated
 * lazily, released with the model, and never a single mutable slot.
 *
 * <h2>Concurrency and failure</h2>
 *
 * <p>Concurrent identical misses <b>compile once</b> — the first caller holds the entry's monitor
 * and the others wait on it — and a <b>failed compilation removes the pending entry</b> and
 * propagates to every waiter, so no failure is cached and a later request retries.
 *
 * <p>{@code invoke(.)} on a resulting entry remains <b>not thread-safe</b>; that is the caller's
 * serialization to arrange, not this class's.
 *
 * <h2>What a hit actually means</h2>
 *
 * <p>A hit means "the same compiled program, bound to the same physical arrays". The {@link
 * BindingDomain} in the key is what carries the second half, and it compares by identity rather
 * than by shape — two domains with identical dimensions are still two domains. Getting that wrong
 * would hand one session a program bound to another session's buffers, which under CUDA graph
 * capture produces wrong output rather than an error.
 */
public final class CompiledProgramCache {

    private final Map<ProgramCacheKey, Entry> entries = new HashMap<>();

    /**
     * Returns the compiled program for {@code key}, compiling it once if absent.
     *
     * @param compile invoked at most once per key; a throw removes the pending entry
     */
    public TornadoVMMasterPlan acquire(ProgramCacheKey key, Supplier<TornadoVMMasterPlan> compile) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(compile, "compile");
        Entry entry;
        synchronized (entries) {
            entry = entries.computeIfAbsent(key, k -> new Entry());
        }
        synchronized (entry) {
            if (entry.plan != null) {
                return entry.plan;
            }
            try {
                entry.plan = compile.get();
            } catch (RuntimeException | Error failure) {
                synchronized (entries) {
                    entries.remove(key, entry);
                }
                throw failure;
            }
            return entry.plan;
        }
    }

    /** How many distinct compiled programs this cache holds. For tests and diagnostics. */
    public int size() {
        synchronized (entries) {
            return entries.size();
        }
    }

    /**
     * The keys this cache holds, rendered for diagnostics.
     *
     * <p>Exists because "two entries where one was expected" is unactionable without seeing which
     * component of the key differs.
     */
    public java.util.List<String> describeKeys() {
        synchronized (entries) {
            return entries.keySet().stream()
                    .map(
                            k ->
                                    "signature#"
                                            + k.signature().hashCode()
                                            + " backend="
                                            + k.backend()
                                            + " device="
                                            + k.device()
                                            + " compile="
                                            + k.compileOptions().fingerprint()
                                            + " caps="
                                            + k.capabilityFingerprint()
                                            + " domain="
                                            + System.identityHashCode(k.bindingDomain()))
                    .toList();
        }
    }

    /** Releases every compiled program. Called at model close, when no session can be live. */
    public void close() {
        Map<ProgramCacheKey, Entry> snapshot;
        synchronized (entries) {
            snapshot = Map.copyOf(entries);
            entries.clear();
        }
        for (Entry entry : snapshot.values()) {
            synchronized (entry) {
                if (entry.plan != null) {
                    entry.plan.freeTornadoExecutionPlan();
                    entry.plan = null;
                }
            }
        }
    }

    private static final class Entry {
        private TornadoVMMasterPlan plan;
    }
}

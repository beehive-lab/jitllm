package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Objects;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.workspace.TornadoLogits;
import org.beehive.jitllm.inference.Logits;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * One session's view of a compiled program whose workspace it shares with other sessions.
 *
 * <p>Sessions sharing a binding domain share one compiled program and one fixed device workspace.
 * They take turns in it, and this decorator is where the turn-taking happens — per
 * <b>invocation</b>, not per {@code generate()} call, because {@code TokenGenerationLoop} runs its
 * token callback inside its own loop and a lock held across {@code generate()} would hold it across
 * every callback.
 *
 * <p>Each invocation:
 *
 * <pre>
 *   lock the domain's compiled entry
 *     → write this session's slot into the fixed control array
 *     → execute
 *     → copy the logits out of the shared carrier into session-owned storage
 *   unlock
 *     → the caller's callback then runs outside the lock
 * </pre>
 *
 * <p><b>The copy is not an optimization to skip.</b> Returning the shared carrier would leave a
 * session holding a mutable view of storage the next session is about to overwrite — precisely what
 * the ownership split forbids, and a data race that would show up as another session's logits.
 */
public final class SharedWorkspacePlan implements TornadoVMMasterPlan, InvocationBoundary {

    @Override
    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        synchronized (lock) {
            return shared.executionInfo();
        }
    }

    private final TornadoVMMasterPlan shared;
    private final Object lock;
    private final IntArray control;
    private final int slot;
    private final FloatArray privateLogits;

    /**
     * The neutral view over {@link #privateLogits}, built once.
     *
     * <p>Its target is this session's own array and its identity is fixed for the plan's life, so
     * the view is a field rather than something the invocation builds — an invocation that built
     * one would add an allocation to the per-token path.
     */
    private final Logits privateLogitsView;

    private final EmbeddingStager stager;
    private final boolean deviceSample;

    /** Stages one token's embedding into a domain-owned carrier. Supplied by the lowering. */
    @FunctionalInterface
    public interface EmbeddingStager {
        void stage(int token);
    }

    /**
     * @param shared the domain's compiled program
     * @param lock the domain's invocation lock — one per compiled entry
     * @param control the fixed control array the slot is written into
     * @param slot this session's slot in the shared key/value pool
     * @param vocabularySize how many logits to copy back per invocation
     */
    public SharedWorkspacePlan(
            TornadoVMMasterPlan shared,
            Object lock,
            IntArray control,
            int slot,
            int vocabularySize,
            EmbeddingStager stager,
            boolean deviceSample,
            IntArray sampledToken) {
        this.shared = Objects.requireNonNull(shared, "shared");
        this.lock = Objects.requireNonNull(lock, "lock");
        this.control = Objects.requireNonNull(control, "control");
        this.slot = slot;
        this.privateLogits = new FloatArray(vocabularySize);
        this.privateLogitsView = TornadoLogits.of(this.privateLogits);
        this.stager = Objects.requireNonNull(stager, "stager");
        this.deviceSample = deviceSample;
        this.sampledToken = sampledToken;
    }

    private final IntArray sampledToken;

    /**
     * One invocation, whole, with the lock held across staging, execution and copy-out.
     *
     * <p>The staging happens <b>here</b> rather than in {@code InferenceCore}, which is the point
     * of the boundary: a caller that staged into its own device arrays would be writing where this
     * program does not read.
     */
    @Override
    public Result invoke(int token, int position) {
        int chosen;
        synchronized (lock) {
            stager.stage(token); // into the domain's carrier, not the session's
            control.set(1, slot); // this session's slot: an invocation value
            FloatArray sharedLogits = shared.tornadoVMForwardDecode(position);
            for (int i = 0; i < privateLogits.getSize(); i++) {
                privateLogits.set(i, sharedLogits.get(i));
            }
            chosen = deviceSample && sampledToken != null ? sampledToken.get(0) : -1;
        }
        // Unlocked: the caller may now sample and run its callback, holding only its own copies.
        return new Result(privateLogitsView, chosen);
    }

    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        synchronized (lock) {
            // This session's slot is an invocation value: written into the fixed control array,
            // never bound as a different array.
            control.set(1, slot);
            FloatArray sharedLogits = shared.tornadoVMForwardDecode(position);
            for (int i = 0; i < privateLogits.getSize(); i++) {
                privateLogits.set(i, sharedLogits.get(i));
            }
        }
        // Outside the lock: the caller may now run its callback, and what it holds is this
        // session's own copy rather than a view of storage the next session will overwrite.
        return privateLogits;
    }

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        return shared.createExecutionPlan();
    }

    @Override
    public void forceCopyInReadOnlyData() {
        synchronized (lock) {
            shared.forceCopyInReadOnlyData();
        }
    }

    /**
     * Delegates, holding the domain's lock.
     *
     * <p>The recurrent state lives in the shared workspace, so this is a write another session
     * could observe. It is taken under the lock for that reason, and it is only correct because a
     * reset is a session-level act on a workspace the sessions take turns in — the same contract
     * every other invocation here relies on.
     */
    @Override
    public void resetSequenceState() {
        synchronized (lock) {
            shared.resetSequenceState();
        }
    }

    /**
     * Releases nothing.
     *
     * <p>The compiled program belongs to the domain and outlives every session that borrowed it; a
     * session that freed it would tear down device memory the cache still hands out. The domain
     * releases it at model close.
     */
    @Override
    public void freeTornadoExecutionPlan() {
        // deliberately empty — see the javadoc
    }
}

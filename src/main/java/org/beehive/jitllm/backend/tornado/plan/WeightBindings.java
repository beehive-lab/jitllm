package org.beehive.jitllm.backend.tornado.plan;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Which task graph uploads each weight, so that every weight is on the device once.
 *
 * <p>Example: {@code batchPrefillLayer_3} reads layer 3's Q8_0 weights and records them here. The
 * decode graph of layer 3 then reuses them ({@code consumeFromDevice}) instead of uploading a
 * second copy. A weight nobody recorded is uploaded by the graph that needs it.
 *
 * <p>One per execution plan. Weights are compared by identity (the same array object).
 */
public final class WeightBindings {

    private final Map<Object, String> owners = new IdentityHashMap<>();

    private final Set<Object> replaced = Collections.newSetFromMap(new IdentityHashMap<>());

    /** Records that {@code graph} uploads {@code weight}. The first graph to record it keeps it. */
    public void record(Object weight, String graph) {
        owners.putIfAbsent(weight, graph);
    }

    /** The graph that uploads {@code weight}, or {@code null} if no graph recorded it yet. */
    public String owner(Object weight) {
        return owners.get(weight);
    }

    /**
     * Records that the prefill graphs read their own copy of {@code weight} instead of the weight.
     *
     * <p>Example: under MLX, Q8_0 projections are read as FP16 copies, so {@code
     * batchPrefillLayer_3} never uploads layer 3's {@code wq}, and decode uploads it itself.
     */
    public void recordReplaced(Object weight) {
        replaced.add(weight);
    }

    /** Whether the prefill graphs replaced {@code weight} with a copy of their own. */
    public boolean replaced(Object weight) {
        return replaced.contains(weight);
    }
}

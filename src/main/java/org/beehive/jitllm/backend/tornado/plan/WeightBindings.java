package org.beehive.jitllm.backend.tornado.plan;

import java.util.IdentityHashMap;
import java.util.Map;

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

    /** Records that {@code graph} uploads {@code weight}. The first graph to record it keeps it. */
    public void record(Object weight, String graph) {
        owners.putIfAbsent(weight, graph);
    }

    /** The graph that uploads {@code weight}, or {@code null} if no graph recorded it yet. */
    public String owner(Object weight) {
        return owners.get(weight);
    }
}

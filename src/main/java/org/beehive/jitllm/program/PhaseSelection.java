package org.beehive.jitllm.program;

import java.util.List;
import java.util.Objects;

/**
 * Which of a program's components one phase runs, in order.
 *
 * <p><b>An ordered subsequence, not necessarily contiguous.</b> Prefill skips the vocabulary
 * projection at the end; a phase may equally skip something in the middle. What it may never do is
 * reorder, repeat, or name a component the program did not declare.
 *
 * <p>The indices are a {@code List<Integer>} rather than an {@code int[]} for one specific reason:
 * a signature is a cache key, and Java arrays compare by identity, so an array here would make two
 * structurally identical programs compare unequal — a silent recompile, or a claimed match that is
 * not one.
 *
 * @param phase the phase this selection is for
 * @param componentIndices indices into the program's component list, strictly ascending
 */
public record PhaseSelection(PhaseId phase, List<Integer> componentIndices) {

    public PhaseSelection {
        Objects.requireNonNull(phase, "phase");
        Objects.requireNonNull(componentIndices, "componentIndices");
        componentIndices = List.copyOf(componentIndices);
        if (componentIndices.isEmpty()) {
            throw new IllegalArgumentException("phase " + phase + " selects no components");
        }
        int previous = -1;
        for (Integer index : componentIndices) {
            Objects.requireNonNull(index, "componentIndices must not contain null");
            if (index < 0) {
                throw new IllegalArgumentException(
                        "component index must not be negative: " + index);
            }
            if (index <= previous) {
                throw new IllegalArgumentException(
                        "phase "
                                + phase
                                + " must select components in"
                                + " strictly ascending order — no reordering and no repeats: "
                                + componentIndices);
            }
            previous = index;
        }
    }
}

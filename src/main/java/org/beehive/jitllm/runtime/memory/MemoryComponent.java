package org.beehive.jitllm.runtime.memory;

import org.beehive.jitllm.api.Experimental;

/**
 * One line of a memory plan: a logical buffer group, its size, and what the backend does to it.
 *
 * @param name human-readable component name, for the report
 * @param bufferClass the role, which is what multiplicity is decided from
 * @param logicalBytes the bytes the buffers themselves occupy, counting <b>shared storage once</b>
 * @param multiplicity how many device allocations the backend makes per logical byte — 1 unless a
 *     backend binds the same storage into separate allocation domains
 * @param overheadBytes backend-required additions that are not part of the logical payload:
 *     native-array headers and alignment
 */
@Experimental
public record MemoryComponent(
        String name,
        BufferClass bufferClass,
        long logicalBytes,
        int multiplicity,
        long overheadBytes) {

    public MemoryComponent {
        if (logicalBytes < 0 || overheadBytes < 0) {
            throw new IllegalArgumentException(name + ": byte counts cannot be negative");
        }
        if (multiplicity < 1) {
            throw new IllegalArgumentException(
                    name
                            + ": multiplicity is at least 1, was "
                            + multiplicity
                            + ". A component that is not allocated should be absent, not"
                            + " present with multiplicity 0 — the report must not imply a buffer exists"
                            + " and costs nothing.");
        }
    }

    /** What this component is predicted to charge against the backend's memory budget. */
    public long predictedBytes() {
        return logicalBytes * multiplicity + overheadBytes;
    }

    /**
     * The bytes duplication adds — reported separately so multiplicity is never hidden in a total.
     */
    public long duplicationBytes() {
        return logicalBytes * (multiplicity - 1L);
    }
}

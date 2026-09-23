package org.beehive.jitllm.runtime.tensor;

/**
 * Where a model is being loaded to execute — the axis that decides which representation a tensor is
 * materialized in.
 *
 * <p>It exists because the answer genuinely differs. The host decodes K-quants and Q4_0 inside the
 * dot product and needs no conversion; the accelerator path has no kernel for them and materializes
 * {@link DataType#Q8_0} at load. A mapping without this parameter would have to pick one of those
 * answers and be wrong about the other, silently.
 */
public enum ExecutionTarget {

    /** Executed on the host CPU, which decodes quantized blocks during compute. */
    CPU,

    /**
     * Executed on an accelerator through TornadoVM, which needs a kernel for each representation.
     */
    GPU
}

package org.beehive.jllm.runtime.backend;

import org.beehive.jllm.api.Experimental;

/**
 * Selected execution settings, describing the prepared plan rather than requested flags. The kernel
 * descriptions apply to prefill; native libraries choose their own algorithms. A prefill batch size
 * of one also describes the single-token execution path.
 */
@Experimental
public record ExecutionInfo(
        String backend,
        String device,
        String runtime,
        String mode,
        int prefillBatchSize,
        String kvCache,
        String prefillKernels,
        String prefillAttention,
        boolean cudaGraphs,
        boolean stagedTransfers,
        String nativeLibraries) {
    public ExecutionInfo(
            String backend,
            String device,
            String runtime,
            String mode,
            int prefillBatchSize,
            String kvCache,
            String prefillKernels,
            String prefillAttention,
            boolean cudaGraphs,
            boolean stagedTransfers) {
        this(
                backend,
                device,
                runtime,
                mode,
                prefillBatchSize,
                kvCache,
                prefillKernels,
                prefillAttention,
                cudaGraphs,
                stagedTransfers,
                "none");
    }
}

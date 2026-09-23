package org.beehive.jllm.runtime.backend;

/**
 * What changes the compiled artefact without changing the program's description.
 *
 * <p>A cache-key component, and separate from the program on purpose: the same {@link
 * org.beehive.jllm.program.InferenceProgram} compiled with CUDA graph capture on and off is the
 * same computation and two different compiled programs.
 *
 * @param cudaGraphCapture whether {@code withCUDAGraph()} is applied — the property behind {@code
 *     jllm.cudaGraphs}, and the reason <a
 *     href="././././././././docs/architecture/memory-and-concurrency.md">capability C1</a>'s fixed
 *     device addresses matter
 */
public record CompileOptions(boolean cudaGraphCapture) {

    /** A stable string for the cache key and for diagnostics. */
    public String fingerprint() {
        return cudaGraphCapture ? "cudaGraph=on" : "cudaGraph=off";
    }
}

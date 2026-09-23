package org.beehive.jitllm.runtime.memory;

import org.beehive.jitllm.api.Experimental;

/**
 * How much storage a model's weights need, split by whether a layer binds them.
 *
 * <p>The neutral hand-off between the file format and a backend's memory model. The format layer
 * knows tensor descriptors; a backend knows what a layer graph costs; <b>neither should name the
 * other</b>, and Rule 4 enforces that a runtime or backend type never names GGUF. This record is
 * what crosses.
 *
 * <p><b>The split is the whole point.</b> A backend that builds more than one family of layer
 * graphs allocates the per-layer weights once per family, while the embeddings and output
 * projection — bound by the single activation and logits graphs — are allocated once regardless.
 * Measured on Llama-3.2-1B-F16, batched prefill costs 1872 MiB more than single-token against 1856
 * MiB of per-layer weights; the whole weight set is 2357 MiB, which would have over-predicted by
 * 500 MiB.
 *
 * <p>Byte counts are <b>logical</b>: storage shared between tensors is counted once. Two wrappers
 * over one segment — as a tied embedding/output pair is — contribute their bytes a single time.
 *
 * @param perLayerBytes bytes of weights a layer graph binds
 * @param perLayerTensors how many distinct tensors those are, for per-allocation overhead
 * @param globalBytes bytes of weights bound outside the layer graphs
 * @param globalTensors how many distinct tensors those are
 */
@Experimental
public record WeightFootprint(
        long perLayerBytes, int perLayerTensors, long globalBytes, int globalTensors) {

    public WeightFootprint {
        if (perLayerBytes < 0 || globalBytes < 0 || perLayerTensors < 0 || globalTensors < 0) {
            throw new IllegalArgumentException("weight footprint counts cannot be negative");
        }
    }

    /** Total logical weight bytes, shared storage counted once. */
    public long totalBytes() {
        return perLayerBytes + globalBytes;
    }
}

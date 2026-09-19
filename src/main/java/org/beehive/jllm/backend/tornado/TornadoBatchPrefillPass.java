package org.beehive.jllm.backend.tornado;

import java.lang.foreign.MemorySegment;
import org.beehive.jllm.inference.Logits;
import org.beehive.jllm.inference.state.Qwen2MoEState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;

/** The accelerated <b>batched prefill</b> pass and its decode step. */
public final class TornadoBatchPrefillPass {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private static final int Q4_0_BLOCK_SIZE = 32;
    private static final int Q4_0_BLOCK_BYTES = 18;

    // @formatter:off
    /**
     * Opt-in timing for the per-chunk host staging, which no kernel profiler counts.
     *
     * <p>A measurement tool with no assertion attached, kept because this cost is invisible
     * otherwise and turned out to matter: a per-task profile showed Gemma 4's Q4_0 prefill spending
     * 230 ms of kernel time per four chunks while the wall clock said 98 ms per chunk, and the
     * difference was entirely here. A static final read of a system property, so the branch is gone
     * when it is off.
     */
    // @formatter:on
    private static final boolean TIME_STAGING = Boolean.getBoolean("jllm.bench.timeStaging");

    /** Rule 16: library code routes its output through the platform logger. */
    private static final System.Logger LOGGER =
            System.getLogger(TornadoBatchPrefillPass.class.getName());

    private TornadoBatchPrefillPass() {}

    /**
     * Stages {@code chunkSize} token embeddings into the session's device batch carrier, then runs
     * the batch activation and layer graphs. The logits graph is skipped: no token in a prefill
     * batch needs its logits.
     *
     * @param model the model
     * @param state the session's state
     * @param tokens token ids for this chunk
     * @param startPos sequence position of {@code tokens[0]}
     * @param chunkSize number of tokens in this chunk
     * @param plan the batched prefill/decode GPU plan
     */
    public static void batchPrefill(
            Model model,
            State state,
            int[] tokens,
            int startPos,
            int chunkSize,
            TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        // Which prefill family takes this chunk.
        //
        // cuDNN's causal mask aligns query i to key i, which is the right mask only when the
        // query block IS the whole prefix. A chunk starting past zero has its queries at an
        // offset into a longer key range, and the library binding exposes no bottom-right
        // alignment to express that. A PARTIAL first chunk is fine: its queries still start at
        // zero and the padded rows are defined.
        //
        // So a chunk past position 0 goes to the fallback family instead: the same layer
        // pipeline, the same native projections over the same stacked weights, with the batched
        // JIT paged attention that takes an arbitrary start position. It binds every buffer from
        // the primary family, so it costs graphs and no memory, and it stays batched -- one graph
        // set per chunk, not per token.
        //
        // The plan only builds that family when the primary uses cuDNN attention; where it does
        // not, the primary already handles every chunk itself.
        boolean useFallbackFamily = startPos != 0 && plan.hasBatchPrefillFallback();

        state.workspace.batchStartPosHolder.set(0, startPos);
        // The kernels launch a fixed batchSize rows; this tells them how many are real, so the
        // padding rows do not rotate, do not write KV, and cannot run past this layer's KV slice.
        state.workspace.batchStartPosHolder.set(1, chunkSize);
        // The KV slot travels with the chunk, the way it travels with the position on the
        // single-token path. Forgetting it would address slot 0 — another session's KV.
        if (state.workspace.batchStartPosHolder.getSize() > 2) {
            state.workspace.batchStartPosHolder.set(2, state.kvSlot);
        }
        if (state instanceof Qwen2MoEState moeState
                && moeState.workspace.activeBatchSizeHolder != null) {
            moeState.workspace.activeBatchSizeHolder.set(0, chunkSize);
        }

        long embStart = TIME_STAGING ? System.nanoTime() : 0L;
        // The embedding tensor's own representation, not the model-wide one: a mixed model holds
        // them apart, and reading 18-byte blocks as 34-byte ones is a plausible activation and
        // wrong output.
        switch (weights.getTokenEmbeddingTable().dataType()) {
            case F16 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                long dimBytes = (long) config.dim() * Short.BYTES;
                for (int b = 0; b < chunkSize; b++) {
                    MemorySegment.copy(
                            embTable,
                            (long) tokens[b] * dimBytes,
                            state.workspace.embeddingXBatch.getSegment(),
                            (long) b * dimBytes,
                            dimBytes);
                }
            }
            case Q8_0 -> {
                var embTable = weights.getTokenEmbeddingTable().asByteArray();
                int dim = config.dim();
                int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                for (int b = 0; b < chunkSize; b++) {
                    int tokenId = tokens[b];
                    for (int j = 0; j < dim; j++) {
                        int blockByteOffset =
                                (tokenId * blocksPerRow + j / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                        float scale = embTable.getHalfFloat(blockByteOffset).getFloat32();
                        float quant = embTable.get(blockByteOffset + 2 + j % Q8_0_BLOCK_SIZE);
                        state.workspace.wrapXBatch.set(b * dim + j, quant * scale);
                    }
                }
            }
            case Q4_0 -> {
                // Retained: 18 bytes per 32 weights, an unsigned nibble recentred by eight. Decoded
                // here into the FP32 batch carrier, as the Q8_0 branch above decodes its own — the
                // batch activation graph then passes it through rather than converting.
                var embTable = weights.getTokenEmbeddingTable().asByteArray();
                int dim = config.dim();
                int blocksPerRow = (dim + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
                for (int b = 0; b < chunkSize; b++) {
                    int tokenId = tokens[b];
                    for (int j = 0; j < dim; j++) {
                        int blockByteOffset =
                                (tokenId * blocksPerRow + j / Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_BYTES;
                        float scale = embTable.getHalfFloat(blockByteOffset).getFloat32();
                        int within = j % Q4_0_BLOCK_SIZE;
                        int half = within / 16;
                        int packed =
                                embTable.get(blockByteOffset + 2 + (within - half * 16)) & 0xFF;
                        int quant = half == 0 ? (packed & 0xF) : ((packed >> 4) & 0xF);
                        state.workspace.wrapXBatch.set(b * dim + j, scale * (quant - 8));
                    }
                }
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported embedding weight type: "
                                    + weights.getTokenEmbeddingTable().dataType());
        }

        // Whatever this family stages per prompt token that is not the token embedding — Gemma 4's
        // per-layer embedding rows are the only case today — for the whole chunk, before the graphs
        // that read it run.
        if (TIME_STAGING) {
            LOGGER.log(
                    System.Logger.Level.INFO,
                    "staging: token embeddings for {0} tokens in {1} ms",
                    chunkSize,
                    (System.nanoTime() - embStart) / 1e6);
        }
        long stageStart = TIME_STAGING ? System.nanoTime() : 0L;
        model.stageBatchDeviceInputs(state, tokens, chunkSize);
        if (TIME_STAGING) {
            LOGGER.log(
                    System.Logger.Level.INFO,
                    "staging: per-layer rows for {0} tokens in {1} ms",
                    chunkSize,
                    (System.nanoTime() - stageStart) / 1e6);
        }

        if (useFallbackFamily) {
            plan.tornadoVMForwardBatchPrefillFallback();
        } else {
            plan.tornadoVMForwardBatchPrefill();
        }
    }

    /**
     * The decode step of the batched path: stage one token's embedding, then run the decode
     * activation, layer and logits graphs.
     *
     * <p>Returns the neutral {@link Logits} view over the array the plan produced. The logits stay
     * <b>device-resident</b> exactly as before — the view reads the same {@code FloatArray} in
     * place, and no readback, copy or synchronization is added or removed.
     *
     * @param model the model
     * @param state the session's state
     * @param token current token id
     * @param position sequence position
     * @param plan the batched prefill/decode GPU plan
     * @return the logits this invocation produced, for sampling
     */
    public static Logits decode(
            Model model,
            State state,
            int token,
            int position,
            TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        // The same per-token staging the single-token pass does first: a family with a device input
        // besides the token embedding needs it on every decode step, batched plan or not.
        model.stagePerTokenDeviceInputs(state, token);

        switch (weights.getTokenEmbeddingTable().dataType()) {
            case F16 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                MemorySegment.copy(
                        embTable,
                        (long) token * config.dim() * Short.BYTES,
                        state.workspace.embeddingX.getSegment(),
                        0L,
                        (long) config.dim() * Short.BYTES);
            }
            case Q8_0 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blocksPerToken = (config.dim() + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;
                MemorySegment.copy(
                        embTable,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0L,
                        bytesPerToken);
            }
            case Q4_0 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blocksPerToken = (config.dim() + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
                long bytesPerToken = (long) blocksPerToken * Q4_0_BLOCK_BYTES;
                MemorySegment.copy(
                        embTable,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0L,
                        bytesPerToken);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported embedding weight type: "
                                    + weights.getTokenEmbeddingTable().dataType());
        }

        return state.workspace.logitsView(plan.tornadoVMForwardDecode(position));
    }
}

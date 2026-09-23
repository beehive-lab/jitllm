package org.beehive.jitllm.model.gemma4;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.TokenGenerationLoop;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.Gemma4State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jitllm.model.AbstractModel;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

public class Gemma4 extends AbstractModel {

    Gemma4Configuration configuration;

    public Gemma4(
            Gemma4Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public Gemma4Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_4;
    }

    @Override
    public Gemma4Tokenizer tokenizer() {
        return (Gemma4Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Gemma4State(configuration(), -1);
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma4State(configuration(), batchsize);
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    /**
     * Gathers the current token's row out of {@code per_layer_token_embd} (~2.35 billion elements
     * -- far too large to keep resident on the GPU, see {@link
     * Gemma4TornadoWeights#perLayerTokenEmbd}) directly into {@link
     * Gemma4State#wrapPerLayerTokenEmbedRow}, pre-scaled by {@code sqrt(embeddingLengthPerLayer)}
     * (mirroring step 2 of {@code InferenceCore.forwardJavaGemma4}), ready for transfer to the GPU
     * as part of layer 0's per-layer-embedding setup.
     */
    @Override
    public void stagePerTokenDeviceInputs(org.beehive.jitllm.inference.state.State state, int token) {
        gatherPerLayerTokenEmbeddingRow((Gemma4State) state, token);
    }

    // @formatter:off
    /**
     * The same gather for every token of a prefill chunk, into one row per token.
     *
     * <p>This is the one part of a batched prefill that does not get cheaper per token: the table
     * is 2.35 billion elements and stays on the host, so a chunk of B tokens costs B row gathers
     * just as B single-token steps would. It is the host-side floor under this family's prefill
     * rate, and naming it here is what makes it measurable rather than mysterious.
     */
    // @formatter:on
    @Override
    public void stageBatchDeviceInputs(
            org.beehive.jitllm.inference.state.State state, int[] tokens, int chunkSize) {
        Gemma4State gemma4State = (Gemma4State) state;
        int nEmbdPerLayer = configuration.embeddingLengthPerLayer();
        int perLayerTotal = configuration.numberOfLayers() * nEmbdPerLayer;
        float scale = (float) Math.sqrt(nEmbdPerLayer);
        Gemma4TornadoWeights gemma4Weights = (Gemma4TornadoWeights) weights;
        // Across tokens, because a chunk's rows are independent and this is the one part of a
        // batched prefill that does not get cheaper per token: the table stays on the host, so B
        // tokens cost B row decodes just as B single-token steps would. Measured on a 512-token
        // chunk, sequentially: 9.10 ms when the table is Q8_0 and 31.77 ms when it is Q5_K, inside
        // the timed window and counted by no kernel profiler. Each row writes its own disjoint
        // slice of the destination and reads a tensor nothing mutates.
        org.beehive.jitllm.auxiliary.Parallel.parallelFor(
                0,
                chunkSize,
                b ->
                        org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader
                                .copyEmbeddingRowToFloatArray(
                                        gemma4Weights.perLayerTokenEmbd,
                                        tokens[b],
                                        perLayerTotal,
                                        gemma4State.workspace.wrapPerLayerTokenEmbedRowBatch,
                                        b * perLayerTotal,
                                        scale));
    }

    private void gatherPerLayerTokenEmbeddingRow(Gemma4State state, int token) {
        Gemma4TornadoWeights gemma4Weights = (Gemma4TornadoWeights) weights;
        int nEmbdPerLayer = configuration.embeddingLengthPerLayer();
        int perLayerTotal = configuration.numberOfLayers() * nEmbdPerLayer;
        float scale = (float) Math.sqrt(nEmbdPerLayer);
        org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader.copyEmbeddingRowToFloatArray(
                gemma4Weights.perLayerTokenEmbd,
                token,
                perLayerTotal,
                state.workspace.wrapPerLayerTokenEmbedRow,
                scale);
    }

    @Override
    public List<Integer> generateTokens(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {
        return TokenGenerationLoop.generateTokensQwen3(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        if (state.executionPolicy().phaseStrategy()
                == org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            // Prompt ingestion as its own phase, charging the whole prompt against the budget as
            // this family's decode loop does. Without this the batched plan is built and never
            // driven: the interleaved loop would run it by the single-token plan's graph indices.
            return TokenGenerationLoop.generateTokensGPUPrefillDecode(
                    this,
                    state,
                    startPosition,
                    promptTokens,
                    stopTokens,
                    maxTokens,
                    sampler,
                    echo,
                    onTokenGenerated,
                    tornadoVMPlan);
        }
        return TokenGenerationLoop.generateTokensGPUQwen3(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated,
                tornadoVMPlan);
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jitllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jitllm.runtime.model.ArchitectureId.of("gemma4");
    }
}

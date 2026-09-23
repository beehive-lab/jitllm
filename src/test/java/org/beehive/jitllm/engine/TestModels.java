package org.beehive.jllm.engine;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.tokenizer.Tokenizer;

/**
 * Models that answer the one question the engine asks at construction, and nothing else.
 *
 * <p>The engine only needs {@code supportsSharedKvStorage()} and an identity for its error message;
 * it never runs the model — that is the {@link BatchExecutor}'s job. So these stubs throw from
 * everything else rather than pretending, which keeps a future test honest: if the engine starts
 * calling the model, a stub will say so loudly instead of returning a plausible null.
 */
final class TestModels {

    private TestModels() {}

    /** A family whose layer graphs chain the block table: eligible for engine execution. */
    static Model sharedKvCapable() {
        return new StubModel(true);
    }

    /** A family that addresses through a table but does not chain it — Qwen2's position today. */
    static Model privateKvOnly() {
        return new StubModel(false);
    }

    private record StubModel(boolean sharedKv) implements Model {

        @Override
        public boolean supportsSharedKvStorage() {
            return sharedKv;
        }

        @Override
        public ModelType getModelType() {
            return ModelType.LLAMA_3;
        }

        @Override
        public Configuration configuration() {
            throw notNeeded("configuration");
        }

        @Override
        public Tokenizer tokenizer() {
            throw notNeeded("tokenizer");
        }

        @Override
        public Weights weights() {
            throw notNeeded("weights");
        }

        @Override
        public ChatFormat chatFormat() {
            throw notNeeded("chatFormat");
        }

        @Override
        public State createNewState() {
            throw notNeeded("createNewState");
        }

        @Override
        public State createNewState(int batchsize) {
            throw notNeeded("createNewState");
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
            throw notNeeded("generateTokens");
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
                TornadoVMMasterPlan plan) {
            throw notNeeded("generateTokensGPU");
        }

        private static UnsupportedOperationException notNeeded(String what) {
            return new UnsupportedOperationException(
                    "the engine should not be calling "
                            + what
                            + "(): it schedules and delivers, and the BatchExecutor runs the model."
                            + " If this fires, the engine grew a dependency it is not supposed to have");
        }
    }
}

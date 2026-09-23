package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.junit.Test;

/**
 * These four statements are the whole of the decision, and each of them is the kind of thing that
 * is easy to get subtly wrong: throwing but releasing anyway, force-closing to be helpful, failing
 * on a second close, or handing out a session after close.
 */
public class FacadeLifecycleTest {

    /** The smallest thing that satisfies {@code Model}; the facade only reads its configuration. */
    private static final class StubModel implements Model {

        private int statesCreated;

        @Override
        public Configuration configuration() {
            return STUB_CONFIGURATION;
        }

        @Override
        public Tokenizer tokenizer() {
            return null;
        }

        @Override
        public Weights weights() {
            return () -> org.beehive.jitllm.runtime.tensor.DataType.Q8_0;
        }

        @Override
        public ChatFormat chatFormat() {
            return null;
        }

        @Override
        public ModelType getModelType() {
            return ModelType.LLAMA_3;
        }

        @Override
        public State createNewState() {
            statesCreated++;
            return null;
        }

        @Override
        public State createNewState(int batchsize) {
            statesCreated++;
            return null;
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
            throw new UnsupportedOperationException("not exercised by the lifecycle tests");
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
            throw new UnsupportedOperationException("not exercised by the lifecycle tests");
        }
    }

    private static final Configuration STUB_CONFIGURATION =
            new Configuration() {
                @Override
                public String quantization() {
                    return "Q8_0";
                }

                @Override
                public int dim() {
                    return 2048;
                }

                @Override
                public int hiddenDim() {
                    return 8192;
                }

                @Override
                public int numberOfLayers() {
                    return 16;
                }

                @Override
                public int numberOfHeads() {
                    return 32;
                }

                @Override
                public int numberOfKeyValueHeads() {
                    return 8;
                }

                @Override
                public int numberOfHeadsKey() {
                    return 8;
                }

                @Override
                public int vocabularySize() {
                    return 128256;
                }

                @Override
                public int contextLength() {
                    return 512;
                }

                @Override
                public int contextLengthModel() {
                    return 131072;
                }

                @Override
                public float rmsNormEps() {
                    return 1e-5f;
                }

                @Override
                public float ropeTheta() {
                    return 500000f;
                }

                @Override
                public int headSize() {
                    return 64;
                }

                @Override
                public int kvDim() {
                    return 512;
                }

                @Override
                public int kvMul() {
                    return 4;
                }
            };

    private static TextGenerationModel model() {
        return new DelegatingModel(new StubModel(), Path.of("stub.gguf"), false);
    }

    private static TextGenerationModel model(int maxConcurrentSessions) {
        return new DelegatingModel(
                new StubModel(),
                Path.of("stub.gguf"),
                false,
                org.beehive.jitllm.runtime.policy.ExecutionPolicy.fromSystemProperties(),
                org.beehive.jitllm.runtime.policy.StorageOptions.fromSystemProperties(),
                ThinkingMode.DEFAULT,
                maxConcurrentSessions);
    }

    @Test
    public void preparationIsIdempotentAndRespectsSessionLifetime() {
        try (TextGenerationModel model = model()) {
            GenerationSession session = model.newSession();
            assertEquals("CPU", session.prepare().backend());
            assertEquals(session.prepare(), session.prepare());
            assertEquals(0, session.position());
            assertEquals("single-token", session.prepare().mode());
            session.close();
            assertThrows(IllegalStateException.class, session::prepare);
        }
    }

    @Test
    public void closingWithALiveSessionThrowsAndNamesIt() {
        TextGenerationModel model = model();
        GenerationSession session = model.newSession();

        IllegalStateException failure = assertThrows(IllegalStateException.class, model::close);
        assertTrue(failure.getMessage(), failure.getMessage().contains("session"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("open"));
    }

    /**
     * The failed close has no effect: the model is still usable, which is what makes retry sane.
     */
    @Test
    public void aFailedCloseLeavesTheModelOpen() {
        TextGenerationModel model = model(2);
        GenerationSession first = model.newSession();
        assertThrows(IllegalStateException.class, model::close);

        assertNotNull(
                "the model must still hand out sessions after a refused close", model.newSession());
        assertNotNull(model.info());
    }

    @Test
    public void closingTheSessionsFirstMakesTheModelCloseable() {
        TextGenerationModel model = model(2);
        GenerationSession a = model.newSession();
        GenerationSession b = model.newSession();
        a.close();
        assertThrows("one session still open", IllegalStateException.class, model::close);
        b.close();
        model.close();
    }

    /** One open session unless the caller asked for more, and the refusal says how to ask. */
    @Test
    public void aModelHoldsOneSessionUnlessToldOtherwise() {
        TextGenerationModel model = model();
        GenerationSession first = model.newSession();
        IllegalStateException refused =
                assertThrows(IllegalStateException.class, model::newSession);
        assertTrue(refused.getMessage(), refused.getMessage().contains("maxConcurrentSessions"));
        first.close();
        // The place is reusable once freed: a closed session releases its slot.
        model.newSession().close();
        model.close();
    }

    @Test
    public void aModelHoldsExactlyTheSessionsItWasLoadedFor() {
        TextGenerationModel model = model(3);
        GenerationSession a = model.newSession();
        GenerationSession b = model.newSession();
        GenerationSession c = model.newSession();
        assertThrows(IllegalStateException.class, model::newSession);
        b.close();
        GenerationSession d = model.newSession();
        a.close();
        c.close();
        d.close();
        model.close();
    }

    @Test
    public void aSessionCountBelowOneIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> model(0));
        assertThrows(
                IllegalArgumentException.class,
                () -> ModelOptions.builder().maxConcurrentSessions(0));
        assertEquals(1, ModelOptions.defaults().maxConcurrentSessions());
        assertEquals(
                4, ModelOptions.builder().maxConcurrentSessions(4).build().maxConcurrentSessions());
    }

    @Test
    public void closeIsIdempotentOnBothTheModelAndTheSession() {
        TextGenerationModel model = model();
        GenerationSession session = model.newSession();
        session.close();
        session.close(); // no-op, not an error
        model.close();
        model.close();
    }

    @Test
    public void noSessionMayBeCreatedAfterTheModelIsClosed() {
        TextGenerationModel model = model();
        model.close();
        IllegalStateException failure =
                assertThrows(IllegalStateException.class, model::newSession);
        assertTrue(failure.getMessage(), failure.getMessage().contains("closed"));
    }

    @Test
    public void aSessionOfAClosedModelRefusesToGenerate() {
        TextGenerationModel model = model();
        GenerationSession session = model.newSession();
        session.close();
        model.close();
        assertThrows(
                IllegalStateException.class, () -> session.generate(GenerationRequest.of("x")));
    }

    @Test
    public void aSessionMayNotOutgrowItsModelsContext() {
        TextGenerationModel model = model();
        assertThrows(
                IllegalArgumentException.class,
                () -> model.newSession(SessionOptions.builder().contextLength(4096).build()));
        assertNotNull(model.newSession(SessionOptions.builder().contextLength(256).build()));
    }

    @Test
    public void theModelReportsWhatItLoaded() {
        TextGenerationModel model = model();
        assertEquals(512, model.info().contextLength());
        assertEquals(org.beehive.jitllm.runtime.tensor.DataType.Q8_0, model.info().computeType());
        assertEquals(Path.of("stub.gguf"), model.info().source());
        assertEquals(16, model.configuration().layers());
        assertEquals(8, model.configuration().keyValueHeads());
        assertEquals(131072, model.configuration().maxContextLength());
    }
}

package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.backend.cpu.Glm5NextForward;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.Glm5NextStandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * The {@code glm5next} decode plan against the host forward pass over the tiny reference model's
 * Q8_0 files — the whole model, and one-layer slices of it — for 40 tokens, long enough for the
 * indexer to drop pools.
 *
 * <p>The latent cache is FP32, which the plan requires. The device projections quantize their
 * activations to int8; the host here is made to round them the same way, so a position agrees to
 * float rounding unless a value upstream sat on an int8 rounding boundary, where the two paths'
 * last-bit differences can round it apart. That moves one element by 1/127 of its block and the
 * logits by about a percent, and it fades within a few positions as the convolution window and the
 * decaying KDA state forget it. The slices are therefore held to exactness at most positions, and
 * the whole model to its first token and its greedy choices.
 */
public class Glm5NextDecodeParityAccelTest {

    private static final int CONTEXT = 64;

    /**
     * One KDA layer with a dense feed-forward. Where no rounding flipped, the device equals the
     * host to float rounding, which is most positions.
     */
    @Test
    public void aKdaLayerMatchesTheHost() throws Exception {
        float[] errors = run("model-Q8_0-layer0.gguf").errors;
        assertTrue("exact at too few positions", exactPositions(errors, 0) >= errors.length / 4);
        assertTrue("worst " + max(errors), max(errors) < 0.05f);
    }

    /**
     * One sparse-attention layer with the mixture of experts, exact at some position after the
     * indexer has started dropping pools (position 11 on, with a budget of two pools of four).
     */
    @Test
    public void aSparseAttentionLayerMatchesTheHost() throws Exception {
        float[] errors = run("model-Q8_0-layer3.gguf").errors;
        assertTrue("exact at too few positions", exactPositions(errors, 0) >= errors.length / 4);
        assertTrue("never exact once pools are dropped", exactPositions(errors, 12) > 0);
        assertTrue("worst " + max(errors), max(errors) < 0.05f);
    }

    /** The whole tiny model: exact at the first token, and the same greedy choices. */
    @Test
    public void theWholeModelDecodesLikeTheHost() throws Exception {
        Result r = run("model-Q8_0.gguf");
        assertTrue("first token " + r.errors[0], r.errors[0] < 1e-4f);
        assertTrue("greedy choices agree at " + r.agree, r.agree >= r.errors.length * 9 / 10);
    }

    /** Prefilled in chunks of eight (two whole, one partial), then decoded token by token. */
    private static final int PREFILL = 17;

    private static final int CHUNK = 8;

    /** A KDA layer after a batched prefill: the state the chunks leave is the one decode needs. */
    @Test
    public void aKdaLayerDecodesLikeTheHostAfterABatchedPrefill() throws Exception {
        float[] errors = run("model-Q8_0-layer0.gguf", PREFILL, CHUNK).errors;
        assertTrue("exact at too few positions", exactPositions(errors, PREFILL) >= 3);
        assertTrue("worst " + max(errors), max(errors) < 0.05f);
    }

    /** A sparse-attention layer after a batched prefill: the chunks' latents and pool keys. */
    @Test
    public void aSparseAttentionLayerDecodesLikeTheHostAfterABatchedPrefill() throws Exception {
        float[] errors = run("model-Q8_0-layer3.gguf", PREFILL, CHUNK).errors;
        assertTrue("exact at too few positions", exactPositions(errors, PREFILL) >= 3);
        assertTrue("worst " + max(errors), max(errors) < 0.05f);
    }

    /** The whole tiny model after a batched prefill: the same greedy choices. */
    @Test
    public void theWholeModelDecodesLikeTheHostAfterABatchedPrefill() throws Exception {
        Result r = run("model-Q8_0.gguf", PREFILL, CHUNK);
        int decoded = r.errors.length - PREFILL;
        assertTrue("greedy choices agree at " + r.agree, r.agree >= decoded * 8 / 10);
    }

    /**
     * The sequential prefill/decode plan, whose layers are one task graph for the whole trunk: it
     * must run every graph the layers built, not one per layer.
     */
    @Test
    public void theWholeModelDecodesLikeTheHostInPrefillDecodeMode() throws Exception {
        Result r = run("model-Q8_0.gguf", 0, 1, true);
        assertTrue("first token " + r.errors[0], r.errors[0] < 1e-4f);
        assertTrue("greedy choices agree at " + r.agree, r.agree >= r.errors.length * 9 / 10);
    }

    private record Result(float[] errors, int agree) {}

    /** Positions from {@code from} whose logits agree with the host to float rounding. */
    private static int exactPositions(float[] errors, int from) {
        int n = 0;
        for (int t = from; t < errors.length; t++) {
            n += errors[t] < 1e-4f ? 1 : 0;
        }
        return n;
    }

    private static float max(float[] errors) {
        float m = 0f;
        for (float e : errors) {
            m = Math.max(m, e);
        }
        return m;
    }

    private static Result run(String modelFile) throws Exception {
        return run(modelFile, 0, 1, false);
    }

    private static Result run(String modelFile, int prefill, int chunk) throws Exception {
        return run(modelFile, prefill, chunk, chunk > 1);
    }

    // @formatter:off
    /**
     * Runs the golden tokens on the device and on the int8-emulating host. The device prefills the
     * first {@code prefill} in chunks of {@code chunk} through the batched plan when {@code chunk >
     * 1}, then decodes the rest; the errors before {@code prefill} are zero, since a prefill
     * produces no logits.
     */
    // @formatter:on
    private static Result run(String modelFile, int prefill, int chunk, boolean prefillDecode)
            throws Exception {
        Path file = resource(modelFile);
        int[] tokens =
                (int[])
                        GGUF.loadGGUFMetadata(resource("goldens.gguf"))
                                .getMetadata()
                                .get("golden.tokens");

        Model host = ModelLoader.loadModel(file, CONTEXT, true, false);
        emulateInt8Activations((Glm5NextStandardWeights) host.weights());
        State hostState = host.createNewState();

        System.setProperty("use.tornadovm", "true");
        Model device = ModelLoader.loadModel(file, CONTEXT, true, true);
        // The latent cache is FP32; the default FP16 cache is refused.
        State deviceState =
                State.withPrefillBatchSize(
                        chunk,
                        () ->
                                State.withStorageOptions(
                                        StorageOptions.fp32(), device::createNewState));
        if (prefillDecode) {
            deviceState.resolveExecutionPolicy(
                    ExecutionPolicy.builder()
                            .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                            .prefillBatchSize(chunk)
                            .build());
        }
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(deviceState, device);

        float[] errors = new float[tokens.length];
        int agree = 0;
        try {
            if (prefill > 0) {
                TornadoBatchPrefillPass.batchPrefillAll(
                        device,
                        deviceState,
                        java.util.Arrays.copyOf(tokens, prefill),
                        0,
                        prefill,
                        chunk,
                        (BatchPrefillDecodePlan) plan);
            }
            for (int t = 0; t < tokens.length; t++) {
                FloatTensor expected = Glm5NextForward.forward(host, hostState, tokens[t], t);
                if (t < prefill) {
                    continue;
                }
                Logits actual = TornadoForwardPass.forward(device, deviceState, tokens[t], t, plan);
                float scale = 0f;
                float diff = 0f;
                int bestHost = 0;
                int bestDevice = 0;
                for (int i = 0; i < actual.size(); i++) {
                    scale = Math.max(scale, Math.abs(expected.getFloat(i)));
                    diff = Math.max(diff, Math.abs(expected.getFloat(i) - actual.get(i)));
                    bestHost = expected.getFloat(i) > expected.getFloat(bestHost) ? i : bestHost;
                    bestDevice = actual.get(i) > actual.get(bestDevice) ? i : bestDevice;
                }
                errors[t] = diff / scale;
                agree += bestHost == bestDevice ? 1 : 0;
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
        System.out.println(
                "[glm5next] "
                        + modelFile
                        + ": "
                        + java.util.Arrays.toString(errors)
                        + ", greedy agreement "
                        + agree
                        + "/"
                        + tokens.length);
        return new Result(errors, agree);
    }

    // @formatter:off
    /**
     * Wraps every host projection whose device kernel quantizes its input — all but {@code
     * attn_k_b} and {@code attn_v_b}, which read the query and the latent in FP32 — so the host
     * rounds the activation to int8 per 32-block exactly as the device does. The two paths then
     * make the same discrete choices (routed experts, indexer pools) and differ by summation order
     * only.
     */
    // @formatter:on
    private static void emulateInt8Activations(Glm5NextStandardWeights weights)
            throws ReflectiveOperationException {
        // The logits kernel quantizes the final hidden state too.
        var output = Glm5NextStandardWeights.class.getDeclaredField("output");
        output.setAccessible(true);
        output.set(weights, new Int8Activations(weights.output));
        var w = weights.layers;
        var k = w.kda();
        var s = w.sparse();
        var f = w.ffn();
        for (FloatTensor[] slots :
                List.of(
                        k.q(),
                        k.k(),
                        k.v(),
                        k.decayA(),
                        k.decayB(),
                        k.beta(),
                        k.gateA(),
                        k.gateB(),
                        k.output(),
                        s.qA(),
                        s.qB(),
                        s.kvA(),
                        s.output(),
                        s.indexQ(),
                        s.indexK(),
                        s.indexWeights(),
                        s.poolGate(),
                        f.gate(),
                        f.up(),
                        f.down(),
                        f.sharedGate(),
                        f.sharedUp(),
                        f.sharedDown())) {
            for (int l = 0; l < slots.length; l++) {
                if (slots[l] != null) {
                    slots[l] = new Int8Activations(slots[l]);
                }
            }
        }
        for (List<List<FloatTensor>> stacks :
                List.of(f.gateExperts(), f.upExperts(), f.downExperts())) {
            for (List<FloatTensor> layer : stacks) {
                layer.replaceAll(Int8Activations::new);
            }
        }
    }

    /** A weight whose dot products first round the activation to int8 per 32-block. */
    private static final class Int8Activations extends FloatTensor {
        private final FloatTensor weight;

        Int8Activations(FloatTensor weight) {
            this.weight = weight;
        }

        @Override
        public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
            float result = 0f;
            for (int b = 0; b < size; b += 32) {
                float max = 0f;
                for (int j = 0; j < 32; j++) {
                    max = Math.max(max, Math.abs(that.getFloat(thatOffset + b + j)));
                }
                float inverse = max > 0f ? 127f / max : 0f;
                float scale = max / 127f;
                float block = 0f;
                for (int j = 0; j < 32; j++) {
                    float scaled = that.getFloat(thatOffset + b + j) * inverse;
                    int q = (int) (scaled + (scaled >= 0f ? 0.5f : -0.5f));
                    q = Math.min(127, Math.max(-127, q));
                    block += weight.getFloat(thisOffset + b + j) * q;
                }
                result += block * scale;
            }
            return result;
        }

        @Override
        public int size() {
            return weight.size();
        }

        @Override
        public float getFloat(int index) {
            return weight.getFloat(index);
        }

        @Override
        public void setFloat(int index, float value) {
            throw new UnsupportedOperationException("a weight");
        }

        @Override
        protected FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
            throw new UnsupportedOperationException("read through dot");
        }

        @Override
        protected GGMLType type() {
            return GGMLType.Q8_0;
        }

        @Override
        public MemorySegment asMemorySegment() {
            return weight.asMemorySegment();
        }
    }

    private static Path resource(String file) throws URISyntaxException {
        return Path.of(
                Glm5NextDecodeParityAccelTest.class
                        .getResource("/goldens/glm5next-tiny/" + file)
                        .toURI());
    }
}

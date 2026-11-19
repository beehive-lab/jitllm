package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.jitllm.backend.cpu.DeepSeek4Forward;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.golden.Fp32KeyValueCache;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.DeepSeek4StandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.DeepSeek4ModelLoader;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.ClassRule;
import org.junit.Test;

/**
 * The {@code deepseek4} decode plan against the host forward pass over the tiny reference model's
 * Q8_0 file, for 160 tokens: long enough for the indexer to choose 4 of up to 40 blocks and for the
 * ratio-128 layer to complete its first block.
 *
 * <p>The device projections quantize their activations to int8; the host here is made to round them
 * the same way, so a position agrees to float rounding unless a value upstream sat on an int8
 * rounding boundary, where the two paths' last-bit differences round it apart. That moves one
 * element by 1/127 of its block, and the attention window and the compressed blocks keep the
 * difference, so later positions differ by about a percent. The model is held to exactness at its
 * first token and to its greedy choices. Both keep the attention window in FP32.
 *
 * <p>The batched plan prefills the first tokens in chunks wider than the window, past the ratio-128
 * layer's first block, and then decodes; the sequential prefill/decode plan runs the decode graphs
 * from the first token. Split into stages on one device, both plans must compute what they do on
 * one stage.
 */
public class DeepSeek4DecodeParityAccelTest {

    @ClassRule public static final Fp32KeyValueCache FP32 = new Fp32KeyValueCache();

    private static final int CONTEXT = 256;
    private static final int TOKENS = 160;

    /**
     * The batched prefill's prompt: past the ratio-128 layer's first block, in chunks four times
     * the tiny model's window of 8, the last of them partial.
     */
    private static final int PREFILL = 136;

    private static final int CHUNK = 32;

    /**
     * The devices of the three-stage and the two-stage split, and the transport between them: one
     * device and host memory unless set, so that a machine with several GPUs can run the split
     * across them.
     */
    private static final String THREE_STAGES =
            System.getProperty("deepseek4.test.devices3", "0:0,0:0,0:0");

    private static final String TWO_STAGES =
            System.getProperty("deepseek4.test.devices2", "0:0,0:0");

    private static final String TRANSPORT = System.getProperty("deepseek4.test.transport", "host");

    /** Exact at the first token, close after it, and the same greedy choices. */
    @Test
    public void theWholeModelDecodesLikeTheHost() throws Exception {
        Result r = run("model-Q8_0.gguf");
        assertTrue("first token " + r.errors[0], r.errors[0] < 1e-5f);
        assertTrue("median error " + median(r.errors), median(r.errors) < 0.05f);
        assertTrue("greedy choices agree at " + r.agree, r.agree >= TOKENS * 9 / 10);
    }

    /** Loaded as two stacks of four experts: the kernels read both. */
    @Test
    public void twoExpertStacksDecodeTheSame() throws Exception {
        Result one = run("model-Q8_0.gguf");
        Result two = run("model-Q8_0.gguf", 20_000);
        for (int t = 0; t < TOKENS; t++) {
            assertTrue(
                    "position " + t + ": " + one.errors[t] + " and " + two.errors[t],
                    Math.abs(one.errors[t] - two.errors[t]) < 1e-5f);
        }
    }

    /**
     * Decoded after a batched prefill: the window, the pending compressor entries and the
     * compressed and indexer keys the chunks leave are the ones decode continues from.
     */
    @Test
    public void theWholeModelDecodesLikeTheHostAfterABatchedPrefill() throws Exception {
        Result r = run("model-Q8_0.gguf", 0, PREFILL, CHUNK, true);
        int decoded = TOKENS - PREFILL;
        assertTrue("median error " + median(r.errors, PREFILL), median(r.errors, PREFILL) < 0.05f);
        assertTrue("greedy choices agree at " + r.agree, r.agree >= decoded * 8 / 10);
    }

    /** The batched prefill with the experts in two stacks. */
    @Test
    public void twoExpertStacksDecodeLikeTheHostAfterABatchedPrefill() throws Exception {
        Result r = run("model-Q8_0.gguf", 20_000, PREFILL, CHUNK, true);
        int decoded = TOKENS - PREFILL;
        assertTrue("greedy choices agree at " + r.agree, r.agree >= decoded * 8 / 10);
    }

    /**
     * The medium model, whose widths are whole GEMM tiles, so its batched prefill runs the routed
     * experts on the grouped int8 GEMMs and the shared expert on the int8 GEMMs: decoded after the
     * prefill, it must track the host as closely as decoding every token one at a time does.
     */
    @Test
    public void theMediumModelDecodesTheSameAfterABatchedPrefill() throws Exception {
        assertTracksTokenByToken(MEDIUM, 0);
    }

    /** The medium model's grouped experts loaded as two stacks: each stack runs its own tiles. */
    @Test
    public void theMediumModelWithTwoExpertStacksDecodesTheSameAfterABatchedPrefill()
            throws Exception {
        assertTracksTokenByToken(MEDIUM, MEDIUM_STACK_BYTES);
    }

    /** The medium model with Q4_0 routed experts, which the grouped GEMMs read as Q4_0. */
    @Test
    public void theMediumModelWithQ4_0ExpertsDecodesTheSameAfterABatchedPrefill() throws Exception {
        assertTracksTokenByToken(MEDIUM_Q4_0_EXPERTS, 0);
    }

    private static final String MEDIUM = "model-medium-Q8_0.gguf";

    private static final String MEDIUM_Q4_0_EXPERTS = "model-medium-Q8_0-experts-Q4_0.gguf";

    /** Four of the medium model's eight experts per stack. */
    private static final long MEDIUM_STACK_BYTES = 150_000;

    private static void assertTracksTokenByToken(String model, long stackBytes) throws Exception {
        float[] single = run(model, stackBytes).errors;
        Result batched = run(model, stackBytes, PREFILL, CHUNK, true);
        float singleMedian = median(single, PREFILL);
        float batchedMedian = median(batched.errors, PREFILL);
        assertTrue(
                "median error " + batchedMedian + " against " + singleMedian + " token by token",
                batchedMedian <= 1.25f * singleMedian);
        int decoded = TOKENS - PREFILL;
        assertTrue("greedy choices agree at " + batched.agree, batched.agree >= decoded * 8 / 10);
    }

    /** The sequential prefill/decode plan runs the same trunk graph token by token. */
    @Test
    public void theWholeModelDecodesLikeTheHostInPrefillDecodeMode() throws Exception {
        Result r = run("model-Q8_0.gguf", 0, 0, 1, true);
        assertTrue("first token " + r.errors[0], r.errors[0] < 1e-5f);
        assertTrue("greedy choices agree at " + r.agree, r.agree >= TOKENS * 9 / 10);
    }

    /**
     * Split into three stages on one device: the residual streams cross two stage boundaries every
     * token, and every logit must be what the plan on one stage computes.
     */
    @Test
    public void theWholeModelDecodesTheSameSplitIntoStages() throws Exception {
        Result one = run("model-Q8_0.gguf");
        Result three = run("model-Q8_0.gguf", 0, 0, 1, false, THREE_STAGES);
        assertArrayEquals("errors against the host", one.errors, three.errors, 0f);
        assertEquals("greedy agreement", one.agree, three.agree);
    }

    /**
     * The batched prefill and decode split into two stages on one device: the chunks' streams and
     * the decoded tokens' streams cross the boundary, and the hash-routed layers' experts reach
     * both stages.
     */
    @Test
    public void theBatchedPlanPrefillsAndDecodesTheSameSplitIntoStages() throws Exception {
        Result one = run("model-Q8_0.gguf", 0, PREFILL, CHUNK, true);
        Result two = run("model-Q8_0.gguf", 0, PREFILL, CHUNK, true, TWO_STAGES);
        assertArrayEquals("errors against the host", one.errors, two.errors, 0f);
        assertEquals("greedy agreement", one.agree, two.agree);
    }

    private record Result(float[] errors, int agree) {}

    private static float median(float[] errors) {
        return median(errors, 0);
    }

    /** The median of the errors from position {@code from} on. */
    private static float median(float[] errors, int from) {
        float[] sorted = java.util.Arrays.copyOfRange(errors, from, errors.length);
        java.util.Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static Result run(String modelFile) throws Exception {
        return run(modelFile, 0);
    }

    private static Result run(String modelFile, long stackBytes) throws Exception {
        return run(modelFile, stackBytes, 0, 1, false);
    }

    // @formatter:off
    /**
     * Runs the golden tokens on the device and on the int8-emulating host. With {@code stackBytes >
     * 0} the experts load as stacks of at most that size. With {@code prefillDecode} the plan is
     * the batched one for {@code chunk > 1}, which prefills the first {@code prefill} tokens in
     * chunks of {@code chunk} before decoding the rest, and the sequential prefill/decode one
     * otherwise. A prefilled position has no logits, and its error stays zero.
     */
    // @formatter:on
    private static Result run(
            String modelFile, long stackBytes, int prefill, int chunk, boolean prefillDecode)
            throws Exception {
        return run(modelFile, stackBytes, prefill, chunk, prefillDecode, null);
    }

    /** {@link #run(String, long, int, int, boolean)} with the plan split across {@code devices}. */
    private static Result run(
            String modelFile,
            long stackBytes,
            int prefill,
            int chunk,
            boolean prefillDecode,
            String devices)
            throws Exception {
        Path file = resource(modelFile);
        int[] tokens =
                (int[])
                        GGUF.loadGGUFMetadata(resource("goldens.gguf"))
                                .getMetadata()
                                .get("golden.tokens");
        if (stackBytes > 0) {
            System.setProperty(
                    DeepSeek4ModelLoader.STACK_BYTES_PROPERTY, Long.toString(stackBytes));
        }
        Model host;
        Model device;
        try {
            host = ModelLoader.loadModel(file, CONTEXT, true, false);
            System.setProperty("use.tornadovm", "true");
            device = ModelLoader.loadModel(file, CONTEXT, true, true);
        } finally {
            System.clearProperty(DeepSeek4ModelLoader.STACK_BYTES_PROPERTY);
        }
        emulateInt8Activations((DeepSeek4StandardWeights) host.weights());
        State hostState = host.createNewState();
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
        if (devices != null) {
            System.setProperty(DeviceSplit.DEVICES_PROPERTY, devices);
            System.setProperty(DeviceSplit.TRANSPORT_PROPERTY, TRANSPORT);
        }
        TornadoVMMasterPlan plan;
        try {
            plan = TornadoVMMasterPlan.initializeTornadoVMPlan(deviceState, device);
        } finally {
            System.clearProperty(DeviceSplit.DEVICES_PROPERTY);
            System.clearProperty(DeviceSplit.TRANSPORT_PROPERTY);
        }

        float[] errors = new float[TOKENS];
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
            for (int t = 0; t < TOKENS; t++) {
                FloatTensor expected = DeepSeek4Forward.forward(host, hostState, tokens[t], t);
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
            System.clearProperty("use.tornadovm");
        }
        System.out.println(
                "[deepseek4] "
                        + modelFile
                        + ": "
                        + java.util.Arrays.toString(errors)
                        + ", greedy agreement "
                        + agree
                        + "/"
                        + (TOKENS - prefill));
        return new Result(errors, agree);
    }

    /**
     * Wraps every host projection whose device kernel quantizes its input — all but the
     * compressors, the indexer's head weights and the router, which the device reads in F32 — so
     * the host rounds the activation to int8 per 32-block exactly as the device does.
     */
    private static void emulateInt8Activations(DeepSeek4StandardWeights weights)
            throws ReflectiveOperationException {
        var output = DeepSeek4StandardWeights.class.getDeclaredField("output");
        output.setAccessible(true);
        output.set(weights, new Int8Activations(weights.output));
        var w = weights.layers;
        var a = w.attention();
        var x = w.indexer();
        var f = w.ffn();
        for (FloatTensor[] slots :
                List.of(
                        a.qA(),
                        a.qB(),
                        a.kv(),
                        a.outputA(),
                        a.outputB(),
                        x.qB(),
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
                if (layer != null) {
                    layer.replaceAll(Int8Activations::new);
                }
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
                DeepSeek4DecodeParityAccelTest.class
                        .getResource("/goldens/deepseek4-tiny/" + file)
                        .toURI());
    }
}

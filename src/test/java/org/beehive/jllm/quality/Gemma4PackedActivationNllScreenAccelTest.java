package org.beehive.jllm.quality;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.backend.tornado.TornadoForwardPass;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.golden.TupleInfo;
import org.beehive.jllm.inference.Logits;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

// @formatter:off
/**
 * A bounded teacher-forced quality screen for Gemma 4's packed-integer projections.
 *
 * <p>Those projections quantize the normalized activation to eight bits per block of 32 and do the
 * dot product in packed integers. The parity gate shows the logits move — that is what {@code
 * Q4_0_PACKED_ACTIVATION} is for — and shows that the decisions do not: zero argmax disagreements
 * over 63 rows, and greedy output token-identical to the host. <b>Neither is proof of quality
 * equivalence.</b> Retaining a token choice says the top of the distribution survived; it says
 * nothing about the rest of it, which is what sampling reads.
 *
 * <p>So this scores the whole distribution. Negative log-likelihood of held-out text under both
 * paths on the same file, teacher-forced through identical token ids, reported per passage and
 * pooled. The host decodes the file's own Q4_0 blocks with no activation packing, so it is the
 * reference; the device is what the packing changed.
 *
 * <p>Unlike the Qwen 3.5 screen, there is no property to switch here: this family selects the
 * packed path from the weights' representation and a device capability, so the comparison is
 * against the host rather than against the same device in another configuration. That is the
 * stronger reference anyway — a GPU-versus-GPU comparison cannot see a defect that moves the whole
 * GPU.
 *
 * <p>The bound is deliberately loose and is a <b>regression</b> limit, not a quality claim: what is
 * asserted is that the packed path does not make held-out text materially less likely. Every number
 * is printed whether it passes or not.
 */
// @formatter:on
public class Gemma4PackedActivationNllScreenAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    /** Tokens per passage, after tokenizing from the recorded offset. */
    private static final int TOKENS = 256;

    /**
     * How much worse the packed path's pooled NLL may be, in nats per token.
     *
     * <p>0.02 nats is about a 2% change in perplexity. Chosen as a regression limit around a
     * measured difference, not calibrated against any quality benchmark — the measured value is
     * printed on every run, so a drift toward this limit is visible long before it trips.
     */
    private static final double MAX_POOLED_NLL_INCREASE = 0.02;

    private record Passage(String name, String path, int byteOffset, int byteLength) {}

    private static final List<Passage> PASSAGES =
            List.of(
                    new Passage("prose", "README.md", 0, 4000),
                    new Passage(
                            "java-source",
                            "src/main/java/org/beehive/jllm/backend/tornado/kernels/"
                                    + "TransformerComputeKernelsQ4_0.java",
                            0,
                            4000),
                    new Passage("verification-prose", "docs/architecture/verification.md", 0, 4000),
                    new Passage("changelog", "CHANGELOG.md", 0, 4000),
                    new Passage("architecture", "docs/architecture/architecture.md", 0, 4000));

    @Test
    public void thePackedPathDoesNotMakeHeldOutTextLessLikely() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.GEMMA_4_E2B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.GEMMA_4_E2B_Q4_0));
            assumeTrue("environment absent: fixture", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] environment absent — no TornadoVM device");
            assumeTrue("environment absent: no accelerator", false);
        }
        boolean packed =
                TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.PACKED_INTEGER_DOT);
        System.out.printf(
                "[NLL] model=%s packedIntegerDot=%s device=%s%n",
                modelPath.getFileName(), packed, TupleInfo.deviceName());
        if (!packed) {
            System.out.println(
                    "[SKIP] this device does not grant PACKED_INTEGER_DOT, so the path "
                            + "this screen exists to score is not the one that would run");
            assumeTrue("environment absent: no packed-integer path", false);
        }

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        double deviceTotal = 0;
        double hostTotal = 0;
        long counted = 0;
        try {
            Model gpuModel = ModelLoader.loadModel(modelPath, 1024, true, true);
            State gpuState = gpuModel.createNewState();
            TornadoVMMasterPlan plan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(gpuState, gpuModel);
            Model cpuModel = ModelLoader.loadModel(modelPath, 1024, true, false);

            try {
                for (Passage passage : PASSAGES) {
                    int[] tokens = tokenize(gpuModel, passage);
                    State cpuState = cpuModel.createNewState();

                    double deviceSum = 0;
                    double hostSum = 0;
                    int n = 0;
                    for (int[] pair : NllScoring.scoredPositions(tokens)) {
                        Logits deviceLogits =
                                TornadoForwardPass.forward(
                                        gpuModel, gpuState, tokens[pair[0]], pair[0], plan);
                        deviceSum += NllScoring.negativeLogLikelihood(toRow(deviceLogits), pair[1]);

                        FloatTensor hostLogits =
                                InferenceCore.forwardJavaGemma4(
                                        cpuModel, cpuState, tokens[pair[0]], pair[0]);
                        hostSum += NllScoring.negativeLogLikelihood(toRow(hostLogits), pair[1]);
                        n++;
                    }
                    deviceTotal += deviceSum;
                    hostTotal += hostSum;
                    counted += n;
                    System.out.printf(
                            "[NLL] %-20s tokens=%3d  host=%.5f  packed=%.5f  delta=%+.5f nats/token%n",
                            passage.name(),
                            n,
                            hostSum / n,
                            deviceSum / n,
                            (deviceSum - hostSum) / n);
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }

        double hostMean = hostTotal / counted;
        double deviceMean = deviceTotal / counted;
        double delta = deviceMean - hostMean;
        System.out.printf(
                "[NLL] pooled over %d tokens: host=%.5f  packed=%.5f  delta=%+.5f nats/token "
                        + "(perplexity ratio %.4f)%n",
                counted, hostMean, deviceMean, delta, Math.exp(delta));
        assertTrue(
                String.format(
                        "packed path raised pooled NLL by %+.5f nats/token, above the %.3f "
                                + "regression limit (host %.5f, packed %.5f over %d tokens)",
                        delta, MAX_POOLED_NLL_INCREASE, hostMean, deviceMean, counted),
                delta <= MAX_POOLED_NLL_INCREASE);
    }

    private static int[] tokenize(Model model, Passage passage) throws Exception {
        byte[] raw = Files.readAllBytes(Paths.get(passage.path()));
        assertTrue(
                passage.path() + " is shorter than the recorded range",
                raw.length >= passage.byteOffset() + passage.byteLength());
        String text =
                new String(raw, passage.byteOffset(), passage.byteLength(), StandardCharsets.UTF_8);
        List<Integer> encoded = model.tokenizer().encodeAsList(text);
        assertTrue(
                passage.name() + " tokenizes to " + encoded.size() + ", fewer than " + TOKENS,
                encoded.size() >= TOKENS);
        int[] tokens = new int[TOKENS];
        for (int i = 0; i < TOKENS; i++) {
            tokens[i] = encoded.get(i);
        }
        return tokens;
    }

    private static float[] toRow(Logits logits) {
        float[] row = new float[logits.size()];
        for (int i = 0; i < row.length; i++) {
            row[i] = logits.get(i);
        }
        return row;
    }

    private static float[] toRow(FloatTensor logits) {
        float[] row = new float[logits.size()];
        for (int i = 0; i < row.length; i++) {
            row[i] = logits.getFloat(i);
        }
        return row;
    }
}

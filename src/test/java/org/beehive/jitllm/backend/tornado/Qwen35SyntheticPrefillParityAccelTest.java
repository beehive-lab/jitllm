package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import org.beehive.jitllm.backend.cpu.Qwen35Forward;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.model.qwen35.Qwen35;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

// @formatter:off
/**
 * Sequential prefill on the device against the host running the same sequence one token at a time.
 *
 * <p>Prefill is the decode graphs with the logits graph skipped, so what this can catch is not
 * arithmetic — the parity test beside it already settles that — but the two things prompt ingestion
 * adds: whether the recurrence carries across the prefill/decode boundary, and whether the boundary
 * lands on the right position.
 *
 * <p><b>The state after ingestion is checked through the decode rows it produces.</b> The
 * convolution window and the delta-net matrices are device buffers no graph reads back, so
 * asserting on them directly would mean adding a transfer that production does not make. They are
 * instead observed where they are consumed: a wrong window or a stale state changes the first
 * decoded logits, and every one after it. Three decode rows are compared rather than one, because
 * the first would also agree if the recurrence had been frozen rather than carried.
 */
// @formatter:on
public class Qwen35SyntheticPrefillParityAccelTest {

    /** Prompt tokens, then the positions decode continues at. */
    private static final int[] PROMPT = {7, 91, 5, 42, 63};

    private static final int DECODE_ROWS = 3;

    @Test
    public void sequentialPrefillLeavesTheStateDecodeExpects() throws Exception {
        String previousDevice = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        // These cases compare the device against the host exactly: their subject is addressing
        // and chunk invariance, not arithmetic. A quantized activation cannot be exact, so the
        // Q4_0 projections stay on the floating-point path here. Their precision is covered on the
        // real model by the parity tests, against bounds written for it. This class gets its own
        // JVM (reuseForks=false), so the property is read before the layer builder loads.
        System.setProperty("jitllm.qwen35.packedIntegerDot", "false");
        try (Arena owned = Arena.ofShared()) {
            Qwen35Configuration config = Qwen35SyntheticModel.config();
            Qwen35SyntheticModel.Weights both = new Qwen35SyntheticModel(owned).weights(config);

            Qwen35 hostModel = new Qwen35(config, null, both.host(), null);
            Qwen35 deviceModel = new Qwen35(config, null, both.device(), null);
            Qwen35State hostState = new Qwen35State(config, -1);
            Qwen35State deviceState = new Qwen35State(config, -1);

            // The host reference: every position through the single-token forward pass, which is
            // what STANDARD does and what prefill has to be indistinguishable from.
            FloatTensor[] expected = new FloatTensor[PROMPT.length + DECODE_ROWS];
            int[] sequence = new int[PROMPT.length + DECODE_ROWS];
            System.arraycopy(PROMPT, 0, sequence, 0, PROMPT.length);
            for (int i = PROMPT.length; i < sequence.length; i++) {
                sequence[i] = 11 + i; // teacher forced: a fixed continuation, not the argmax
            }
            for (int position = 0; position < sequence.length; position++) {
                FloatTensor logits =
                        Qwen35Forward.forward(hostModel, hostState, sequence[position], position);
                expected[position] = copyOf(logits);
            }

            TornadoVMMasterPlanPrefillDecode plan =
                    new TornadoVMMasterPlanPrefillDecode(
                            deviceState, deviceModel, MetricsSink.disabled());
            try {
                // Ingest the prompt: no logits, no sampling, no readback.
                for (int position = 0; position < PROMPT.length; position++) {
                    TornadoPrefillPass.prefill(
                            deviceModel, deviceState, PROMPT[position], position, plan);
                }

                // Decode continues at the position after the last one ingested.
                for (int row = 0; row < DECODE_ROWS; row++) {
                    int position = PROMPT.length + row;
                    var actual =
                            TornadoForwardPass.forward(
                                    deviceModel, deviceState, sequence[position], position, plan);
                    assertRow(position, expected[position], actual);
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            restore(previousDevice);
        }
    }

    /**
     * Ingesting the prompt and then decoding from its last token again is the defect this pins.
     *
     * <p>It leaves that token in the key/value cache twice and advances the recurrence one step too
     * far, and the output stays fluent — the generic batched path carried exactly this off-by-one.
     * Here the wrong boundary is constructed deliberately, and it must disagree with the reference
     * that the correct one matches.
     */
    @Test
    public void aDuplicatedFinalPromptTokenIsDetectable() throws Exception {
        String previousDevice = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        // The floating-point path here too: this case compares the device against the host
        // exactly, and a quantized activation cannot be exact.
        System.setProperty("jitllm.qwen35.packedIntegerDot", "false");
        try (Arena owned = Arena.ofShared()) {
            Qwen35Configuration config = Qwen35SyntheticModel.config();
            Qwen35SyntheticModel.Weights both = new Qwen35SyntheticModel(owned).weights(config);

            Qwen35 hostModel = new Qwen35(config, null, both.host(), null);
            Qwen35 deviceModel = new Qwen35(config, null, both.device(), null);
            Qwen35State hostState = new Qwen35State(config, -1);
            Qwen35State deviceState = new Qwen35State(config, -1);

            int next = 33;
            for (int position = 0; position < PROMPT.length; position++) {
                Qwen35Forward.forward(hostModel, hostState, PROMPT[position], position);
            }
            FloatTensor expected =
                    copyOf(Qwen35Forward.forward(hostModel, hostState, next, PROMPT.length));

            TornadoVMMasterPlanPrefillDecode plan =
                    new TornadoVMMasterPlanPrefillDecode(
                            deviceState, deviceModel, MetricsSink.disabled());
            try {
                // The wrong shape: every prompt token ingested, then the last one fed again.
                for (int position = 0; position < PROMPT.length; position++) {
                    TornadoPrefillPass.prefill(
                            deviceModel, deviceState, PROMPT[position], position, plan);
                }
                var duplicated =
                        TornadoForwardPass.forward(
                                deviceModel,
                                deviceState,
                                PROMPT[PROMPT.length - 1],
                                PROMPT.length,
                                plan);

                boolean differs = false;
                for (int i = 0; i < Qwen35SyntheticModel.VOCAB && !differs; i++) {
                    differs = Math.abs(expected.getFloat(i) - duplicated.get(i)) > 1e-2f;
                }
                assertTrue(
                        "feeding the last prompt token again produced the same logits as feeding"
                                + " the next one, so this test could not see the off-by-one it"
                                + " exists to catch",
                        differs);
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            restore(previousDevice);
        }
    }

    private static FloatTensor copyOf(FloatTensor logits) {
        float[] values = new float[logits.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = logits.getFloat(i);
        }
        return new org.beehive.jitllm.tensor.standard.ArrayFloatTensor(values);
    }

    private static void assertRow(
            int position, FloatTensor expected, org.beehive.jitllm.inference.Logits actual) {
        float maxAbs = 0f;
        for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(expected.getFloat(i)));
        }
        for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
            float device = actual.get(i);
            assertTrue(
                    "logit " + i + " at position " + position + " is " + device,
                    Float.isFinite(device));
            assertEquals(
                    "logit " + i + " at position " + position,
                    expected.getFloat(i),
                    device,
                    Math.max(1e-3f, maxAbs * 3e-4f));
        }
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty("use.tornadovm");
        } else {
            System.setProperty("use.tornadovm", previous);
        }
    }
}

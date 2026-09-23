package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import org.beehive.jllm.backend.cpu.Qwen35Forward;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.model.qwen35.Qwen35;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

// @formatter:off
/**
 * A whole {@code qwen35} forward pass on the device against the same one on the host, on a model
 * small enough to build in a test.
 *
 * <p>The kernel tests settle each operation on its own and the topology test settles the plan's
 * shape. Neither can see the defects that only appear when the parts are wired together: a buffer
 * one task writes and the next does not read, a layer consuming the wrong predecessor's output, a
 * recurrent state that does not carry from one token to the next, an operand bound in the wrong
 * order. Those produce fluent, wrong text on the real model, and a multi-minute load before you
 * find out.
 *
 * <p>Both paths read the <b>same bytes</b>: each weight is generated once and wrapped twice, as the
 * host tensor and as the device tensor for that representation. So a disagreement here is the
 * engine's, not the fixture's.
 *
 * <p>The model is mixed the way the real one is — Q4_1 down projections on the early blocks, Q5_K
 * recurrent outputs, a Q6_K vocabulary projection, Q4_0 elsewhere, F32 norms and SSM parameters —
 * and it has both layer kinds, so both mixers run.
 */
// @formatter:on
public class Qwen35SyntheticParityAccelTest {

    private static void assertFinite(String what, FloatTensor values) {
        for (int i = 0; i < values.size(); i++) {
            float v = values.getFloat(i);
            assertTrue(what + "[" + i + "] is " + v, Float.isFinite(v));
        }
    }

    // @formatter:off
    /**
     * Two positions of the same sequence, host against device.
     *
     * <p>Two rather than one because the first says nothing about the recurrence: at position 0 the
     * convolution window and the delta-net state are zero, so a layer that failed to carry them
     * forward would still agree. The second position is the one that reads what the first wrote.
     */
    // @formatter:on
    @Test
    public void theWholeForwardPassAgreesWithTheHost() throws Exception {
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        // These cases compare the device against the host exactly: their subject is addressing
        // and chunk invariance, not arithmetic. A quantized activation cannot be exact, so the
        // Q4_0 projections stay on the floating-point path here. Their precision is covered on the
        // real model by the parity tests, against bounds written for it. This class gets its own
        // JVM (reuseForks=false), so the property is read before the layer builder loads.
        System.setProperty("jllm.qwen35.packedIntegerDot", "false");
        try (Arena owned = Arena.ofShared()) {
            Qwen35Configuration config = Qwen35SyntheticModel.config();
            Qwen35SyntheticModel.Weights both = new Qwen35SyntheticModel(owned).weights(config);

            Qwen35 hostModel = new Qwen35(config, null, both.host(), null);
            Qwen35 deviceModel = new Qwen35(config, null, both.device(), null);
            Qwen35State hostState = new Qwen35State(config, -1);
            Qwen35State deviceState = new Qwen35State(config, -1);

            TornadoVMMasterPlanSingleToken plan =
                    new TornadoVMMasterPlanSingleToken(
                            deviceState, deviceModel, MetricsSink.disabled());
            try {
                int[] tokens = {7, 91};
                for (int position = 0; position < tokens.length; position++) {
                    FloatTensor expected =
                            Qwen35Forward.forward(hostModel, hostState, tokens[position], position);
                    assertFinite("host logits at position " + position, expected);

                    var actual =
                            TornadoForwardPass.forward(
                                    deviceModel, deviceState, tokens[position], position, plan);

                    float maxAbs = 0f;
                    for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
                        maxAbs = Math.max(maxAbs, Math.abs(expected.getFloat(i)));
                    }
                    for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
                        float device = actual.get(i);
                        assertTrue(
                                "device logit " + i + " at position " + position + " is " + device,
                                Float.isFinite(device));
                        // Relative to the row's own scale: the device reduces in a different order
                        // and through a different attention decomposition, so this is not bit
                        // equality — but it is far tighter than a defect would survive.
                        assertEquals(
                                "logit " + i + " at position " + position,
                                expected.getFloat(i),
                                device,
                                Math.max(1e-3f, maxAbs * 3e-4f));
                    }
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
    }
}

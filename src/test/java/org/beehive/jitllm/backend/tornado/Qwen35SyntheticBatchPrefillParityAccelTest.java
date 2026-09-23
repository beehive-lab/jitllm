package org.beehive.jllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.beehive.jllm.backend.cpu.Qwen35Forward;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.qwen35.Qwen35;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

// @formatter:off
/**
 * Batched prompt ingestion on the device against the host taking the same tokens one at a time.
 *
 * <p>This is the gate the batched path exists to pass. Three quarters of this stack is recurrent,
 * so a chunk is not a set of independent rows: the convolution window and the delta-net matrices
 * make token {@code t} depend on token {@code t-1} <i>inside</i> the layer. A batched
 * implementation that reordered them, or that reset the state per chunk, would still produce fluent
 * output and would disagree here.
 *
 * <p>Three properties are checked, and each fails differently:
 *
 * <ul>
 *   <li><b>Against the host.</b> The decode rows after ingestion must match the host running the
 *       whole sequence one token at a time — which is what {@code STANDARD} does.
 *   <li><b>Across chunk sizes.</b> The same prompt ingested at width 1, 2, 7, 8 and wider than the
 *       prompt itself must leave the same state, so the decode rows must agree with each other as
 *       well as with the host.
 *   <li><b>At chunk boundaries.</b> Prompt lengths exactly on, one below and one above a multiple
 *       of the width, so the partially active final chunk is exercised rather than assumed.
 * </ul>
 */
// @formatter:on
public class Qwen35SyntheticBatchPrefillParityAccelTest {

    static {
        // The scalar batched path, whose kernels reproduce the host's arithmetic closely enough
        // for the host comparison below. The tensor-core default rounds Q and P to FP16 in the
        // attention and is covered by its own numerics tests at its own bounds.
        System.setProperty("jllm.qwen35.tensorCores", "false");
    }

    private static final int DECODE_ROWS = 3;

    /** Widths worth separating: one, a pair, a non-power-of-two, a power of two, and too wide. */
    private static final int[] BATCH_WIDTHS = {1, 2, 7, 8, 16};

    /**
     * Only recurrent layers, and only attention layers.
     *
     * <p>The mixed stack cannot say which mixer moved. An interval past the layer count makes every
     * block recurrent and an interval of one makes every block attend, so a disagreement here names
     * the branch instead of leaving it to be bisected.
     */
    @Test
    public void eachMixerBatchesOnItsOwn() throws Exception {
        int[] prompt = {7, 91, 5, 42};
        assertMatchesHost(prompt, 1000, new int[] {2});
        assertMatchesHost(prompt, 1, new int[] {2});
    }

    @Test
    public void batchedIngestionMatchesTheHostAtEveryWidth() throws Exception {
        // Eight prompt tokens: exactly a width-8 chunk, one above width 7, one below width 9.
        int[] prompt = {7, 91, 5, 42, 63, 12, 200, 31};
        assertMatchesHost(prompt);
    }

    @Test
    public void aPromptOneBelowTheBoundaryLeavesAPartialChunk() throws Exception {
        int[] prompt = {7, 91, 5, 42, 63, 12, 200};
        assertMatchesHost(prompt, Qwen35SyntheticModel.INTERVAL, new int[] {8});
    }

    @Test
    public void aPromptOneAboveTheBoundarySpansTwoChunks() throws Exception {
        int[] prompt = {7, 91, 5, 42, 63, 12, 200, 31, 4};
        assertMatchesHost(prompt, Qwen35SyntheticModel.INTERVAL, new int[] {8});
    }

    // ── the comparison ────────────────────────────────────────────────────────

    private void assertMatchesHost(int[] prompt) throws Exception {
        assertMatchesHost(prompt, Qwen35SyntheticModel.INTERVAL, BATCH_WIDTHS);
    }

    // Every plan built in a JVM keeps its device buffers: TornadoVM returns freed memory to its
    // own provider, not to the driver. So a case builds the widths it actually needs rather than
    // the whole sweep, and the sweep lives in one case.
    private void assertMatchesHost(int[] prompt, int attentionInterval, int[] widths)
            throws Exception {
        String previousDevice = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        // These cases compare the device against the host exactly: their subject is addressing
        // and chunk invariance, not arithmetic. A quantized activation cannot be exact, so the
        // Q4_0 projections stay on the floating-point path here. Their precision is covered on the
        // real model by the parity tests, against bounds written for it. This class gets its own
        // JVM (reuseForks=false), so the property is read before the layer builder loads.
        System.setProperty("jllm.qwen35.packedIntegerDot", "false");
        try (Arena owned = Arena.ofShared()) {
            Qwen35Configuration config = Qwen35SyntheticModel.config(attentionInterval);
            Qwen35SyntheticModel.Weights both = new Qwen35SyntheticModel(owned).weights(config);

            int[] sequence = new int[prompt.length + DECODE_ROWS];
            System.arraycopy(prompt, 0, sequence, 0, prompt.length);
            for (int i = prompt.length; i < sequence.length; i++) {
                sequence[i] = 11 + i; // teacher forced
            }

            // The reference: one token at a time on the host.
            Qwen35 hostModel = new Qwen35(config, null, both.host(), null);
            Qwen35State hostState = new Qwen35State(config, -1);
            List<FloatTensor> expected = new ArrayList<>();
            for (int position = 0; position < sequence.length; position++) {
                FloatTensor logits =
                        Qwen35Forward.forward(hostModel, hostState, sequence[position], position);
                if (position >= prompt.length) {
                    expected.add(copyOf(logits));
                }
            }

            List<float[][]> perWidth = new ArrayList<>();
            for (int width : widths) {
                float[][] rows = ingestThenDecode(config, both, prompt, sequence, width);
                perWidth.add(rows);
                for (int row = 0; row < DECODE_ROWS; row++) {
                    assertRow(width, row, expected.get(row), rows[row]);
                }
            }

            // And the batched widths against each other. A chunk is a scheduling unit: the same
            // kernels run in the same per-lane order whatever the width, so this is far tighter
            // than the comparison against the host — which crosses two different plans, and two
            // different attention decompositions with them.
            int firstBatched = widths[0] > 1 ? 0 : 1; // width 1 runs the sequential plan
            if (perWidth.size() <= firstBatched + 1) {
                return; // one batched width in this case; the host comparison above is the check
            }
            float[][] reference = perWidth.get(firstBatched);
            for (int w = firstBatched + 1; w < perWidth.size(); w++) {
                float[][] other = perWidth.get(w);
                for (int row = 0; row < DECODE_ROWS; row++) {
                    for (int i = 0; i < reference[row].length; i++) {
                        assertEquals(
                                "width "
                                        + widths[w]
                                        + " disagrees with width "
                                        + widths[firstBatched]
                                        + " at decode row "
                                        + row
                                        + ", logit "
                                        + i,
                                reference[row][i],
                                other[row][i],
                                Math.max(1e-4f, Math.abs(reference[row][i]) * 1e-5f));
                    }
                }
            }
        } finally {
            restore(previousDevice);
        }
    }

    /** Ingests the prompt in chunks of {@code width}, then decodes, returning the decode rows. */
    private float[][] ingestThenDecode(
            Qwen35Configuration config,
            Qwen35SyntheticModel.Weights both,
            int[] prompt,
            int[] sequence,
            int width)
            throws Exception {
        Qwen35 deviceModel = new Qwen35(config, null, both.device(), null);
        // Both the workspace width and the policy the plan reads its worker grids from. They are
        // one number in production, resolved together by the session; a harness that set only the
        // first got a plan sized for one row and rows that were never computed.
        Qwen35State deviceState = stateForWidth(config, width);

        float[][] rows = new float[DECODE_ROWS][];
        if (width <= 1) {
            TornadoVMMasterPlanPrefillDecode plan =
                    new TornadoVMMasterPlanPrefillDecode(
                            deviceState, deviceModel, MetricsSink.disabled());
            try {
                for (int position = 0; position < prompt.length; position++) {
                    TornadoPrefillPass.prefill(
                            deviceModel, deviceState, prompt[position], position, plan);
                }
                for (int row = 0; row < DECODE_ROWS; row++) {
                    int position = prompt.length + row;
                    rows[row] =
                            copyOf(
                                    TornadoForwardPass.forward(
                                            deviceModel,
                                            deviceState,
                                            sequence[position],
                                            position,
                                            plan));
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
            return rows;
        }

        TornadoVMMasterPlanBatchPrefillDecode plan =
                // The width travels in the state's policy, which is where the plan reads it and
                // where the batch workspace was sized from.
                new TornadoVMMasterPlanBatchPrefillDecode(
                        deviceState, deviceModel, MetricsSink.disabled());
        try {
            for (int start = 0; start < prompt.length; start += width) {
                int end = Math.min(start + width, prompt.length);
                int[] chunk = Arrays.copyOfRange(prompt, start, end);
                TornadoBatchPrefillPass.batchPrefill(
                        deviceModel, deviceState, chunk, start, chunk.length, plan);
            }
            for (int row = 0; row < DECODE_ROWS; row++) {
                int position = prompt.length + row;
                rows[row] =
                        copyOf(
                                TornadoBatchPrefillPass.decode(
                                        deviceModel,
                                        deviceState,
                                        sequence[position],
                                        position,
                                        plan));
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
        return rows;
    }

    private static Qwen35State stateForWidth(Qwen35Configuration config, int width) {
        String previousBatch = System.getProperty("jllm.prefillBatchSize");
        String previousPhase = System.getProperty("jllm.withPrefillDecode");
        // A batch width is only a width when the phase strategy asks for a separate prefill:
        // ExecutionPolicy pins it to one otherwise, and the plan reads its worker grids from it.
        System.setProperty("jllm.prefillBatchSize", String.valueOf(Math.max(width, 1)));
        System.setProperty("jllm.withPrefillDecode", "true");
        try {
            return State.withPrefillBatchSize(
                    Math.max(width, 1), () -> new Qwen35State(config, -1));
        } finally {
            restore("jllm.prefillBatchSize", previousBatch);
            restore("jllm.withPrefillDecode", previousPhase);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static float[] copyOf(org.beehive.jllm.inference.Logits logits) {
        float[] values = new float[Qwen35SyntheticModel.VOCAB];
        for (int i = 0; i < values.length; i++) {
            values[i] = logits.get(i);
        }
        return values;
    }

    private static FloatTensor copyOf(FloatTensor logits) {
        float[] values = new float[logits.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = logits.getFloat(i);
        }
        return new ArrayFloatTensor(values);
    }

    private static void assertRow(int width, int row, FloatTensor expected, float[] actual) {
        float maxAbs = 0f;
        for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(expected.getFloat(i)));
        }
        for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
            assertTrue(
                    "width " + width + " decode row " + row + " logit " + i + " is " + actual[i],
                    Float.isFinite(actual[i]));
            assertEquals(
                    "width " + width + ", decode row " + row + ", logit " + i,
                    expected.getFloat(i),
                    actual[i],
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

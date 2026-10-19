package org.beehive.jitllm.golden;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * The CPU reference against the accelerator with the prompt ingested through the batched prefill on
 * Apple MLX ({@code --with-native-libraries} on Metal), then decoded on the JIT kernels.
 *
 * <p>Two things are under test at once, and the teacher-forced rows separate them. The prompt
 * position's row is the MLX prefill's arithmetic: its GEMMs and fused attention. Every row after it
 * is the decode layers, which bind from the prefill graphs every weight those graphs still hold and
 * upload the ones MLX replaced — a decode graph that bound a weight the prefill graph never
 * uploaded reads zeros and fails here by orders of magnitude.
 *
 * <p>The width is {@value #WIDTH}, below the prompt's length, so the prompt is several chunks and
 * the last is partial: every chunk after the first attends over keys and values an earlier chunk
 * wrote, through the contiguous gather.
 *
 * <p>Except for the affine repack, which runs at {@value #AFFINE_WIDTH}. Below some hundreds of
 * rows MLX's quantized GEMM splits its depth and keeps the partial sums in FP16 ({@code
 * affine_qmm_t_splitk}), which adds rounding the FP16 envelope does not allow for: at 16 rows
 * Llama-3.2-1B Q8_0 measures a relative L2 of 3.2e-3 against 2e-3, with no argmax change. MLX's own
 * {@code quantized_matmul} behaves the same, and at {@value #AFFINE_WIDTH} rows it does not split,
 * where the affine GEMM's error equals the FP16 one's. The narrow-chunk precision is tracked
 * separately rather than allowed for here.
 *
 * <p>Each subclass compares against references captured with an FP32 key/value cache, as the JIT
 * batched-prefill parity does; the MLX attention then reads keys and values its gather rounded to
 * FP16.
 *
 * <p>Each capture asserts it ran MLX — tasks that exist only on that path ({@link #mlxTasks}) — so
 * a host where the path quietly did not engage fails rather than comparing the JIT kernels with
 * themselves.
 */
// @formatter:on
abstract class MlxPrefillParity extends CpuGpuParity {

    /** Above MLX's fused-attention minimum of 9 query rows, below the prompt's length. */
    static final int WIDTH = 16;

    /** A width at which MLX's quantized GEMM does not split its depth for these models. */
    static final int AFFINE_WIDTH = 512;

    /** Pins the representation of Q8_0 and Q4_0 projections: FP16 copies, or the affine repack. */
    enum Weights {
        AS_LOADED(null),
        FP16_COPIES("resident"),
        AFFINE("per-chunk");

        final String property;

        Weights(String property) {
            this.property = property;
        }
    }

    void assertMlxParity(Fixture fixture, Bounds bounds, Weights weights) throws Exception {
        assertMlxParity(width -> assertParityBatched(fixture, bounds, width), weights);
    }

    /** {@link #assertMlxParity(Fixture, Bounds, Weights)} for a model file found by name. */
    void assertMlxParity(java.nio.file.Path model, String label, Bounds bounds, Weights weights)
            throws Exception {
        assertMlxParity(width -> assertParityBatched(model, label, bounds, width), weights);
    }

    /** A batched parity run at a chunk width. */
    interface BatchedParity {
        GoldenCapture.Result run(int width) throws Exception;
    }

    private void assertMlxParity(BatchedParity parity, Weights weights) throws Exception {
        assumeTrue(
                "the MLX prefill runs on Metal only",
                BackendId.METAL.equals(TornadoDevices.current().backend()));
        assumeTrue("MLX's mlx.metallib is not loadable here", MlxPrefill.available());
        String previousNative = System.getProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY);
        String previousDecoded = System.getProperty(ExecutionPolicy.DECODED_PROJECTIONS_PROPERTY);
        System.setProperty(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, "true");
        if (weights.property != null) {
            System.setProperty(ExecutionPolicy.DECODED_PROJECTIONS_PROPERTY, weights.property);
        }
        try {
            GoldenCapture.Result gpu = parity.run(weights == Weights.AFFINE ? AFFINE_WIDTH : WIDTH);
            assertTrue(
                    "the plan did not record its dispatch, so it cannot show it ran MLX",
                    gpu.gridScheduler != null);
            for (String marker : mlxTasks()) {
                assertTrue(
                        "no "
                                + marker
                                + " task in the batched prefill: the MLX path did not engage",
                        gpu.gridScheduler.keySet().stream().anyMatch(k -> k.endsWith(marker)));
            }
        } finally {
            restore(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, previousNative);
            restore(ExecutionPolicy.DECODED_PROJECTIONS_PROPERTY, previousDecoded);
        }
    }

    /**
     * Tasks that exist only on the MLX path, by the suffix of their grid's name: the QKV split and
     * the contiguous key/value gather of the MLX attention.
     */
    String[] mlxTasks() {
        return new String[] {".batch_qkv_split", ".batch_attention_gather_kv"};
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}

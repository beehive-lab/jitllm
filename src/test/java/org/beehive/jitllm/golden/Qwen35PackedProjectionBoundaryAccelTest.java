package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The packed-integer projection at the boundary it is actually used at, on the real model.
 *
 * <p><b>Scope, stated precisely, because it is narrow.</b> This compares the device's quantization
 * and one packed projection against a host reference performing the <i>same</i> block quantization,
 * at the <b>first recurrent layer's</b> {@code ssm_qkv} and {@code ssm_gate} boundaries, over four
 * token positions. The activation is the model's own: the embedding of a real token put through
 * layer 0's own attention-norm weights, which is exactly what a sequence's first position presents
 * at that boundary. It is <b>not</b> a running residual from a longer context, and it says nothing
 * about layers past the first.
 *
 * <p>What it checks, in order of what can go wrong: the block scales, the sums of quants and the
 * packed signed bytes the device produced; then the projection outputs themselves. Comparing the
 * final logits cannot separate a wrong scale from a wrong buffer from accumulated rounding, which
 * is why none of those are what this asserts.
 *
 * <p>The gap against the unquantized kernel on the same input is reported rather than bounded: it
 * is the cost of the representation, and it belongs in the quality screen, not here.
 */
// @formatter:on
public class Qwen35PackedProjectionBoundaryAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    private static final int QK = 32;

    /** Token ids to take embeddings from. Arbitrary, fixed, and spread across the vocabulary. */
    private static final int[] TOKENS = {15, 4242, 60123, 128000};

    @Test
    public void theDeviceQuantizesAndProjectsWhatTheReferenceDoes() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println("[SKIP] environment absent");
            assumeTrue("environment absent", false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] no TornadoVM device");
            assumeTrue("environment absent", false);
        }

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Model model = ModelLoader.loadModel(modelPath, 512, true, true);
            Qwen35Configuration config = (Qwen35Configuration) model.configuration();
            Qwen35TornadoWeights weights = (Qwen35TornadoWeights) model.weights();
            int dim = config.dim();

            // Layer 0 is recurrent in this family, and its two Q4_0 projections off the normed
            // activation are the ones the dispatch packs.
            assertTrue("layer 0 is recurrent", config.isRecurrentLayer(0));
            TornadoTensor qkv = weights.ssmQkv[0];
            TornadoTensor gate = weights.ssmGate[0];
            float[] normWeight = toHost(weights.rms_att_weightLayered[0], dim);

            for (int token : TOKENS) {
                float[] activation = normedEmbedding(model, config, normWeight, token);

                Quantized reference = new Quantized(activation);
                Device device = runOnDevice(activation, qkv, config.deltaNetConvDim(), dim);

                // 1. The quantization itself, element for element.
                for (int block = 0; block < dim / QK; block++) {
                    assertEquals(
                            "token " + token + " block " + block + " scale",
                            reference.scales[block],
                            device.scales[block],
                            0.0f);
                    assertEquals(
                            "token " + token + " block " + block + " sum of quants",
                            reference.sums[block],
                            device.sums[block]);
                }
                for (int i = 0; i < dim / 4; i++) {
                    assertEquals(
                            "token " + token + " packed quants at " + i,
                            reference.quants[i],
                            device.quants[i]);
                }

                // 2. The projection output, against the same-quantization reference.
                assertProjection(
                        "ssm_qkv, token " + token,
                        qkv,
                        reference,
                        device.packed,
                        config.deltaNetConvDim(),
                        dim);

                // 3. And what the representation costs at this boundary, reported not bounded.
                double worst = 0;
                double largest = 0;
                for (int row = 0; row < config.deltaNetConvDim(); row++) {
                    largest = Math.max(largest, Math.abs(device.unquantized[row]));
                    worst = Math.max(worst, Math.abs(device.unquantized[row] - device.packed[row]));
                }
                System.out.printf(
                        "[BOUNDARY] token %d ssm_qkv: packed vs unquantized worst %.6g,"
                                + " largest |out| %.6g (%.4g%%)%n",
                        token, worst, largest, 100.0 * worst / largest);
            }

            // The second packed projection off the same activation, on one token.
            float[] activation = normedEmbedding(model, config, normWeight, TOKENS[0]);
            Quantized reference = new Quantized(activation);
            Device device = runOnDevice(activation, gate, config.deltaNetValueDim(), dim);
            assertProjection(
                    "ssm_gate, token " + TOKENS[0],
                    gate,
                    reference,
                    device.packed,
                    config.deltaNetValueDim(),
                    dim);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    /** The reference projection: the same quants, the same weights, summed in double. */
    private static void assertProjection(
            String what, TornadoTensor w, Quantized reference, float[] got, int rows, int n) {
        ByteArray raw = w.asByteArray();
        double largest = 0;
        double worst = 0;
        int worstRow = -1;
        for (int row = 0; row < rows; row++) {
            double expected = 0;
            for (int block = 0; block < n / QK; block++) {
                int base = (row * (n / QK) + block) * 18;
                float weightScale = raw.getHalfFloat(base).getFloat32();
                int dot = 0;
                for (int i = 0; i < QK; i++) {
                    int packedByte = raw.get(base + 2 + (i & 15)) & 0xFF;
                    int nibble = i < 16 ? (packedByte & 0xF) : ((packedByte >> 4) & 0xF);
                    int packedQuant = reference.quants[block * (QK / 4) + i / 4];
                    int quant = (byte) ((packedQuant >> ((i % 4) * 8)) & 0xFF);
                    dot += (nibble - 8) * quant;
                }
                expected += (double) weightScale * reference.scales[block] * dot;
            }
            assertTrue(what + " row " + row + " is " + got[row], Float.isFinite(got[row]));
            largest = Math.max(largest, Math.abs(expected));
            if (Math.abs(expected - got[row]) > worst) {
                worst = Math.abs(expected - got[row]);
                worstRow = row;
            }
        }
        System.out.printf(
                "[BOUNDARY] %s: against the same-quantization reference worst %.6g at row %d,"
                        + " largest |out| %.6g%n",
                what, worst, worstRow, largest);
        assertTrue(
                what + ": worst " + worst + " at row " + worstRow + ", largest " + largest,
                worst <= 1e-4 * largest);
    }

    /** Embedding of {@code token}, through layer 0's attention norm. */
    private static float[] normedEmbedding(
            Model model, Qwen35Configuration config, float[] normWeight, int token) {
        int dim = config.dim();
        float[] x = new float[dim];
        var table = ((Qwen35TornadoWeights) model.weights()).getTokenEmbeddingTable();
        ByteArray raw = table.asByteArray();
        int blocksPerRow = dim / QK;
        for (int i = 0; i < dim; i++) {
            int base = (token * blocksPerRow + i / QK) * 18;
            float scale = raw.getHalfFloat(base).getFloat32();
            int within = i % QK;
            int packed = raw.get(base + 2 + (within & 15)) & 0xFF;
            int nibble = within < 16 ? (packed & 0xF) : ((packed >> 4) & 0xF);
            x[i] = scale * (nibble - 8);
        }
        double sumSquares = 0;
        for (float v : x) {
            sumSquares += (double) v * v;
        }
        float inverse = (float) (1.0 / Math.sqrt(sumSquares / dim + config.rmsNormEps()));
        float[] normed = new float[dim];
        for (int i = 0; i < dim; i++) {
            normed[i] = normWeight[i] * x[i] * inverse;
        }
        return normed;
    }

    private static float[] toHost(TornadoTensor tensor, int n) {
        FloatArray array = tensor.asFloatArray();
        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            host[i] = array.get(i);
        }
        return host;
    }

    /** What the device produced: the quantization, the packed projection, and the FP32 one. */
    private record Device(
            int[] quants, float[] scales, int[] sums, float[] packed, float[] unquantized) {}

    private static Device runOnDevice(float[] activation, TornadoTensor w, int rows, int n)
            throws Exception {
        FloatArray x = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            x.set(i, activation[i]);
        }
        IntArray quants = new IntArray(n / 4);
        FloatArray scales = new FloatArray(n / QK);
        IntArray sums = new IntArray(n / QK);
        quants.init(0);
        scales.init(0.0f);
        sums.init(0);
        FloatArray packed = new FloatArray(rows);
        FloatArray unquantized = new FloatArray(rows);
        packed.init(0.0f);
        unquantized.init(0.0f);
        ByteArray raw = w.asByteArray();

        int local = 128;
        TaskGraph graph =
                new TaskGraph("boundary")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                x,
                                raw,
                                quants,
                                scales,
                                sums,
                                packed,
                                unquantized)
                        .task(
                                "quantize",
                                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                new KernelContext(),
                                x,
                                quants,
                                scales,
                                sums)
                        .task(
                                "packed",
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                                new KernelContext(),
                                quants,
                                scales,
                                sums,
                                packed,
                                raw,
                                n,
                                rows,
                                local)
                        .task(
                                "unquantized",
                                TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                                new KernelContext(),
                                x,
                                unquantized,
                                raw,
                                n,
                                rows,
                                local)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION,
                                quants,
                                scales,
                                sums,
                                packed,
                                unquantized);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D blocks = new WorkerGrid1D(n);
        blocks.setLocalWork(QK, 1, 1);
        scheduler.addWorkerGrid("boundary.quantize", blocks);
        WorkerGrid1D rowGrid = new WorkerGrid1D(rows * local);
        rowGrid.setLocalWork(local, 1, 1);
        scheduler.addWorkerGrid("boundary.packed", rowGrid);
        scheduler.addWorkerGrid("boundary.unquantized", rowGrid);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int[] hostQuants = new int[n / 4];
        for (int i = 0; i < hostQuants.length; i++) {
            hostQuants[i] = quants.get(i);
        }
        float[] hostScales = new float[n / QK];
        int[] hostSums = new int[n / QK];
        for (int i = 0; i < hostScales.length; i++) {
            hostScales[i] = scales.get(i);
            hostSums[i] = sums.get(i);
        }
        float[] hostPacked = new float[rows];
        float[] hostUnquantized = new float[rows];
        for (int i = 0; i < rows; i++) {
            hostPacked[i] = packed.get(i);
            hostUnquantized[i] = unquantized.get(i);
        }
        return new Device(hostQuants, hostScales, hostSums, hostPacked, hostUnquantized);
    }

    /** The host's copy of the kernel's activation quantization. */
    private static final class Quantized {
        final int[] quants;
        final float[] scales;
        final int[] sums;

        Quantized(float[] x) {
            int n = x.length;
            quants = new int[n / 4];
            scales = new float[n / QK];
            sums = new int[n / QK];
            for (int block = 0; block < n / QK; block++) {
                int base = block * QK;
                float maxAbs = 0;
                for (int i = 0; i < QK; i++) {
                    maxAbs = Math.max(maxAbs, Math.abs(x[base + i]));
                }
                scales[block] = maxAbs / 127.0f;
                float inverse = maxAbs > 0 ? 127.0f / maxAbs : 0.0f;
                int sum = 0;
                for (int g = 0; g < QK / 4; g++) {
                    int packed = 0;
                    for (int lane = 0; lane < 4; lane++) {
                        float v = x[base + g * 4 + lane] * inverse;
                        int q = (int) (v + (v >= 0 ? 0.5f : -0.5f));
                        q = Math.min(127, Math.max(-127, q));
                        sum += q;
                        packed |= (q & 0xFF) << (lane * 8);
                    }
                    quants[block * (QK / 4) + g] = packed;
                }
                sums[block] = sum;
            }
        }
    }
}

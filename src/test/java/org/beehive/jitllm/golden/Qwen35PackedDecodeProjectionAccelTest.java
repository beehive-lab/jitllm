package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ8_0DP4A;
import org.beehive.jitllm.backend.tornado.layers.PackedActivationTasks;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The packed decode projections of Qwen3.5-0.8B Q4_K_M, on the file's own weights and on
 * activations the host model produces, against a host reference that quantizes the same input the
 * same way.
 *
 * <p>Decoding on a device with packed integer dot products, every recurrent layer quantizes two
 * activations to eight bits in blocks of 32 and reads them in packed integers: the attention norm's
 * output, quantized by the norm's apply, for the Q8_0 {@code ssm_alpha} and {@code ssm_beta}; and
 * the delta-net readout, quantized by {@code ssm_out_quantize}, for the Q5_K {@code ssm_out}, which
 * adds into the residual stream. This runs those kernels on the grids the layer gives them, for
 * every recurrent layer at every prompt position, with the residual stream and the readout recorded
 * from the host model's own forward pass.
 *
 * <p>What it holds them to:
 *
 * <ul>
 *   <li>the normalized activation bit for bit against {@code weight * (scale * x)} on the host, and
 *       the block scales, quants and sums exactly against the host quantizing the same values;
 *   <li>each projection output within a summation bound of the double-precision sum over the same
 *       quants and the file's weights: {@link #BOUND} times the sum of the absolute values of the
 *       terms the kernel adds, the residual included. A row whose terms add to 4.0 in absolute
 *       value may differ by 7.6e-6;
 *   <li>finite outputs. The stored outputs start as NaN, so a row the kernel skips fails too.
 * </ul>
 *
 * <p>The data has to tell defects apart at that bound: in more than half the rows, a reference that
 * reads the activation one element late, one that takes the next block's scale, and one that skips
 * the quantization must each fall outside it.
 *
 * <p>Uses the file under the fixture directory, by name; skips when it is absent or the device does
 * not lower {@code dp4a}, where decoding does not take this path.
 */
// @formatter:on
public class Qwen35PackedDecodeProjectionAccelTest {

    private static final int QK = 32;
    private static final int QK_K = 256;
    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int Q5_K_BLOCK_BYTES = 176;

    /** Lanes per output row, as the layer launches a packed projection on CUDA. */
    private static final int LOCAL = 128;

    /** The allowed difference per unit of the terms' absolute sum: 32 FP32 unit roundoffs. */
    static final double BOUND = Math.scalb(1.0, -19);

    /** The three wrong references the data must tell apart from the right one. */
    private static final String[] WRONG = {
        "activation one element late", "next block's scale", "unquantized activation"
    };

    @Test
    public void packedProjectionsMatchTheSameQuantization() throws Exception {
        Path file = GoldenFixture.modelsRoot().resolve("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf");
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        assumeTrue("environment absent: no accelerator", TupleInfo.acceleratorPresent());
        assumeTrue(
                "the device does not lower dp4a, so decoding does not take the packed path",
                PackedActivationTasks.supported());

        Model model = ModelLoader.loadModel(file, GoldenCapture.CONTEXT_LENGTH, true, false);
        Qwen35Configuration config = (Qwen35Configuration) model.configuration();
        Qwen35StandardWeights weights = (Qwen35StandardWeights) model.weights();
        int dim = config.dim();
        int heads = config.numberOfValueHeads();
        int valueDim = config.deltaNetValueDim();

        List<Integer> recurrent = new ArrayList<>();
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isRecurrentLayer(layer)) {
                assertEquals(DataType.Q8_0, weights.ssmAlpha[layer].dataType());
                assertEquals(DataType.Q8_0, weights.ssmBeta[layer].dataType());
                assertEquals(DataType.Q5_K, weights.ssmOut[layer].dataType());
                recurrent.add(layer);
            }
        }
        assertEquals("recurrent layers", 18, recurrent.size());

        // The original tensors, before the recorders take their place.
        FloatTensor[] alpha = weights.ssmAlpha.clone();
        FloatTensor[] beta = weights.ssmBeta.clone();
        FloatTensor[] out = weights.ssmOut.clone();
        List<List<Sample>> samples = record(model, weights, recurrent);

        Device device = new Device(dim, heads, valueDim);
        Totals q8 = new Totals("ssm_alpha and ssm_beta (Q8_0)");
        Totals q5 = new Totals("ssm_out (Q5_K)");
        try (TornadoExecutionPlan plan = device.plan()) {
            for (int i = 0; i < recurrent.size(); i++) {
                int layer = recurrent.get(i);
                Q8_0Rows alphaRows = new Q8_0Rows(alpha[layer], heads, dim);
                Q8_0Rows betaRows = new Q8_0Rows(beta[layer], heads, dim);
                Q5_KRows outRows = new Q5_KRows(out[layer], dim, valueDim);
                float[] norm = new float[dim];
                for (int k = 0; k < dim; k++) {
                    norm[k] = weights.attnNorm[layer].getFloat(k);
                }
                device.load(alphaRows.raw, betaRows.raw, outRows.raw, norm);
                for (int position = 0; position < samples.get(i).size(); position++) {
                    Sample sample = samples.get(i).get(position);
                    String where = "layer " + layer + " position " + position;
                    float ss = rmsScale(sample.x, config.rmsNormEps());
                    device.run(plan, sample, ss);

                    float[] normed = new float[dim];
                    for (int k = 0; k < dim; k++) {
                        normed[k] = norm[k] * (ss * sample.x[k]);
                    }
                    assertNormed(where, normed, sample.xb, device.xb);
                    Quantized xq = new Quantized(normed);
                    xq.assertDevice(where + " norm", device.quantsA, device.scalesA, device.sumsA);
                    alphaRows.check(where + " ssm_alpha", xq, normed, device.alphaOut, q8);
                    betaRows.check(where + " ssm_beta", xq, normed, device.betaOut, q8);

                    Quantized rq = new Quantized(sample.readout);
                    rq.assertDevice(
                            where + " readout", device.quantsO, device.scalesO, device.sumsO);
                    outRows.check(
                            where + " ssm_out", rq, sample.readout, sample.residual, device.hb, q5);
                }
            }
        }
        q8.report();
        q5.report();
    }

    /** NaN and both infinities are outside any bound; a finite value inside it is not. */
    @Test
    public void nonFiniteOutputsAreOutside() {
        assertTrue(outside(Float.NaN, 0.0, 1.0));
        assertTrue(outside(Float.POSITIVE_INFINITY, 0.0, 1.0));
        assertTrue(outside(Float.NEGATIVE_INFINITY, 0.0, 1.0));
        assertFalse(outside(0.5f, 0.0, 1.0));
        assertTrue(outside(1.5f, 0.0, 1.0));
    }

    /** Whether {@code got} is non-finite or further than {@code bound} from {@code expected}. */
    static boolean outside(float got, double expected, double bound) {
        return !Float.isFinite(got) || !(Math.abs(got - expected) <= bound);
    }

    // ---------------------------------------------------------------- the host model's activations

    /** One recurrent layer at one position: what its packed projections read. */
    private record Sample(float[] x, float[] xb, float[] readout, float[] residual) {}

    /**
     * The residual stream and the normalized input at each {@code ssm_alpha} call, and the readout
     * and the residual stream at each {@code ssm_out} call, while the host ingests the prompt.
     */
    private static List<List<Sample>> record(
            Model model, Qwen35StandardWeights weights, List<Integer> recurrent) throws Exception {
        State state = model.createNewState();
        List<Recorder> norms = new ArrayList<>();
        List<Recorder> readouts = new ArrayList<>();
        for (int layer : recurrent) {
            Recorder norm = new Recorder(weights.ssmAlpha[layer], state.x);
            Recorder readout = new Recorder(weights.ssmOut[layer], state.x);
            weights.ssmAlpha[layer] = norm;
            weights.ssmOut[layer] = readout;
            norms.add(norm);
            readouts.add(readout);
        }
        List<Integer> prompt = GoldenCapture.promptTokens(model);
        int n = prompt.size();
        Sampler none =
                tensor -> {
                    throw new IllegalStateException("only the prompt is ingested");
                };
        model.generateTokens(
                state, 0, prompt.subList(0, n - 1), Set.of(), n - 1, none, false, null);

        List<List<Sample>> samples = new ArrayList<>();
        for (int i = 0; i < recurrent.size(); i++) {
            List<float[][]> a = norms.get(i).calls;
            List<float[][]> o = readouts.get(i).calls;
            assertEquals("ssm_alpha and ssm_out calls", a.size(), o.size());
            assertTrue("positions recorded: " + a.size(), a.size() >= 2);
            List<Sample> layer = new ArrayList<>();
            for (int p = 0; p < a.size(); p++) {
                Sample sample = new Sample(a.get(p)[1], a.get(p)[0], o.get(p)[0], o.get(p)[1]);
                for (float[] values : new float[][] {sample.x, sample.xb, sample.readout}) {
                    for (float v : values) {
                        assertTrue("a recorded activation is " + v, Float.isFinite(v));
                    }
                }
                layer.add(sample);
            }
            samples.add(layer);
        }
        return samples;
    }

    /**
     * A host projection that records its input and the residual stream, then computes as before.
     */
    private static final class Recorder extends FloatTensor {
        private final FloatTensor w;
        private final FloatTensor stream;
        final List<float[][]> calls = new ArrayList<>();

        Recorder(FloatTensor w, FloatTensor stream) {
            this.w = w;
            this.stream = stream;
        }

        @Override
        public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
            calls.add(new float[][] {values(that, dim1), values(stream, stream.size())});
            w.matmul(that, out, dim0, dim1);
        }

        private static float[] values(FloatTensor t, int n) {
            float[] copy = new float[n];
            for (int i = 0; i < n; i++) {
                copy[i] = t.getFloat(i);
            }
            return copy;
        }

        @Override
        public int size() {
            return w.size();
        }

        @Override
        public float getFloat(int index) {
            return w.getFloat(index);
        }

        @Override
        public void setFloat(int index, float value) {
            throw new UnsupportedOperationException("a projection's weights are read only");
        }

        @Override
        protected jdk.incubator.vector.FloatVector getFloatVector(
                jdk.incubator.vector.VectorSpecies<Float> species, int offset) {
            throw new UnsupportedOperationException("read through matmul only");
        }

        @Override
        @SuppressWarnings("deprecation")
        protected org.beehive.jitllm.format.GGMLType type() {
            return org.beehive.jitllm.format.GGMLType.F32;
        }

        @Override
        public DataType dataType() {
            return w.dataType();
        }

        @Override
        public MemorySegment asMemorySegment() {
            return w.asMemorySegment();
        }
    }

    /** The RMS norm's scale, which the layer's reduce leaves in {@code temp[0]}. */
    private static float rmsScale(float[] x, float eps) {
        float ss = 0.0f;
        for (float v : x) {
            ss += v * v;
        }
        ss /= x.length;
        ss += eps;
        return (float) (1.0 / Math.sqrt(ss));
    }

    /** The device's normalized activation, bit for bit, and close to the host model's own. */
    private static void assertNormed(String where, float[] normed, float[] host, FloatArray xb) {
        double largest = 0;
        double worst = 0;
        for (int k = 0; k < normed.length; k++) {
            assertEquals(
                    where + " normalized activation at " + k,
                    Float.floatToRawIntBits(normed[k]),
                    Float.floatToRawIntBits(xb.get(k)));
            largest = Math.max(largest, Math.abs(host[k]));
            worst = Math.max(worst, Math.abs(host[k] - normed[k]));
        }
        assertTrue(
                where + ": the host model normalized another input, " + worst + " of " + largest,
                worst <= 1e-5 * largest);
    }

    // ---------------------------------------------------------------- the quantization

    /** The activation quantization the device performs, repeated on the host. */
    private static final class Quantized {
        final int[] quants;
        final float[] scales;
        final int[] sums;

        Quantized(float[] x) {
            int blocks = x.length / QK;
            quants = new int[x.length];
            scales = new float[blocks];
            sums = new int[blocks];
            for (int block = 0; block < blocks; block++) {
                int base = block * QK;
                float maxAbs = 0.0f;
                for (int i = 0; i < QK; i++) {
                    maxAbs = Math.max(maxAbs, Math.abs(x[base + i]));
                }
                float inverse = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
                int sum = 0;
                for (int i = 0; i < QK; i++) {
                    float scaled = x[base + i] * inverse;
                    int q = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
                    q = Math.min(127, Math.max(-127, q));
                    quants[base + i] = q;
                    sum += q;
                }
                scales[block] = maxAbs / 127.0f;
                sums[block] = sum;
            }
        }

        /** The activation value the packed sum multiplies at {@code i}: quant times block scale. */
        double value(int i) {
            return (double) scales[i / QK] * quants[i];
        }

        void assertDevice(String where, IntArray q, FloatArray s, IntArray sum) {
            for (int block = 0; block < scales.length; block++) {
                assertEquals(
                        where + " block " + block + " scale",
                        Float.floatToRawIntBits(scales[block]),
                        Float.floatToRawIntBits(s.get(block)));
                assertEquals(where + " block " + block + " sum", sums[block], sum.get(block));
            }
            for (int g = 0; g < quants.length / 4; g++) {
                int packed = 0;
                for (int lane = 0; lane < 4; lane++) {
                    packed |= (quants[g * 4 + lane] & 0xFF) << (lane * 8);
                }
                assertEquals(where + " quants " + 4 * g + ".." + (4 * g + 3), packed, q.get(g));
            }
        }

        /** The activation each wrong reference reads in place of {@link #value}. */
        double[] wrong(int which, float[] exact) {
            int n = quants.length;
            double[] a = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] =
                        switch (which) {
                            case 0 -> value((i + 1) % n);
                            case 1 -> (double) scales[(i / QK + 1) % scales.length] * quants[i];
                            default -> exact[i];
                        };
            }
            return a;
        }
    }

    // ---------------------------------------------------------------- the references

    /** Rows compared, the worst ratio of difference to bound, and the wrong references' misses. */
    private static final class Totals {
        final String name;
        int rows;
        double worstRatio;
        final int[] wrongOutside = new int[WRONG.length];

        Totals(String name) {
            this.name = name;
        }

        void add(String where, float got, double expected, double magnitude, double[] wrong) {
            double bound = BOUND * magnitude;
            assertFalse(
                    where + ": " + got + " against " + expected + ", bound " + bound,
                    outside(got, expected, bound));
            rows++;
            if (bound > 0) {
                worstRatio = Math.max(worstRatio, Math.abs(got - expected) / bound);
            }
            for (int w = 0; w < WRONG.length; w++) {
                if (outside(got, wrong[w], bound)) {
                    wrongOutside[w]++;
                }
            }
        }

        void report() {
            System.out.printf(
                    "[PACKED] %s: %d rows, worst difference %.3f of the bound%n",
                    name, rows, worstRatio);
            for (int w = 0; w < WRONG.length; w++) {
                System.out.printf(
                        "[PACKED]   %s outside the bound in %d of %d rows%n",
                        WRONG[w], wrongOutside[w], rows);
                assertTrue(
                        String.format(
                                "%s: a reference with the %s is outside the bound in only %d of"
                                        + " %d rows",
                                name, WRONG[w], wrongOutside[w], rows),
                        2 * wrongOutside[w] > rows);
            }
        }
    }

    private static byte[] bytes(FloatTensor t, long expected) {
        MemorySegment segment = t.asMemorySegment();
        assertEquals("weight bytes", expected, segment.byteSize());
        return segment.toArray(ValueLayout.JAVA_BYTE);
    }

    private static float half(byte[] raw, int at) {
        int bits = (raw[at] & 0xFF) | ((raw[at + 1] & 0xFF) << 8);
        return new HalfFloat((short) bits).getFloat32();
    }

    /**
     * A Q8_0 matrix, decoded independently of the kernel: a scale and 32 signed bytes per block.
     */
    private static final class Q8_0Rows {
        final byte[] raw;
        final int rows;
        final int n;

        Q8_0Rows(FloatTensor t, int rows, int n) {
            this.raw = bytes(t, (long) rows * (n / QK) * Q8_0_BLOCK_BYTES);
            this.rows = rows;
            this.n = n;
            // The host tensor decodes the same weights, exactly: an FP16 scale times a byte.
            for (int row = 0; row < rows; row++) {
                for (int k = 0; k < n; k++) {
                    int base = (row * (n / QK) + k / QK) * Q8_0_BLOCK_BYTES;
                    assertEquals(
                            "Q8_0 weight " + row + "," + k,
                            half(raw, base) * raw[base + 2 + k % QK],
                            t.getFloat(row * n + k),
                            0.0f);
                }
            }
        }

        void check(String what, Quantized x, float[] exact, FloatArray got, Totals totals) {
            double[][] wrong = new double[WRONG.length][];
            for (int w = 0; w < WRONG.length; w++) {
                wrong[w] = x.wrong(w, exact);
            }
            for (int row = 0; row < rows; row++) {
                double expected = 0;
                double magnitude = 0;
                double[] wrongSums = new double[WRONG.length];
                for (int block = 0; block < n / QK; block++) {
                    int base = (row * (n / QK) + block) * Q8_0_BLOCK_BYTES;
                    double d = half(raw, base);
                    long dot = 0;
                    for (int i = 0; i < QK; i++) {
                        int k = block * QK + i;
                        int q = raw[base + 2 + i];
                        dot += (long) q * x.quants[k];
                        for (int w = 0; w < WRONG.length; w++) {
                            wrongSums[w] += d * q * wrong[w][k];
                        }
                    }
                    double term = d * x.scales[block] * dot;
                    expected += term;
                    magnitude += Math.abs(term);
                }
                totals.add(what + " row " + row, got.get(row), expected, magnitude, wrongSums);
            }
        }
    }

    // @formatter:off
    /**
     * A Q5_K matrix, decoded independently of the kernel. A super-block of 256 is {@code d} and
     * {@code dmin} (FP16), twelve bytes of six-bit sub-block scales and minima, 32 bytes of fifth
     * bits and 128 bytes of nibbles; a weight is {@code d * sc * q - dmin * m}.
     */
    // @formatter:on
    private static final class Q5_KRows {
        final byte[] raw;
        final int rows;
        final int n;

        /**
         * Each weight's five-bit quant, and each sub-block's {@code d * sc} and {@code dmin * m}.
         */
        private final byte[] quants;

        private final double[] scales;
        private final double[] minima;

        Q5_KRows(FloatTensor t, int rows, int n) {
            this.raw = bytes(t, (long) rows * (n / QK_K) * Q5_K_BLOCK_BYTES);
            this.rows = rows;
            this.n = n;
            quants = new byte[rows * n];
            scales = new double[rows * (n / QK)];
            minima = new double[rows * (n / QK)];
            for (int row = 0; row < rows; row++) {
                for (int sb = 0; sb < n / QK; sb++) {
                    int block = sb / 8;
                    int sub = sb - block * 8;
                    int base = (row * (n / QK_K) + block) * Q5_K_BLOCK_BYTES;
                    scales[row * (n / QK) + sb] =
                            (double) half(raw, base) * subScale(raw, base, sub);
                    minima[row * (n / QK) + sb] =
                            (double) half(raw, base + 2) * subMin(raw, base, sub);
                    for (int i = 0; i < QK; i++) {
                        quants[row * n + sb * QK + i] = (byte) quant(raw, base, sub * QK + i);
                    }
                }
            }
            // The host tensor decodes the same weights in float.
            for (int row = 0; row < rows; row++) {
                for (int k = 0; k < n; k++) {
                    double scale = scales[row * (n / QK) + k / QK];
                    double minimum = minima[row * (n / QK) + k / QK];
                    double w = scale * quants[row * n + k] - minimum;
                    double host = t.getFloat(row * n + k);
                    assertTrue(
                            "Q5_K weight " + row + "," + k + ": " + w + " against " + host,
                            Math.abs(w - host)
                                    <= 1e-6 * (Math.abs(scale * quants[row * n + k]) + minimum));
                }
            }
        }

        private static int quant(byte[] raw, int blockBase, int withinBlock) {
            int pair = withinBlock / 64;
            int inPair = withinBlock - pair * 64;
            int high = inPair / 32;
            int inHalf = inPair - high * 32;
            int qs = raw[blockBase + 48 + pair * 32 + inHalf] & 0xFF;
            int q = high == 0 ? (qs & 0xF) : ((qs >> 4) & 0xF);
            int qh = raw[blockBase + 16 + inHalf] & 0xFF;
            return q + ((qh >> (pair * 2 + high)) & 1) * 16;
        }

        private static int subScale(byte[] raw, int blockBase, int sub) {
            int base = blockBase + 4;
            if (sub < 4) {
                return raw[base + sub] & 63;
            }
            return (raw[base + sub + 4] & 0xF) | (((raw[base + sub - 4] & 0xFF) >> 6) << 4);
        }

        private static int subMin(byte[] raw, int blockBase, int sub) {
            int base = blockBase + 4;
            if (sub < 4) {
                return raw[base + sub + 4] & 63;
            }
            return ((raw[base + sub + 4] & 0xFF) >> 4) | (((raw[base + sub] & 0xFF) >> 6) << 4);
        }

        void check(
                String what,
                Quantized x,
                float[] exact,
                float[] residual,
                FloatArray got,
                Totals totals) {
            double[][] wrong = new double[WRONG.length][];
            for (int w = 0; w < WRONG.length; w++) {
                wrong[w] = x.wrong(w, exact);
            }
            for (int row = 0; row < rows; row++) {
                double expected = residual[row];
                double magnitude = Math.abs(residual[row]);
                double[] wrongSums = new double[WRONG.length];
                for (int w = 0; w < WRONG.length; w++) {
                    wrongSums[w] = residual[row];
                }
                for (int sb = 0; sb < n / QK; sb++) {
                    double scale = scales[row * (n / QK) + sb];
                    double minimum = minima[row * (n / QK) + sb];
                    long dot = 0;
                    for (int i = 0; i < QK; i++) {
                        int k = sb * QK + i;
                        int q = quants[row * n + k];
                        dot += (long) q * x.quants[k];
                        for (int w = 0; w < WRONG.length; w++) {
                            wrongSums[w] += (scale * q - minimum) * wrong[w][k];
                        }
                    }
                    double s = x.scales[sb];
                    expected += s * (scale * dot - minimum * x.sums[sb]);
                    magnitude +=
                            Math.abs(s) * (Math.abs(scale * dot) + Math.abs(minimum * x.sums[sb]));
                }
                totals.add(what + " row " + row, got.get(row), expected, magnitude, wrongSums);
            }
        }
    }

    // ---------------------------------------------------------------- the device

    /**
     * The layer's packed tasks on their own: the norm's apply with its quantization, {@code
     * ssm_alpha} and {@code ssm_beta}, the readout's quantization and {@code ssm_out}.
     */
    private static final class Device {
        final FloatArray x;
        final FloatArray norm;
        final FloatArray temp = new FloatArray(1);
        final FloatArray xb;
        final IntArray quantsA;
        final FloatArray scalesA;
        final IntArray sumsA;
        final ByteArray alphaW;
        final ByteArray betaW;
        final FloatArray alphaOut;
        final FloatArray betaOut;
        final FloatArray readout;
        final IntArray quantsO;
        final FloatArray scalesO;
        final IntArray sumsO;
        final ByteArray outW;
        final FloatArray hb;
        private final int dim;
        private final int heads;
        private final int valueDim;

        Device(int dim, int heads, int valueDim) {
            this.dim = dim;
            this.heads = heads;
            this.valueDim = valueDim;
            x = new FloatArray(dim);
            norm = new FloatArray(dim);
            xb = new FloatArray(dim);
            quantsA = new IntArray(dim / 4);
            scalesA = new FloatArray(dim / QK);
            sumsA = new IntArray(dim / QK);
            alphaW = new ByteArray(heads * (dim / QK) * Q8_0_BLOCK_BYTES);
            betaW = new ByteArray(heads * (dim / QK) * Q8_0_BLOCK_BYTES);
            alphaOut = new FloatArray(heads);
            betaOut = new FloatArray(heads);
            readout = new FloatArray(valueDim);
            quantsO = new IntArray(valueDim / 4);
            scalesO = new FloatArray(valueDim / QK);
            sumsO = new IntArray(valueDim / QK);
            outW = new ByteArray(dim * (valueDim / QK_K) * Q5_K_BLOCK_BYTES);
            hb = new FloatArray(dim);
        }

        TornadoExecutionPlan plan() {
            TaskGraph graph =
                    new TaskGraph("packed")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    x,
                                    norm,
                                    temp,
                                    xb,
                                    quantsA,
                                    scalesA,
                                    sumsA,
                                    alphaW,
                                    betaW,
                                    alphaOut,
                                    betaOut,
                                    readout,
                                    quantsO,
                                    scalesO,
                                    sumsO,
                                    outW,
                                    hb)
                            .task(
                                    "apply",
                                    TransformerComputeKernelsQ4_0
                                            ::rmsApplyAndQuantizeActivationQ8Blocks,
                                    new KernelContext(),
                                    xb,
                                    x,
                                    norm,
                                    temp,
                                    quantsA,
                                    scalesA,
                                    sumsA)
                            .task(
                                    "alpha",
                                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                    new KernelContext(),
                                    quantsA,
                                    scalesA,
                                    alphaOut,
                                    alphaW,
                                    dim,
                                    heads,
                                    LOCAL)
                            .task(
                                    "beta",
                                    TransformerComputeKernelsQ8_0DP4A::matrixVectorGenericQ8_0DP4A,
                                    new KernelContext(),
                                    quantsA,
                                    scalesA,
                                    betaOut,
                                    betaW,
                                    dim,
                                    heads,
                                    LOCAL)
                            .task(
                                    "quantize",
                                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                    new KernelContext(),
                                    readout,
                                    quantsO,
                                    scalesO,
                                    sumsO)
                            .task(
                                    "out",
                                    TransformerComputeKernelsQ5_K
                                            ::matrixVectorGenericWithResidualQ5_KDP4A,
                                    new KernelContext(),
                                    quantsO,
                                    scalesO,
                                    sumsO,
                                    hb,
                                    outW,
                                    valueDim,
                                    dim,
                                    LOCAL)
                            .transferToHost(
                                    DataTransferMode.EVERY_EXECUTION,
                                    xb,
                                    quantsA,
                                    scalesA,
                                    sumsA,
                                    alphaOut,
                                    betaOut,
                                    quantsO,
                                    scalesO,
                                    sumsO,
                                    hb);
            GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("packed.apply", PackedActivationTasks.quantizeWorker(dim));
            scheduler.addWorkerGrid(
                    "packed.alpha", PackedActivationTasks.projectionWorker(heads, LOCAL));
            scheduler.addWorkerGrid(
                    "packed.beta", PackedActivationTasks.projectionWorker(heads, LOCAL));
            scheduler.addWorkerGrid(
                    "packed.quantize", PackedActivationTasks.quantizeWorker(valueDim));
            scheduler.addWorkerGrid(
                    "packed.out", PackedActivationTasks.projectionWorker(dim, LOCAL));
            TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot());
            plan.withGridScheduler(scheduler);
            return plan;
        }

        void load(byte[] alpha, byte[] beta, byte[] out, float[] weights) {
            fill(alphaW, alpha);
            fill(betaW, beta);
            fill(outW, out);
            for (int k = 0; k < dim; k++) {
                norm.set(k, weights[k]);
            }
        }

        private static void fill(ByteArray array, byte[] values) {
            assertEquals("weight bytes", array.getSize(), values.length);
            for (int i = 0; i < values.length; i++) {
                array.set(i, values[i]);
            }
        }

        /** One position: every output starts as NaN or a pattern no quantization produces. */
        void run(TornadoExecutionPlan plan, Sample sample, float ss) {
            for (int k = 0; k < dim; k++) {
                x.set(k, sample.x[k]);
                hb.set(k, sample.residual[k]);
            }
            for (int k = 0; k < valueDim; k++) {
                readout.set(k, sample.readout[k]);
            }
            temp.set(0, ss);
            xb.init(Float.NaN);
            alphaOut.init(Float.NaN);
            betaOut.init(Float.NaN);
            scalesA.init(Float.NaN);
            scalesO.init(Float.NaN);
            quantsA.init(0x80808080);
            quantsO.init(0x80808080);
            sumsA.init(Integer.MIN_VALUE);
            sumsO.init(Integer.MIN_VALUE);
            plan.execute();
        }
    }
}

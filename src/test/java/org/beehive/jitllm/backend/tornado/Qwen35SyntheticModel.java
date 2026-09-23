package org.beehive.jllm.backend.tornado;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jllm.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.beehive.jllm.tensor.standard.Q4_0FloatTensor;
import org.beehive.jllm.tensor.standard.Q4_1FloatTensor;
import org.beehive.jllm.tensor.standard.Q5_KFloatTensor;
import org.beehive.jllm.tensor.standard.Q6_KFloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * A {@code qwen35} model small enough to build in a test and shaped like the real one.
 *
 * <p>Every weight is generated once and wrapped twice — as the host tensor and as the device tensor
 * for that representation, over the same bytes — so a disagreement between the two paths is the
 * engine's and not the fixture's.
 *
 * <p>The dimensions are the 27B's own wherever a shape decides whether a kernel is correct: a
 * 256-wide attention head, a 128-wide delta-net head, 48 value heads against 16 key heads, a
 * partial rotation of 64. What is shrunk is the layer count and the vocabulary, which no kernel
 * branches on.
 *
 * <p>The mixture is the real one too: Q4_1 down projections on the early blocks, Q5_K recurrent
 * outputs, a Q6_K vocabulary projection, Q4_0 elsewhere, F32 norms and SSM parameters.
 */
// @formatter:on
final class Qwen35SyntheticModel {

    private final Random random = new Random(20260908L);
    private final Arena arena;

    Qwen35SyntheticModel(Arena arena) {
        this.arena = arena;
    }

    static final int DIM = 5120;
    static final int HIDDEN = 17408;
    static final int TRUNK = 8;
    static final int BLOCKS = TRUNK + 1;
    static final int HEADS = 24;
    static final int KV_HEADS = 4;
    static final int HEAD_DIM = 256;
    static final int INTERVAL = 4;
    static final int CONV_KERNEL = 4;
    static final int STATE_SIZE = 128;
    static final int GROUPS = 16;
    static final int VALUE_HEADS = 48;
    static final int INNER = 6144;
    static final int ROPE_DIM = 64;
    static final int VOCAB = 2048;
    static final int CONTEXT = 32;

    static Qwen35Configuration config() {
        return config(INTERVAL);
    }

    /**
     * The same model with a stated attention interval.
     *
     * <p>{@code isRecurrentLayer} is {@code (l + 1) % interval != 0}, so an interval of one makes
     * every block attend and an interval past the layer count makes every block recurrent. That is
     * what lets a test separate the two mixers instead of inferring which one moved.
     */
    static Qwen35Configuration config(int attentionInterval) {
        return new Qwen35Configuration(
                "Q8_0",
                DIM,
                HIDDEN,
                TRUNK,
                1,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                attentionInterval,
                CONV_KERNEL,
                STATE_SIZE,
                GROUPS,
                VALUE_HEADS,
                INNER,
                ROPE_DIM,
                VOCAB,
                CONTEXT,
                CONTEXT,
                1e-6f,
                1e7f);
    }

    /** One weight, in both representations of itself. */
    record Pair(FloatTensor host, TornadoTensor device) {}

    private Pair floats(int elements, float scale) {
        float[] values = new float[elements];
        for (int i = 0; i < elements; i++) {
            values[i] = (float) random.nextGaussian() * scale;
        }
        FloatArray device = new FloatArray(elements);
        for (int i = 0; i < elements; i++) {
            device.set(i, values[i]);
        }
        return new Pair(new ArrayFloatTensor(values), new FP32TornadoTensor(device));
    }

    /**
     * A quantized weight: random block payloads with controlled fp16 scales, wrapped as both the
     * host tensor and the device tensor over the identical bytes.
     */
    private Pair quantized(DataType type, int elements) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1 -> 32;
                    case Q5_K, Q6_K -> 256;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        int blockBytes =
                switch (type) {
                    case Q4_0 -> 18;
                    case Q4_1 -> 20;
                    case Q5_K -> 176;
                    case Q6_K -> 210;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        int[] scaleOffsets =
                switch (type) {
                    case Q4_0 -> new int[] {0};
                    case Q4_1, Q5_K -> new int[] {0, 2};
                    case Q6_K -> new int[] {208};
                    default -> throw new IllegalArgumentException(type.toString());
                };
        int blocks = elements / blockSize;
        byte[] raw = new byte[blocks * blockBytes];
        random.nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            for (int offset : scaleOffsets) {
                // ~0.02 to ~0.09: small, normal and different per block, so a whole forward pass
                // stays in a range where the two paths are comparable term by term.
                raw[b * blockBytes + offset] = (byte) (b & 0xFF);
                raw[b * blockBytes + offset + 1] = 0x24;
            }
        }

        MemorySegment segment = arena.allocate(raw.length);
        MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
        ByteArray bytes = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            bytes.set(i, raw[i]);
        }

        return switch (type) {
            case Q4_0 ->
                    new Pair(new Q4_0FloatTensor(elements, segment), new Q4_0TornadoTensor(bytes));
            case Q4_1 ->
                    new Pair(new Q4_1FloatTensor(elements, segment), new Q4_1TornadoTensor(bytes));
            case Q5_K ->
                    new Pair(new Q5_KFloatTensor(elements, segment), new Q5_KTornadoTensor(bytes));
            case Q6_K ->
                    new Pair(new Q6_KFloatTensor(elements, segment), new Q6_KTornadoTensor(bytes));
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /** Both weight sets, over one set of bytes. */
    record Weights(Qwen35StandardWeights host, Qwen35TornadoWeights device) {}

    Weights weights(Qwen35Configuration config) {
        FloatTensor[] hAttnNorm = new FloatTensor[BLOCKS];
        TornadoTensor[] dAttnNorm = new TornadoTensor[BLOCKS];
        FloatTensor[] hFfnNorm = new FloatTensor[BLOCKS];
        TornadoTensor[] dFfnNorm = new TornadoTensor[BLOCKS];
        FloatTensor[] hGate = new FloatTensor[BLOCKS];
        TornadoTensor[] dGate = new TornadoTensor[BLOCKS];
        FloatTensor[] hDown = new FloatTensor[BLOCKS];
        TornadoTensor[] dDown = new TornadoTensor[BLOCKS];
        FloatTensor[] hUp = new FloatTensor[BLOCKS];
        TornadoTensor[] dUp = new TornadoTensor[BLOCKS];
        FloatTensor[] hWq = new FloatTensor[BLOCKS];
        TornadoTensor[] dWq = new TornadoTensor[BLOCKS];
        FloatTensor[] hWk = new FloatTensor[BLOCKS];
        TornadoTensor[] dWk = new TornadoTensor[BLOCKS];
        FloatTensor[] hWv = new FloatTensor[BLOCKS];
        TornadoTensor[] dWv = new TornadoTensor[BLOCKS];
        FloatTensor[] hWo = new FloatTensor[BLOCKS];
        TornadoTensor[] dWo = new TornadoTensor[BLOCKS];
        FloatTensor[] hQNorm = new FloatTensor[BLOCKS];
        TornadoTensor[] dQNorm = new TornadoTensor[BLOCKS];
        FloatTensor[] hKNorm = new FloatTensor[BLOCKS];
        TornadoTensor[] dKNorm = new TornadoTensor[BLOCKS];
        FloatTensor[] hQkv = new FloatTensor[TRUNK];
        TornadoTensor[] dQkv = new TornadoTensor[TRUNK];
        FloatTensor[] hSsmGate = new FloatTensor[TRUNK];
        TornadoTensor[] dSsmGate = new TornadoTensor[TRUNK];
        FloatTensor[] hConv = new FloatTensor[TRUNK];
        TornadoTensor[] dConv = new TornadoTensor[TRUNK];
        FloatTensor[] hAlpha = new FloatTensor[TRUNK];
        TornadoTensor[] dAlpha = new TornadoTensor[TRUNK];
        FloatTensor[] hBeta = new FloatTensor[TRUNK];
        TornadoTensor[] dBeta = new TornadoTensor[TRUNK];
        FloatTensor[] hDtBias = new FloatTensor[TRUNK];
        TornadoTensor[] dDtBias = new TornadoTensor[TRUNK];
        FloatTensor[] hA = new FloatTensor[TRUNK];
        TornadoTensor[] dA = new TornadoTensor[TRUNK];
        FloatTensor[] hSsmNorm = new FloatTensor[TRUNK];
        TornadoTensor[] dSsmNorm = new TornadoTensor[TRUNK];
        FloatTensor[] hSsmOut = new FloatTensor[TRUNK];
        TornadoTensor[] dSsmOut = new TornadoTensor[TRUNK];

        for (int l = 0; l < BLOCKS; l++) {
            Pair attnNorm = floats(DIM, 0.1f);
            hAttnNorm[l] = attnNorm.host();
            dAttnNorm[l] = attnNorm.device();
            Pair ffnNorm = floats(DIM, 0.1f);
            hFfnNorm[l] = ffnNorm.host();
            dFfnNorm[l] = ffnNorm.device();

            Pair gate = quantized(DataType.Q4_0, DIM * HIDDEN);
            hGate[l] = gate.host();
            dGate[l] = gate.device();
            Pair up = quantized(DataType.Q4_0, DIM * HIDDEN);
            hUp[l] = up.host();
            dUp[l] = up.device();
            // The 27B holds Q4_1 down projections on its early blocks only.
            Pair down = quantized(l < 2 ? DataType.Q4_1 : DataType.Q4_0, HIDDEN * DIM);
            hDown[l] = down.host();
            dDown[l] = down.device();

            if (l < TRUNK && config.isRecurrentLayer(l)) {
                Pair qkv = quantized(DataType.Q4_0, DIM * config.deltaNetConvDim());
                hQkv[l] = qkv.host();
                dQkv[l] = qkv.device();
                Pair z = quantized(DataType.Q4_0, DIM * config.deltaNetValueDim());
                hSsmGate[l] = z.host();
                dSsmGate[l] = z.device();
                Pair conv = floats(config.deltaNetConvDim() * CONV_KERNEL, 0.3f);
                hConv[l] = conv.host();
                dConv[l] = conv.device();
                Pair alpha = floats(DIM * VALUE_HEADS, 0.05f);
                hAlpha[l] = alpha.host();
                dAlpha[l] = alpha.device();
                Pair beta = floats(DIM * VALUE_HEADS, 0.05f);
                hBeta[l] = beta.host();
                dBeta[l] = beta.device();
                Pair dtBias = floats(VALUE_HEADS, 0.5f);
                hDtBias[l] = dtBias.host();
                dDtBias[l] = dtBias.device();
                // ssm_a is -exp(A_log) in the file, so it is negative and the decay is in (0, 1).
                float[] a = new float[VALUE_HEADS];
                FloatArray aDevice = new FloatArray(VALUE_HEADS);
                for (int h = 0; h < VALUE_HEADS; h++) {
                    a[h] = -(float) Math.exp(random.nextGaussian() * 0.5);
                    aDevice.set(h, a[h]);
                }
                hA[l] = new ArrayFloatTensor(a);
                dA[l] = new FP32TornadoTensor(aDevice);
                Pair ssmNorm = floats(config.headValueDim(), 0.5f);
                hSsmNorm[l] = ssmNorm.host();
                dSsmNorm[l] = ssmNorm.device();
                Pair out = quantized(DataType.Q5_K, config.deltaNetValueDim() * DIM);
                hSsmOut[l] = out.host();
                dSsmOut[l] = out.device();
            } else {
                Pair wq = quantized(DataType.Q4_0, DIM * config.queryGateDim());
                hWq[l] = wq.host();
                dWq[l] = wq.device();
                Pair wk = quantized(DataType.Q4_0, DIM * config.kvDim());
                hWk[l] = wk.host();
                dWk[l] = wk.device();
                Pair wv = quantized(DataType.Q4_0, DIM * config.kvDim());
                hWv[l] = wv.host();
                dWv[l] = wv.device();
                Pair wo = quantized(DataType.Q4_0, config.attentionOutputInputDim() * DIM);
                hWo[l] = wo.host();
                dWo[l] = wo.device();
                Pair qNorm = floats(HEAD_DIM, 0.5f);
                hQNorm[l] = qNorm.host();
                dQNorm[l] = qNorm.device();
                Pair kNorm = floats(HEAD_DIM, 0.5f);
                hKNorm[l] = kNorm.host();
                dKNorm[l] = kNorm.device();
            }
        }

        Pair embeddings = quantized(DataType.Q4_0, VOCAB * DIM);
        Pair outputNorm = floats(DIM, 0.2f);
        Pair output = quantized(DataType.Q6_K, VOCAB * DIM);

        // The rotary tables, computed once and shared: the two paths must index them identically.
        int half = ROPE_DIM / 2;
        float[] real = new float[CONTEXT * half];
        float[] imag = new float[CONTEXT * half];
        FloatArray realDevice = new FloatArray(real.length);
        FloatArray imagDevice = new FloatArray(imag.length);
        for (int position = 0; position < CONTEXT; position++) {
            for (int i = 0; i < half; i++) {
                double frequency = 1.0 / Math.pow(1e7, (2.0 * i) / ROPE_DIM);
                double angle = position * frequency;
                real[position * half + i] = (float) Math.cos(angle);
                imag[position * half + i] = (float) Math.sin(angle);
                realDevice.set(position * half + i, real[position * half + i]);
                imagDevice.set(position * half + i, imag[position * half + i]);
            }
        }

        Qwen35StandardWeights host =
                new Qwen35StandardWeights(
                        BLOCKS,
                        embeddings.host(),
                        hAttnNorm,
                        hFfnNorm,
                        hGate,
                        hDown,
                        hUp,
                        outputNorm.host(),
                        output.host(),
                        new ArrayFloatTensor(real),
                        new ArrayFloatTensor(imag),
                        hWq,
                        hWk,
                        hWv,
                        hWo,
                        hQNorm,
                        hKNorm,
                        hQkv,
                        hSsmGate,
                        hConv,
                        hAlpha,
                        hBeta,
                        hDtBias,
                        hA,
                        hSsmNorm,
                        hSsmOut,
                        new FloatTensor[BLOCKS],
                        new FloatTensor[BLOCKS],
                        new FloatTensor[BLOCKS],
                        new FloatTensor[BLOCKS],
                        DataType.Q4_0);

        Qwen35TornadoWeights device =
                new Qwen35TornadoWeights(
                        BLOCKS,
                        embeddings.device(),
                        dAttnNorm,
                        dFfnNorm,
                        dGate,
                        dDown,
                        dUp,
                        outputNorm.device(),
                        output.device(),
                        new FP32TornadoTensor(realDevice),
                        new FP32TornadoTensor(imagDevice),
                        dWq,
                        dWk,
                        dWv,
                        dWo,
                        dQNorm,
                        dKNorm,
                        dQkv,
                        dSsmGate,
                        dConv,
                        dAlpha,
                        dBeta,
                        dDtBias,
                        dA,
                        dSsmNorm,
                        dSsmOut,
                        DataType.Q4_0);

        return new Weights(host, device);
    }
}

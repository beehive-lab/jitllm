package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.beehive.jllm.inference.op.CpuOperations;
import org.beehive.jllm.model.loader.RopeFrequencies;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The {@code qwen35} attention kernels against the host operations, lane by lane.
 *
 * <p>Qwen3.8-27B's own geometry, and every number in it is one that breaks an assumption some other
 * family is allowed to make:
 *
 * <ul>
 *   <li><b>Head width 256 while {@code dim / heads} is 213.</b> Anything deriving the head
 *       dimension mis-addresses every head.
 *   <li><b>Rotary width 64 of that 256.</b> Three quarters of each head must come through
 *       untouched, and a kernel that rotated the whole head would still produce fluent text.
 *   <li><b>The query projection is twice as wide as the query</b>, carrying an interleaved gate.
 *       Splitting on the wrong boundary swaps a query for a gate, which is again well-formed.
 *   <li><b>24 query heads against 4 key/value heads</b>, so only the first four rotate their keys.
 * </ul>
 */
public class Qwen35AttentionKernelParityTest {

    private static final int HEADS = 24;
    private static final int KV_HEADS = 4;
    private static final int HEAD_DIM = 256;
    private static final int ROTARY_DIM = 64;
    private static final int CONTEXT = 64;
    private static final float ROPE_THETA = 1e7f;

    private static final int QUERY_DIM = HEADS * HEAD_DIM;
    private static final int FUSED_DIM = QUERY_DIM * 2;
    private static final int KV_DIM = KV_HEADS * HEAD_DIM;

    /** Bit-exact: the two sides perform the same operations in the same order. */
    private static final float EXACT = 0f;

    private final Random random = new Random(20260908L);

    private float[] noise(int n, float scale) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = (float) random.nextGaussian() * scale;
        }
        return values;
    }

    private static FloatArray toDevice(float[] values) {
        FloatArray array = new FloatArray(values.length);
        for (int i = 0; i < values.length; i++) {
            array.set(i, values[i]);
        }
        return array;
    }

    private static void assertSame(String what, FloatTensor host, FloatArray device) {
        for (int i = 0; i < host.size(); i++) {
            assertEquals(what + "[" + i + "]", host.getFloat(i), device.get(i), EXACT);
        }
    }

    /** The host's de-interleave, which lives in the forward pass rather than a named operation. */
    private static void hostSplit(float[] fused, float[] query, float[] gate) {
        for (int head = 0; head < HEADS; head++) {
            int fusedBase = head * 2 * HEAD_DIM;
            System.arraycopy(fused, fusedBase, query, head * HEAD_DIM, HEAD_DIM);
            System.arraycopy(fused, fusedBase + HEAD_DIM, gate, head * HEAD_DIM, HEAD_DIM);
        }
    }

    @Test
    public void theQueryGateSplitMatchesTheHost() {
        float[] fused = noise(FUSED_DIM, 1.0f);

        float[] hostQuery = new float[QUERY_DIM];
        float[] hostGate = new float[QUERY_DIM];
        hostSplit(fused, hostQuery, hostGate);

        FloatArray deviceFused = toDevice(fused);
        FloatArray deviceQuery = new FloatArray(QUERY_DIM);
        FloatArray deviceGate = new FloatArray(QUERY_DIM);
        for (int lane = 0; lane < QUERY_DIM; lane++) {
            Qwen35AttentionKernels.splitQueryGateLane(
                    deviceFused, deviceQuery, deviceGate, HEAD_DIM, lane);
        }

        assertSame("query", new ArrayFloatTensor(hostQuery), deviceQuery);
        assertSame("gate", new ArrayFloatTensor(hostGate), deviceGate);
    }

    /**
     * The split must be per head, not "first half is query, second half is gate".
     *
     * <p>Asserted directly because both sides could adopt the same wrong boundary: the whole-buffer
     * split is the obvious reading of a tensor twice the expected width, and it produces a query
     * made of the first twelve heads' queries and gates.
     */
    @Test
    public void theSplitIsPerHeadNotPerBuffer() {
        // Mark every element with its head and which half of that head it is in.
        float[] fused = new float[FUSED_DIM];
        for (int head = 0; head < HEADS; head++) {
            for (int i = 0; i < HEAD_DIM; i++) {
                fused[head * 2 * HEAD_DIM + i] = head; // query slice
                fused[head * 2 * HEAD_DIM + HEAD_DIM + i] = -head - 1; // gate slice
            }
        }

        FloatArray query = new FloatArray(QUERY_DIM);
        FloatArray gate = new FloatArray(QUERY_DIM);
        for (int lane = 0; lane < QUERY_DIM; lane++) {
            Qwen35AttentionKernels.splitQueryGateLane(toDevice(fused), query, gate, HEAD_DIM, lane);
        }

        for (int head = 0; head < HEADS; head++) {
            assertEquals("query head " + head, head, query.get(head * HEAD_DIM + 7), EXACT);
            assertEquals("gate head " + head, -head - 1, gate.get(head * HEAD_DIM + 7), EXACT);
        }
    }

    @Test
    public void thePartialRotationMatchesTheHost() {
        var freqs =
                RopeFrequencies.precomputeFreqsCis(
                        CONTEXT, ROTARY_DIM, ROPE_THETA, false, 0, 0, 0, 0);
        float[] real = freqs.first();
        float[] imag = freqs.second();

        float[] query = noise(QUERY_DIM, 1.0f);
        float[] key = noise(KV_DIM, 1.0f);

        for (int position : new int[] {0, 1, 17, CONTEXT - 1}) {
            FloatTensor hostQuery = new ArrayFloatTensor(query.clone());
            FloatTensor hostKey = new ArrayFloatTensor(key.clone());
            CpuOperations.ropeNeoxPartial(
                    hostQuery,
                    HEADS,
                    HEAD_DIM,
                    ROTARY_DIM,
                    position,
                    new ArrayFloatTensor(real),
                    new ArrayFloatTensor(imag));
            CpuOperations.ropeNeoxPartial(
                    hostKey,
                    KV_HEADS,
                    HEAD_DIM,
                    ROTARY_DIM,
                    position,
                    new ArrayFloatTensor(real),
                    new ArrayFloatTensor(imag));

            FloatArray deviceQuery = toDevice(query);
            FloatArray deviceKey = toDevice(key);
            FloatArray deviceReal = toDevice(real);
            FloatArray deviceImag = toDevice(imag);
            for (int lane = 0; lane < HEADS * (ROTARY_DIM / 2); lane++) {
                Qwen35AttentionKernels.ropeNeoxPartialLane(
                        deviceQuery,
                        deviceKey,
                        deviceReal,
                        deviceImag,
                        position,
                        KV_HEADS,
                        HEAD_DIM,
                        ROTARY_DIM,
                        lane);
            }

            assertSame("query at " + position, hostQuery, deviceQuery);
            assertSame("key at " + position, hostKey, deviceKey);
        }
    }

    /** Everything at or above the rotary width passes through, on both sides. */
    @Test
    public void theTailOfEachHeadIsNotRotated() {
        var freqs =
                RopeFrequencies.precomputeFreqsCis(
                        CONTEXT, ROTARY_DIM, ROPE_THETA, false, 0, 0, 0, 0);
        float[] query = noise(QUERY_DIM, 1.0f);
        float[] key = noise(KV_DIM, 1.0f);

        FloatArray deviceQuery = toDevice(query);
        FloatArray deviceKey = toDevice(key);
        for (int lane = 0; lane < HEADS * (ROTARY_DIM / 2); lane++) {
            Qwen35AttentionKernels.ropeNeoxPartialLane(
                    deviceQuery,
                    deviceKey,
                    toDevice(freqs.first()),
                    toDevice(freqs.second()),
                    5,
                    KV_HEADS,
                    HEAD_DIM,
                    ROTARY_DIM,
                    lane);
        }

        for (int head = 0; head < HEADS; head++) {
            for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
                int index = head * HEAD_DIM + i;
                assertEquals(
                        "query untouched at " + index, query[index], deviceQuery.get(index), EXACT);
            }
        }
        // The key buffer holds only KV_HEADS heads. A lane for a query head past that must not
        // touch it at all — the buffer is exactly KV_DIM long, so addressing one would throw here
        // rather than corrupt something, which is why the loop above runs over all HEADS lanes.
        for (int head = 0; head < KV_HEADS; head++) {
            for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
                int index = head * HEAD_DIM + i;
                assertEquals("key untouched at " + index, key[index], deviceKey.get(index), EXACT);
            }
        }
    }

    /**
     * The append lands both vectors at the paged offset for this (layer, position).
     *
     * <p>The offset is recomputed here from the block layout rather than taken from the same helper
     * the kernel uses, which would prove only that the helper equals itself. The layout is the one
     * {@code State.fillKvFields} allocates: blocks of {@code blockSize} positions, each holding
     * every layer's slice, addressed through a block table that is the identity for one sequence.
     */
    @Test
    public void theAppendLandsAtThePagedOffset() {
        int blockSize = 16;
        int layers = 8;
        int blocksPerSeq = (CONTEXT + blockSize - 1) / blockSize;
        int blockCfg = blockSize | (blocksPerSeq << 16);
        int blockStride = layers * blockSize * KV_DIM;

        IntArray blockTable = new IntArray(blocksPerSeq);
        for (int b = 0; b < blocksPerSeq; b++) {
            blockTable.set(b, b);
        }

        float[] key = noise(KV_DIM, 1.0f);
        float[] value = noise(KV_DIM, 1.0f);
        FloatArray keyCache = new FloatArray(blocksPerSeq * blockStride);
        FloatArray valueCache = new FloatArray(blocksPerSeq * blockStride);

        int layer = 3;
        int position = 21; // block 1, offset 5 within it
        for (int lane = 0; lane < KV_DIM; lane++) {
            Qwen35AttentionKernels.appendKeyValueLane(
                    toDevice(key),
                    toDevice(value),
                    keyCache,
                    valueCache,
                    blockTable,
                    position,
                    0,
                    KV_DIM,
                    layer,
                    blockCfg,
                    blockStride,
                    lane);
        }

        int expectedBase =
                (position / blockSize) * blockStride
                        + layer * blockSize * KV_DIM
                        + (position % blockSize) * KV_DIM;
        for (int i = 0; i < KV_DIM; i++) {
            assertEquals("key at " + i, key[i], keyCache.get(expectedBase + i), EXACT);
            assertEquals("value at " + i, value[i], valueCache.get(expectedBase + i), EXACT);
        }
        // Nothing else was written: a wrong stride would land inside another layer's slice, which
        // reads as a plausible cache rather than as a failure.
        for (int i = 0; i < keyCache.getSize(); i++) {
            if (i < expectedBase || i >= expectedBase + KV_DIM) {
                assertEquals("key cache disturbed at " + i, 0f, keyCache.get(i), EXACT);
            }
        }
    }

    @Test
    public void theOutputGateMatchesTheHost() {
        float[] values = noise(QUERY_DIM, 1.0f);
        float[] gate = noise(QUERY_DIM, 1.0f);

        float[] host = values.clone();
        for (int i = 0; i < QUERY_DIM; i++) {
            host[i] = host[i] * CpuOperations.logistic(gate[i]);
        }

        FloatArray device = toDevice(values);
        FloatArray deviceGate = toDevice(gate);
        for (int lane = 0; lane < QUERY_DIM; lane++) {
            Qwen35AttentionKernels.applyOutputGateLane(device, deviceGate, lane);
        }

        for (int i = 0; i < QUERY_DIM; i++) {
            // A logistic, evaluated in double on the host and in float on the device.
            assertEquals(
                    "gated[" + i + "]",
                    host[i],
                    device.get(i),
                    Math.max(1e-9f, Math.abs(host[i]) * 1e-6f));
        }
    }
}

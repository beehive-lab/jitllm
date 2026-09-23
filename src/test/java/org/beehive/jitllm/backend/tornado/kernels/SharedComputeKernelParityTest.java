package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.beehive.jllm.inference.op.CpuOperations;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The two format-neutral primitives a recurrent mixer needs, run lane by lane on the host against
 * the operations they restate.
 *
 * <p>Both are addressing rather than arithmetic, which is precisely why they get a test: an
 * off-by-one in a three-way split takes the value slice from inside the keys and produces fluent,
 * wrong output. The cases below therefore include unequal widths, a boundary element on each side
 * of both cuts, zeroes, and negative values — a split that dropped a sign or a scale that skipped
 * an element would be invisible in a positive-only fixture.
 */
public class SharedComputeKernelParityTest {

    private static FloatArray toDevice(float[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, v[i]);
        }
        return a;
    }

    /** The 27B's own widths: {@code 2048 | 2048 | 6144}, which are not (q, kv, kv). */
    @Test
    public void theThreeWaySplitMatchesTheHostSlices() {
        final int dimA = 2048;
        final int dimB = 2048;
        final int dimC = 6144;
        final int total = dimA + dimB + dimC;

        Random random = new Random(20260908L);
        float[] fused = new float[total];
        for (int i = 0; i < total; i++) {
            fused[i] = (float) random.nextGaussian();
        }
        // The elements either side of both cuts, and a zero, pinned explicitly.
        fused[dimA - 1] = -7.5f;
        fused[dimA] = 7.5f;
        fused[dimA + dimB - 1] = -0.25f;
        fused[dimA + dimB] = 0.0f;
        fused[total - 1] = 3.25f;

        FloatTensor hostA = ArrayFloatTensor.allocate(dimA);
        FloatTensor hostB = ArrayFloatTensor.allocate(dimB);
        FloatTensor hostC = ArrayFloatTensor.allocate(dimC);
        FloatTensor hostFused = new ArrayFloatTensor(fused.clone());
        hostFused.copyTo(0, hostA, 0, dimA);
        hostFused.copyTo(dimA, hostB, 0, dimB);
        hostFused.copyTo(dimA + dimB, hostC, 0, dimC);

        FloatArray device = toDevice(fused);
        FloatArray a = new FloatArray(dimA);
        FloatArray b = new FloatArray(dimB);
        FloatArray c = new FloatArray(dimC);
        for (int lane = 0; lane < total; lane++) {
            TransformerComputeKernels.splitThreeWayLane(device, a, b, c, dimA, dimB, lane);
        }

        for (int i = 0; i < dimA; i++) {
            assertEquals("a[" + i + "]", hostA.getFloat(i), a.get(i), 0.0f);
        }
        for (int i = 0; i < dimB; i++) {
            assertEquals("b[" + i + "]", hostB.getFloat(i), b.get(i), 0.0f);
        }
        for (int i = 0; i < dimC; i++) {
            assertEquals("c[" + i + "]", hostC.getFloat(i), c.get(i), 0.0f);
        }
        // The source is untouched: nothing here writes into what another lane still reads.
        for (int i = 0; i < total; i++) {
            assertEquals("fused[" + i + "] was modified", fused[i], device.get(i), 0.0f);
        }
    }

    /** A width that is not a multiple of any workgroup size, so the tail is exercised. */
    @Test
    public void theThreeWaySplitHandlesUnalignedWidths() {
        final int dimA = 5;
        final int dimB = 1;
        final int dimC = 7;
        float[] fused = {
            1f, -1f, 0f, 2.5f, -2.5f, // a
            -9f, // b
            3f, -3f, 0f, 4.5f, -4.5f, 6f, -6f // c
        };

        FloatArray device = toDevice(fused);
        FloatArray a = new FloatArray(dimA);
        FloatArray b = new FloatArray(dimB);
        FloatArray c = new FloatArray(dimC);
        for (int lane = 0; lane < dimA + dimB + dimC; lane++) {
            TransformerComputeKernels.splitThreeWayLane(device, a, b, c, dimA, dimB, lane);
        }

        for (int i = 0; i < dimA; i++) {
            assertEquals(fused[i], a.get(i), 0.0f);
        }
        assertEquals(fused[dimA], b.get(0), 0.0f);
        for (int i = 0; i < dimC; i++) {
            assertEquals(fused[dimA + dimB + i], c.get(i), 0.0f);
        }
    }

    /** {@code 1/sqrt(128)}, the factor the delta-net queries carry, over signed values. */
    @Test
    public void theInPlaceScaleMatchesTheHostOperation() {
        final int size = 2048;
        final float factor = (float) (1.0 / Math.sqrt(128));

        Random random = new Random(4L);
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (float) random.nextGaussian() * 3.0f;
        }
        values[0] = 0.0f;
        values[1] = -1.0f;
        values[size - 1] = 1e-8f;

        FloatTensor host = new ArrayFloatTensor(values.clone());
        CpuOperations.scale(host, factor);

        FloatArray device = toDevice(values);
        for (int lane = 0; lane < size; lane++) {
            TransformerComputeKernels.scaleInPlaceLane(device, factor, lane);
        }

        for (int i = 0; i < size; i++) {
            assertEquals("scaled[" + i + "]", host.getFloat(i), device.get(i), 0.0f);
        }
    }
}

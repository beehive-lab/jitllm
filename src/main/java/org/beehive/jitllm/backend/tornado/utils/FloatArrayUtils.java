package org.beehive.jitllm.backend.tornado.utils;

import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Helper class for FloatArray operations that mirror the functionality of FloatTensor methods.
 * Provides utility methods for common tensor operations on TornadoVM's FloatArray type.
 */
public final class FloatArrayUtils {

    private FloatArrayUtils() {
        // Utility class, not meant to be instantiated
    }

    /**
     * Divides all elements in the specified range of a FloatArray by a value in-place. Mirrors the
     * functionality of FloatTensor.divideInPlace().
     *
     * @param array The FloatArray to modify
     * @param start The starting index (inclusive)
     * @param end The ending index (exclusive)
     * @param value The value to divide by
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray divideInPlace(FloatArray array, int start, int end, float value) {
        for (int i = start; i < end; i++) {
            array.set(i, array.get(i) / value);
        }
        return array;
    }

    /**
     * Applies the softmax function to a range of elements in a FloatArray in-place. Mirrors the
     * functionality of FloatTensor.softmaxInPlace().
     *
     * @param array The FloatArray to modify
     * @param start The starting index (inclusive)
     * @param end The ending index (exclusive)
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray softmaxInPlace(FloatArray array, int start, int end) {
        // Find max value for numerical stability
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = start; i < end; i++) {
            float val = array.get(i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Apply exp(x-max) to each element and calculate sum
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            float exp;
            if (TornadoVMSupport.isTornadoVMEnabled()) {
                // Use TornadoMath for GPU execution if possible
                exp = TornadoMath.exp(array.get(i) - maxVal);
            } else {
                // Fallback to standard Math
                exp = (float) Math.exp(array.get(i) - maxVal);
            }
            array.set(i, exp);
            sum += exp;
        }

        // Normalize by sum
        if (sum == 0.0f) {
            // Handle edge case, divide evenly
            float value = 1.0f / (end - start);
            for (int i = start; i < end; i++) {
                array.set(i, value);
            }
        } else {
            // Normal case, divide by sum
            for (int i = start; i < end; i++) {
                array.set(i, array.get(i) / sum);
            }
        }

        return array;
    }

    /** {@link #divideInPlace(FloatArray, int, int, float)} over a heap copy of the scores. */
    public static float[] divideInPlace(float[] array, int start, int end, float value) {
        for (int i = start; i < end; i++) {
            array[i] = array[i] / value;
        }
        return array;
    }

    /**
     * {@link #softmaxInPlace(FloatArray, int, int)} over a heap copy of the scores: the same
     * operations in the same order, so the probabilities are bit-for-bit those of the device array.
     */
    public static float[] softmaxInPlace(float[] array, int start, int end) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = start; i < end; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
            }
        }
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            float exp;
            if (TornadoVMSupport.isTornadoVMEnabled()) {
                exp = TornadoMath.exp(array[i] - maxVal);
            } else {
                exp = (float) Math.exp(array[i] - maxVal);
            }
            array[i] = exp;
            sum += exp;
        }
        if (sum == 0.0f) {
            float value = 1.0f / (end - start);
            for (int i = start; i < end; i++) {
                array[i] = value;
            }
        } else {
            for (int i = start; i < end; i++) {
                array[i] = array[i] / sum;
            }
        }
        return array;
    }

    /**
     * Helper class to check if TornadoVM is enabled. This allows us to decide whether to use
     * TornadoMath or standard Math.
     */
    private static class TornadoVMSupport {
        private static final boolean TORNADO_VM_ENABLED;

        static {
            boolean enabled;
            try {
                // Try to access a TornadoVM-specific class
                Class.forName("uk.ac.manchester.tornado.api.math.TornadoMath");
                // Check for system property
                enabled = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
            } catch (ClassNotFoundException e) {
                enabled = false;
            }
            TORNADO_VM_ENABLED = enabled;
        }

        static boolean isTornadoVMEnabled() {
            return TORNADO_VM_ENABLED;
        }
    }
}

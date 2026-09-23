package org.beehive.jllm.runtime.memory;

import java.util.Set;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * What representation each weight will actually occupy on the device.
 *
 * <p>Asked per tensor, not per model, because a model is not one representation: Qwen3.8-27B holds
 * Q4_0 projections, eight Q4_1 down projections, Q5_K recurrent outputs, a Q6_K vocabulary
 * projection, a Q8_0 MTP projection and F32 norms. A memory prediction built from a single
 * model-wide dtype is wrong for every tensor that is not that dtype.
 *
 * <p>Asked by <b>name</b> as well as representation, because support can differ by role. A family
 * may have a matrix-vector kernel for a representation and no vocabulary-projection kernel for it,
 * and the tensor's name is what distinguishes the two. Most policies will not need the name; it is
 * there so that the ones that do are expressible rather than approximated.
 */
@FunctionalInterface
public interface DeviceRetention {

    /**
     * The representation this tensor will be stored in on the device.
     *
     * @param tensorName the GGUF tensor name, e.g. {@code blk.3.ffn_down.weight}
     * @param source what the file holds it as
     * @return what the device will hold — {@code source} when it is retained
     */
    DataType deviceType(String tensorName, DataType source);

    /** Retains everything in the given set, and reports the rest as promoted to {@code Q8_0}. */
    static DeviceRetention retaining(Set<DataType> nativeTypes) {
        return (name, source) -> {
            if (nativeTypes.contains(source)) {
                return source;
            }
            return switch (source) {
                case Q4_0, Q4_1, Q4_K, Q5_K, Q6_K -> DataType.Q8_0;
                default -> source.narrowedFallback();
            };
        };
    }

    /**
     * The older loading path: every block quantization becomes {@code Q8_0} on the device.
     *
     * <p>Named rather than implicit. A family still on this path costs roughly double for a 4-bit
     * file, and that shows up in its memory plan as the number it really is.
     */
    static DeviceRetention converting() {
        return retaining(Set.of());
    }
}

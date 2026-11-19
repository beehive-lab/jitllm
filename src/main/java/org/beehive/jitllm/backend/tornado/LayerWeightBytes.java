package org.beehive.jitllm.backend.tornado;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.lang.reflect.RecordComponent;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

// @formatter:off
/**
 * How many bytes of device weights each layer of a loaded model holds: what a split across devices
 * with no {@code --tensor-split} balances.
 *
 * <p>Read from the weights themselves, so every family is measured the same way and a family needs
 * no code of its own. A family's per-layer weights are public arrays or lists indexed by layer,
 * either fields of its {@link TornadoWeights} or components of a record it keeps them in (and of
 * the records nested in that). Element {@code l} of each belongs to layer {@code l}; when it is
 * itself a list or an array, as a layer's experts are, all of it does. A tensor reached twice in
 * one layer, through an alias in the base class's slots, counts once.
 *
 * <p>Layers differ by more than a little in some families: glm5next's first three layers are dense
 * and a fourteenth of the size of the mixture-of-experts layers after them, and qwen35's attention
 * and recurrent layers hold different tensors. An equal number of layers per device puts unequal
 * bytes on each.
 */
// @formatter:on
final class LayerWeightBytes {

    private LayerWeightBytes() {}

    /** Bytes per layer, for the first {@code layers} layers. */
    static long[] of(TornadoWeights weights, int layers) {
        long[] bytes = new long[layers];
        for (int layer = 0; layer < layers; layer++) {
            Set<TornadoTensor> seen = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Field field : weights.getClass().getFields()) {
                if (Modifier.isStatic(field.getModifiers())) {
                    continue;
                }
                bytes[layer] += layerBytes(read(field, weights), layer, seen);
            }
        }
        return bytes;
    }

    /** A tensor's bytes on the device; 0 for none. */
    static long bytes(TornadoTensor tensor) {
        return tensor != null && tensor.deviceArray() instanceof TornadoNativeArray array
                ? array.getNumBytesOfSegment()
                : 0;
    }

    /**
     * What layer {@code layer} owns of {@code value}: a per-layer array, list or record of them.
     */
    private static long layerBytes(Object value, int layer, Set<TornadoTensor> seen) {
        if (value instanceof Object[] array) {
            return layer < array.length ? all(array[layer], seen) : 0;
        }
        if (value instanceof List<?> list) {
            return layer < list.size() ? all(list.get(layer), seen) : 0;
        }
        if (value instanceof Record record) {
            long sum = 0;
            for (RecordComponent component : record.getClass().getRecordComponents()) {
                sum += layerBytes(read(component, record), layer, seen);
            }
            return sum;
        }
        return 0;
    }

    /** Every tensor in {@code value}: one tensor, or a list or array of them. */
    private static long all(Object value, Set<TornadoTensor> seen) {
        if (value instanceof TornadoTensor tensor) {
            return seen.add(tensor) ? bytes(tensor) : 0;
        }
        long sum = 0;
        if (value instanceof Object[] array) {
            for (Object o : array) {
                sum += all(o, seen);
            }
        } else if (value instanceof List<?> list) {
            for (Object o : list) {
                sum += all(o, seen);
            }
        }
        return sum;
    }

    private static Object read(Field field, Object owner) {
        try {
            return field.get(owner);
        } catch (IllegalAccessException e) {
            return null;
        }
    }

    private static Object read(RecordComponent component, Record owner) {
        try {
            return component.getAccessor().invoke(owner);
        } catch (IllegalAccessException | InvocationTargetException e) {
            return null;
        }
    }
}

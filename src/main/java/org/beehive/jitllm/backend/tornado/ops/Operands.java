package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.runtime.tensor.TensorRole;

/**
 * The arrays behind the names the operations use: an activation name such as {@code "q"} gives a
 * workspace buffer, and a weight role in a layer gives that layer's weight.
 *
 * <p>Several names can share one buffer: q, k and v can be written side by side into one {@code
 * [q|k|v]} buffer, which all three names then give.
 */
public interface Operands {

    /** The array of activation {@code name}. */
    Object activation(String name);

    /** The array of activation {@code ref}. */
    default Object activation(OperandRef ref) {
        return activation(ref.name());
    }

    /** Layer {@code layer}'s weight with {@code ref}'s role. */
    TornadoTensor weight(OperandRef.Weight ref, int layer);

    /**
     * Operands from a map of activation names to arrays and a lookup of weights by role and layer.
     *
     * @throws IllegalArgumentException if an operation of {@code regions} uses an activation the
     *     map does not have
     */
    static Operands of(
            Map<String, Object> activations,
            BiFunction<TensorRole, Integer, TornadoTensor> weights,
            List<Region> regions) {
        for (Region region : regions) {
            for (Operation op : region.ops()) {
                for (List<OperandRef> refs : List.of(op.inputs(), op.outputs())) {
                    for (OperandRef ref : refs) {
                        if (ref instanceof OperandRef.Activation
                                && !activations.containsKey(ref.name())) {
                            throw new IllegalArgumentException(
                                    "no array for activation "
                                            + ref.name()
                                            + " in layer "
                                            + region.layer());
                        }
                    }
                }
            }
        }
        return new Operands() {
            @Override
            public Object activation(String name) {
                Object array = activations.get(name);
                if (array == null) {
                    throw new IllegalArgumentException("no array for activation " + name);
                }
                return array;
            }

            @Override
            public TornadoTensor weight(OperandRef.Weight ref, int layer) {
                return weights.apply(ref.role(), layer);
            }
        };
    }
}

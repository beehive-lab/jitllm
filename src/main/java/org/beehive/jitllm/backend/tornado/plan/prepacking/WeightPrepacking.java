package org.beehive.jitllm.backend.tornado.plan.prepacking;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanProvider;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;

// @formatter:off
/**
 * Selects which projection weights an execution plan prepacks.
 *
 * <p>Prepacking rearranges the bytes of a Q8_0 or Q4_0 weight from the original GGUF layout into
 * the prepacked layout, which optimized CUDA kernels read. The weight's values and its size stay
 * the same; only the order of its bytes changes.
 *
 * <p>The selection is made once, when the plan is built ({@link #forPlan}), from the kernels the
 * model family has, the device, the plan's execution policy and each weight's shape. System
 * properties can force prepacking on or off.
 *
 * <p>This class only selects. {@link WeightPrepacker} rearranges the selected weights on the GPU,
 * in the plan's own device buffers; the weights on the host keep the original layout. The layer
 * builders, the prepacker and the memory preflight all ask this class, so they agree on which
 * weights are prepacked.
 */
// @formatter:on
public final class WeightPrepacking {

    /**
     * Forces prepacking of Q8_0 weights on ({@code true}) or off ({@code false}); unset, the device
     * and policy decide.
     */
    public static final String Q8_PROPERTY = "jitllm.q8_0.prepack";

    /**
     * Forces prepacking of Q4_0 weights on ({@code true}) or off ({@code false}); unset, the device
     * and policy decide.
     */
    public static final String Q4_PROPERTY = "jitllm.q4_0.prepack";

    /** Rows of the int8 prefill GEMM's block: a batched prefill's width must be a multiple. */
    private static final int GEMM_ROWS = 128;

    /** Selects no weight. */
    public static final WeightPrepacking NONE = new WeightPrepacking(Set.of());

    private final Set<TornadoTensor> prepacked;

    private WeightPrepacking(Set<TornadoTensor> prepacked) {
        this.prepacked = prepacked;
    }

    /**
     * Selects exactly {@code tensors}. Tensors are matched by identity, not equality: two tensors
     * with the same shape and bytes are different weights.
     */
    public static WeightPrepacking of(Collection<? extends TornadoTensor> tensors) {
        if (tensors.isEmpty()) {
            return NONE;
        }
        Set<TornadoTensor> prepacked = Collections.newSetFromMap(new IdentityHashMap<>());
        prepacked.addAll(tensors);
        return new WeightPrepacking(Collections.unmodifiableSet(prepacked));
    }

    /**
     * Selects the projection weights that a plan of {@code family} prepacks under {@code policy}.
     *
     * <p>A weight is selected if it is a layer projection ({@link TornadoWeights#projections}),
     * prepacking is on for its type ({@link #prepackedTypes}), and its shape {@link #fits}.
     */
    public static WeightPrepacking forPlan(
            TornadoPlanProvider family, TornadoWeights weights, ExecutionPolicy policy) {
        Set<DataType> types = prepackedTypes(family, policy);
        if (types.isEmpty()) {
            return NONE;
        }
        List<TornadoTensor> prepacked = new ArrayList<>();
        for (TornadoTensor[] role : weights.projections()) {
            if (role == null) {
                continue;
            }
            for (TornadoTensor w : role) {
                if (w != null && types.contains(w.dataType()) && fits(w.shape())) {
                    prepacked.add(w);
                }
            }
        }
        return of(prepacked);
    }

    /**
     * The weight types a plan of {@code family} prepacks under {@code policy}: those the family's
     * kernels can read in the prepacked layout ({@link TornadoPlanProvider#prepackableTypes}), for
     * which prepacking is on.
     *
     * <p>Needs no loaded weights, so the memory preflight and the native-library default use it
     * before any plan exists.
     */
    public static Set<DataType> prepackedTypes(TornadoPlanProvider family, ExecutionPolicy policy) {
        Set<DataType> types = EnumSet.noneOf(DataType.class);
        for (DataType type : family.prepackableTypes(policy)) {
            if (on(type, policy)) {
                types.add(type);
            }
        }
        return types;
    }

    /**
     * Whether prepacking is on for {@code type} under {@code policy}.
     *
     * <p>When its property ({@link #Q8_PROPERTY}, {@link #Q4_PROPERTY}) is set, the property
     * decides alone: {@code true} skips the device and prefill checks below, which lets tests and
     * experiments force prepacking. Unset, prepacking is on where the device runs the prepacked
     * kernels and the prefill configuration suits them.
     */
    private static boolean on(DataType type, ExecutionPolicy policy) {
        String property =
                switch (type) {
                    case Q8_0 -> Q8_PROPERTY;
                    case Q4_0 -> Q4_PROPERTY;
                    default -> null;
                };
        if (property == null) {
            return false;
        }
        String asked = System.getProperty(property);
        if (asked != null && !asked.isBlank()) {
            return Boolean.parseBoolean(asked);
        }
        return deviceRunsPrepackedKernels() && prefillWidthFits(policy);
    }

    /**
     * Whether the device runs the prepacked kernels: a CUDA device with packed-integer dot products
     * and int8 tensor cores.
     */
    private static boolean deviceRunsPrepackedKernels() {
        var device = TornadoDevices.current();
        var capabilities = device.capabilities();
        return BackendId.CUDA.equals(device.backend())
                && capabilities.supports(DeviceCapability.PACKED_INTEGER_DOT)
                && capabilities.supports(DeviceCapability.INT8_TENSOR_CORE_MMA);
    }

    /**
     * Whether the plan's prefill can read prepacked weights. A plan that does not prefill in
     * batches always can: its prefill runs the decode kernels, which have prepacked versions. A
     * batched prefill reads them only through the int8 GEMM, which needs a batch width that is a
     * multiple of 128.
     */
    static boolean prefillWidthFits(ExecutionPolicy policy) {
        return ExecutionMode.of(policy) != ExecutionMode.BATCH_PREFILL_DECODE
                || policy.prefillBatchSize() % GEMM_ROWS == 0;
    }

    /**
     * Whether a weight of {@code shape} can be prepacked: the prepacked layout groups 128 rows by
     * 64 inputs, so the weight must be two-dimensional with rows a multiple of 128 and inputs a
     * multiple of 64. GGUF lists the dimensions inputs first, then rows.
     */
    public static boolean fits(Shape shape) {
        return shape.rank() == 2 && shape.dimension(1) % 128 == 0 && shape.dimension(0) % 64 == 0;
    }

    /**
     * Creates the exception thrown by a path that has no kernel for prepacked weights, when it is
     * given one. Such a path reads the original layout, so running it would compute garbage.
     */
    public static UnsupportedOperationException noPrepackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a prepacked projection, and this path has no kernel for prepacked"
                        + " weights. Run with -Djitllm.q8_0.prepack=false"
                        + " -Djitllm.q4_0.prepack=false.");
    }

    /** Whether {@code w} is selected. */
    public boolean prepacked(TornadoTensor w) {
        return w != null && prepacked.contains(w);
    }

    /** Whether {@code w} is selected and Q4_0. */
    public boolean prepackedQ4(TornadoTensor w) {
        return prepacked(w) && w.dataType() == DataType.Q4_0;
    }

    /** Whether {@code w} is selected and Q8_0. */
    public boolean prepackedQ8(TornadoTensor w) {
        return prepacked(w) && w.dataType() == DataType.Q8_0;
    }
}

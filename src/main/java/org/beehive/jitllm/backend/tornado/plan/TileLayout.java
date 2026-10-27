package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;

// @formatter:off
/**
 * Which projection weights one execution plan holds on the device in the packed tile layout: 128
 * rows by 64 inputs, in the order the int8 prefill GEMM and the packed decode kernels read them.
 *
 * <p>Decided once per plan, when it is built, from the plan's execution policy, the device and the
 * representations the family's kernels read as tiles ({@link TornadoPlanProvider#tileTypes}). The
 * loaded tensors keep the file's bytes either way: {@link PackedRepack} rewrites a tiled weight on
 * the device when the plan starts.
 *
 * <p>Three readers, one answer: the layer builders choose a weight's kernel and its grid from
 * {@link #tiled}, {@link PackedRepack} rewrites exactly the tiled weights, and the memory check
 * applies {@link #tiledTypes} to the file's descriptors.
 */
// @formatter:on
public final class TileLayout {

    /** No weight is tiled. */
    public static final TileLayout NONE = new TileLayout(Set.of());

    private final Set<TornadoTensor> tiled;

    private TileLayout(Set<TornadoTensor> tiled) {
        this.tiled = tiled;
    }

    /** The layout that tiles exactly {@code tensors}, compared by identity. */
    public static TileLayout of(Collection<? extends TornadoTensor> tensors) {
        if (tensors.isEmpty()) {
            return NONE;
        }
        Set<TornadoTensor> tiled = Collections.newSetFromMap(new IdentityHashMap<>());
        tiled.addAll(tensors);
        return new TileLayout(Collections.unmodifiableSet(tiled));
    }

    /**
     * The layout a plan of {@code family} builds over {@code weights} under {@code policy}: every
     * projection of a type in {@link #tiledTypes} whose shape divides into tiles.
     */
    public static TileLayout forPlan(
            TornadoPlanProvider family, TornadoWeights weights, ExecutionPolicy policy) {
        Set<DataType> types = tiledTypes(family, policy);
        if (types.isEmpty()) {
            return NONE;
        }
        List<TornadoTensor> tiled = new ArrayList<>();
        for (TornadoTensor[] role : weights.projections()) {
            if (role == null) {
                continue;
            }
            for (TornadoTensor w : role) {
                if (w != null && types.contains(w.dataType()) && fits(w.shape())) {
                    tiled.add(w);
                }
            }
        }
        return of(tiled);
    }

    /**
     * The representations {@code family} holds in tiles under {@code policy} on this device: those
     * its kernels read as tiles, where packing is on for them ({@link PackedWeights#tiles}).
     */
    public static Set<DataType> tiledTypes(TornadoPlanProvider family, ExecutionPolicy policy) {
        Set<DataType> types = EnumSet.noneOf(DataType.class);
        for (DataType type : family.tileTypes(policy)) {
            if (PackedWeights.tiles(type, policy)) {
                types.add(type);
            }
        }
        return types;
    }

    /**
     * Whether a weight of {@code shape} divides into tiles: two dimensions, in GGUF order (inputs,
     * then rows), with rows a multiple of 128 and inputs a multiple of 64.
     */
    public static boolean fits(Shape shape) {
        return shape.rank() == 2 && shape.dimension(1) % 128 == 0 && shape.dimension(0) % 64 == 0;
    }

    /** Whether the plan holds {@code w} in tiles. */
    public boolean tiled(TornadoTensor w) {
        return w != null && tiled.contains(w);
    }

    /** Whether the plan holds {@code w} as Q4_0 tiles. */
    public boolean tiledQ4(TornadoTensor w) {
        return tiled(w) && w.dataType() == DataType.Q4_0;
    }

    /** Whether the plan holds {@code w} as Q8_0 tiles. */
    public boolean tiledQ8(TornadoTensor w) {
        return tiled(w) && w.dataType() == DataType.Q8_0;
    }

    /** Whether the plan tiles any weight. */
    public boolean isEmpty() {
        return tiled.isEmpty();
    }
}

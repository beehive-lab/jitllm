package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import java.util.function.BooleanSupplier;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.layers.FP16Projections;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;

// @formatter:off
/**
 * Whether a batched prefill runs on Apple MLX, and in which weight representation: the one decision
 * the memory preflight, the batched-prefill layers and the decode layers that bind their weights
 * all read.
 *
 * <p>Under native libraries ({@code --with-native-libraries}) on the Metal backend, a family whose
 * provider lists its weights for Metal ({@code TornadoPlanProvider.nativeLibraryWeights}) runs
 * every chunk's attention as MLX's fused causal attention and its four projections (QKV, output,
 * gate/up and down) as MLX GEMMs, launched by TornadoVM on its own command queue and buffers.
 * Attention and projections go together: there is no per-operation switch, as there is none on the
 * cuBLAS path.
 *
 * <p><b>One answer, from immutable inputs.</b> {@link #resolve} reads only the session's policy,
 * the model's weight type and its configuration. The layers that build the MLX tasks construct them
 * from its answer and refuse a tensor that does not match it rather than falling back to the JIT
 * kernels; the decode layers ask the same question to learn which model weights the prefill graphs
 * do not bind ({@code MlxPrefillProjections.notBound}); and the preflight sizes the same buffers
 * ({@link #workspace}). None of them can drift from the others, because none of them decides.
 *
 * <p><b>The weight representation is the residency of decoded projections.</b> F16 weights are
 * multiplied as they are, with Q/K/V and gate/up stacked into one GEMM each. Q8_0 and Q4_0 weights
 * are either decoded once into resident FP16 copies ({@link Projections#FP16_DECODED}) or repacked
 * exactly into MLX's affine format, which the quantized GEMM decodes again for each chunk ({@link
 * Projections#AFFINE}). That is the choice {@link ExecutionPolicy#decodedProjections()} already
 * describes for the tensor-core prefill, so it is resolved the same way: pinned by {@code
 * -Djitllm.prefill.decodedProjections=resident|per-chunk}, otherwise settled against the device
 * budget at load ({@code MemoryPreflight.resolveDecodedProjections}), and without a budget kept
 * resident while the FP16 copies fit {@link FP16Projections#RESIDENT_BYTES}.
 *
 * <p>Needs TornadoVM's {@code tornado.mlx} module and MLX's {@code mlx.metallib} ({@code brew
 * install mlx}) on an M4-class GPU or newer. Resolved lazily: nothing here loads the Objective-C
 * runtime or the kernel library unless a Metal session asks for native libraries.
 *
 * @param weights the model's projection weight type: F16, Q8_0 or Q4_0
 * @param projections the representation the MLX GEMMs read
 */
// @formatter:on
public record MlxPrefill(DataType weights, Projections projections) {

    /** The representation the MLX projection GEMMs read. */
    public enum Projections {
        /** The model's F16 weights; split Q/K/V and gate/up are stacked into one copy each. */
        FP16,
        /** Q8_0 or Q4_0 weights decoded once into resident FP16 copies. */
        FP16_DECODED,
        /** Q8_0 or Q4_0 weights repacked exactly into MLX's affine format, groups of 32. */
        AFFINE
    }

    /** The quantization group of MLX's affine format: one Q8_0 or Q4_0 block. */
    public static final int GROUP = 32;

    public MlxPrefill {
        if (weights != DataType.F16 && weights != DataType.Q8_0 && weights != DataType.Q4_0) {
            throw new IllegalArgumentException(
                    "MLX prefill reads F16, Q8_0 or Q4_0, not " + weights);
        }
        if ((weights == DataType.F16) != (projections == Projections.FP16)) {
            throw new IllegalArgumentException(projections + " is not a form of " + weights);
        }
    }

    /**
     * The MLX prefill a session running {@code policy} builds over a model of {@code weights}
     * weights and configuration {@code config}, or empty when its batched prefill runs the JIT
     * kernels.
     *
     * <p>Present exactly when native libraries are asked for, the prefill is batched, the device is
     * a Metal one, MLX's kernel library loads and the weights are F16, Q8_0 or Q4_0. Whether the
     * family implements those weights on Metal is the admission's question ({@code
     * NativeLibrarySupport}), asked before any plan is built.
     */
    public static Optional<MlxPrefill> resolve(
            ExecutionPolicy policy, DataType weights, Configuration config) {
        if (!policy.nativeLibraries()
                || ExecutionMode.of(policy) != ExecutionMode.BATCH_PREFILL_DECODE
                || !BackendId.METAL.equals(TornadoDevices.target().backend())
                || (weights != DataType.F16 && weights != DataType.Q8_0 && weights != DataType.Q4_0)
                || !available()) {
            return Optional.empty();
        }
        Projections projections;
        if (weights == DataType.F16) {
            projections = Projections.FP16;
        } else {
            projections =
                    FP16Projections.resident(policy, projectionWeights(config))
                            ? Projections.FP16_DECODED
                            : Projections.AFFINE;
        }
        return Optional.of(new MlxPrefill(weights, projections));
    }

    /** Whether MLX's kernel library can be reached from this process at all. */
    public static boolean available() {
        return Probe.MLX;
    }

    /**
     * The projection weights of every layer: QKV, output, gate/up and down. What a resident FP16
     * copy holds, and what the affine repack is a form of.
     */
    public static long projectionWeights(Configuration config) {
        return config.projectionWeights();
    }

    // @formatter:off
    /**
     * What one session's MLX prefill allocates on the device besides the model's own weights, sized
     * from the same arithmetic {@code MlxPrefillProjections} and {@code MlxPrefillAttention}
     * allocate by.
     *
     * <ul>
     *   <li>the projections' MLX form: for F16, the stacked Q/K/V and gate/up copies of a family
     *       that keeps them split (a fused tensor is multiplied as it is, and bound, not copied);
     *       for Q8_0 and Q4_0, the resident FP16 copies or the affine repack, whichever the
     *       residency resolves to;
     *   <li>the FP16 staging every layer graph shares: the GEMM operands and products, and the FP32
     *       per-row scales that keep the output and down projections inside FP16;
     *   <li>the attention's contiguous FP16 query, output, and key and value buffers sized for the
     *       whole context plus one chunk.
     * </ul>
     *
     * <p>The model's projection weights are not charged here: the prefill graphs no longer bind the
     * ones they replaced, and the decode graphs upload those themselves, so the per-layer weights
     * stay counted once ({@code weightBindingFamilies}).
     */
    // @formatter:on
    public static BatchPrefillWorkspace workspace(
            DataType weights,
            Configuration config,
            int batchSize,
            boolean fusedQkv,
            boolean fusedGateUp) {
        long dim = config.dim();
        long qDim = (long) config.numberOfHeads() * config.headSize();
        long kvDim = config.kvDim();
        long hidden = config.hiddenDim();
        long rows = batchSize;
        long staging =
                Short.BYTES * rows * (dim + (qDim + 2 * kvDim) + qDim + dim + 3 * hidden)
                        + Float.BYTES * 2 * rows;
        long attention =
                Short.BYTES
                        * (2 * rows * qDim + 2 * ((long) config.contextLength() + rows) * kvDim);
        long fixed = staging + attention;
        if (weights == DataType.F16) {
            long stacked =
                    (fusedQkv ? 0 : (qDim + 2 * kvDim) * dim)
                            + (fusedGateUp ? 0 : 2 * hidden * dim);
            return BatchPrefillWorkspace.fixed(
                    fixed + Short.BYTES * stacked * config.numberOfLayers());
        }
        long projections = projectionWeights(config);
        return new BatchPrefillWorkspace(
                projections,
                fixed + FP16Projections.residentBytes(projections),
                fixed + affineBytes(weights, projections));
    }

    /**
     * The bytes of {@code projections} weights repacked into MLX's affine format: the packed
     * values, and an FP16 scale and bias per group of 32.
     */
    public static long affineBytes(DataType weights, long projections) {
        int bits = weights == DataType.Q8_0 ? 8 : 4;
        long groups = projections / GROUP;
        return projections * bits / 8 + 2L * Short.BYTES * groups;
    }

    /** The library probe, in its own holder so that asking about the backend loads nothing. */
    private static final class Probe {

        private static final boolean MLX = probe(MlxLibraryProvider::isAvailable);

        private static boolean probe(BooleanSupplier available) {
            try {
                return available.getAsBoolean();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }
    }
}

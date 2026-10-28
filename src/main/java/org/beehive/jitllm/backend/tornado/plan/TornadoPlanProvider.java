package org.beehive.jitllm.backend.tornado.plan;

import java.util.EnumSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.layers.BatchDecodeLayers;
import org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.jitllm.backend.tornado.memory.BatchPrefillWorkspace;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * One architecture's <b>legacy plan components</b>, as this backend builds them.
 *
 * <p>The twin of {@code TornadoLoweringProvider}, for the path that is still the default. The
 * lowering answers "build this program"; this answers "build the plan components this family has
 * always used". Both are registered per identity and discovered, and neither is listed centrally.
 *
 * <p>The concrete {@code State} cast that {@code ForwardPlanFactory} used to do per family moves in
 * here, which is where a family's own file can do it safely. That cast is the reason the factory's
 * switch could not simply be deleted (Rule 15's note on {@code ForwardPlanFactory$1}).
 */
public interface TornadoPlanProvider {

    /** The identity these components implement. Two providers claiming one identity is an error. */
    ArchitectureId architecture();

    /**
     * This family's layers for a batched decode step over a shared key/value cache (continuous
     * batching), or empty when it has none.
     */
    default Optional<BatchDecodeLayers> batchDecodeLayers(
            State state, Model model, BatchDecode decode) {
        return Optional.empty();
    }

    /**
     * Whether this family's logits can sample greedily on the device, and its decode loop read the
     * token from there, for weights in {@code weights}.
     */
    default boolean deviceSampling(DataType weights) {
        return false;
    }

    /**
     * The <b>model-wide</b> weight representations this provider builds plan components for.
     *
     * <p>Provider admission, and nothing else. It answers "given a model whose weights report this
     * representation, can this family build a plan for it", which is the question {@code
     * ForwardPlanFactory} asks. It is deliberately <b>not</b> the set of representations the
     * family's kernels can read: a heterogeneous model reports one representation and holds
     * several, and answering both questions with one set would make one of the two answers false.
     * See {@link #nativeTensorTypes()} for the other half.
     */
    Set<DataType> supportedDataTypes();

    /**
     * The representations this family's tasks read <b>per tensor</b>, in the file's own layout.
     *
     * <p>Read by the memory preflight, which predicts each tensor at the representation it will
     * actually occupy on the device. A family listing {@code Q4_0} has tasks that decode Q4_0
     * blocks, so a Q4_0 tensor stays 4.5 bits per weight instead of becoming 8.5.
     *
     * <p>Defaults to {@link #supportedDataTypes()}, which is right for every family whose model is
     * one representation throughout: there the two questions have one answer. A family with a
     * genuinely mixed model overrides it — {@code qwen35} admits plans for its projections'
     * representation while its tasks also decode Q4_1 down projections, Q5_K recurrent outputs, a
     * Q6_K vocabulary projection and F32 recurrent parameters.
     *
     * <p>Listing a representation here is a claim that some task decodes it natively. It is not a
     * claim that every operation can: an operation with no kernel for a representation must fail by
     * name at plan construction, not be repaired by conversion.
     */
    default Set<DataType> nativeTensorTypes() {
        return supportedDataTypes();
    }

    /**
     * What representation each tensor of a file will occupy on the device, as this family's loader
     * decides it. Read by the memory preflight, from the file's descriptors.
     *
     * <p>Defaults to keeping {@link #nativeTensorTypes()} and materializing the rest as Q8_0, which
     * is right for a family whose loader decides per tensor. A family whose loader decides for the
     * whole model overrides it with that same rule.
     *
     * @param sourceTypes each tensor's representation in the file, by tensor name
     * @param layers the number of transformer blocks
     * @param backend the device's backend
     */
    default DeviceRetention deviceRetention(
            Map<String, DataType> sourceTypes, int layers, BackendId backend) {
        return DeviceRetention.retaining(nativeTensorTypes());
    }

    /** The execution modes this family has plans for. */
    Set<ExecutionMode> supportedModes();

    /**
     * Builds the components for a representation this provider supports.
     *
     * @throws IllegalArgumentException if the state is not this family's, which is the cast that
     *     used to live in the central factory
     */
    SingleTokenForwardPlanComponents components(DataType weights, State state, Model model);

    /**
     * Inner stage boundaries of a model split across devices fall on multiples of this, for a
     * family whose graphs group layers.
     */
    default int stageLayerAlignment() {
        return 1;
    }

    /**
     * Whether a stage's key/value cache holds the stage's layers only, which layers built with a
     * range address relative to their first. A family whose layers index their caches and recurrent
     * state by absolute layer answers false: each stage then keeps the whole model's layout and
     * uses its own layers' part.
     */
    default boolean stageCacheHoldsOnlyItsLayers() {
        return true;
    }

    /**
     * A state for one stage of a model split across devices, holding layers {@code [firstLayer,
     * endLayer)}, with the session's storage options and execution policy: see {@link
     * #stageCacheHoldsOnlyItsLayers}.
     */
    default State stageState(
            Model model, State session, int firstLayer, int endLayer, int prefillBatchSize) {
        State stage =
                State.withStorageOptions(
                        session.storageOptions(),
                        () ->
                                State.withPrefillBatchSize(
                                        prefillBatchSize,
                                        () ->
                                                stageCacheHoldsOnlyItsLayers()
                                                        ? State.withKeyValueLayers(
                                                                endLayer - firstLayer,
                                                                model::createNewState)
                                                        : model.createNewState()));
        stage.resolveExecutionPolicy(session.executionPolicy());
        return stage;
    }

    // @formatter:off
    /**
     * Whether a load keeps every tensor as the file stores it, rather than converting Q4_0, Q4_1
     * and the K-quants to Q8_0: on every device backend (CUDA, Metal and OpenCL), where a migrated
     * family's every path reads each tensor by its own type (the single-token projections, the
     * batched prefill, the tensor-core prefill and the embedding row).
     *
     * <p>OpenCL came last: the per-type float kernels, the tiled batch projections included,
     * compile and match their host references there, and no migrated family's parity there is worse
     * than it was on the Q8_0 copies.
     *
     * <p>The one rule for a migrated family's loader and its memory preflight ({@link
     * #deviceRetention}), so the two cannot disagree about what a tensor becomes.
     */
    // @formatter:on
    static boolean retainsNative(BackendId backend) {
        return BackendId.CUDA.equals(backend)
                || BackendId.METAL.equals(backend)
                || BackendId.OPENCL.equals(backend);
    }

    /**
     * What a migrated family's load keeps on a {@code backend} device: {@code kept} everywhere, and
     * every quantized type besides where {@link #retainsNative} holds.
     */
    static DeviceRetention nativeRetention(Set<DataType> kept, BackendId backend) {
        Set<DataType> retained = EnumSet.copyOf(kept);
        if (retainsNative(backend)) {
            retained.addAll(TornadoSupportSets.NATIVE_QUANTIZED);
        }
        return DeviceRetention.retaining(retained);
    }

    // ── What this family's layers can build ───────────────────────────────────

    /**
     * The weight representations whose batched prefill has a native-library path ({@code
     * --with-native-libraries}) on a {@code backend} device: on CUDA, projections through cuBLAS
     * rather than the JIT kernels; on Metal, attention and projections through Apple MLX ({@code
     * MlxPrefill}). Empty where the family has none, which admission refuses by name.
     */
    default Set<DataType> nativeLibraryWeights(BackendId backend) {
        return Set.of();
    }

    /**
     * Why this family's plans cannot run on a {@code backend} device at all — in any mode, with
     * either key/value cache — or empty when they can. Asked for every plan on a device backend,
     * and for each stage's device of a model split across devices.
     */
    default Optional<String> deviceUnsupported(BackendId backend) {
        return Optional.empty();
    }

    /**
     * Why this family's plan for {@code weights} cannot be split by layers into {@code stages}
     * stages ({@code --devices}) under {@code policy}, or empty when it can. Asked before any stage
     * or transport exists, so it answers what building the stages would: a family answers empty
     * only where its components are {@code StagedForwardPlanComponents} and, for a batched policy,
     * build the batched layers of a stage. A family is not split unless it says so.
     */
    default Optional<String> splitUnsupported(
            DataType weights, ExecutionPolicy policy, int stages) {
        return Optional.of(
                "the " + architecture() + " plan cannot be split across devices; drop --devices");
    }

    /**
     * The representations this family's kernels read prepacked under {@code policy}, or none. A
     * plan prepacks a projection of one of them where prepacking is on and its shape fits ({@link
     * WeightPrepacking}).
     */
    default Set<DataType> prepackableTypes(ExecutionPolicy policy) {
        return Set.of();
    }

    /**
     * What one session's batched prefill of this family allocates for its {@code weights}
     * representation beyond the staging every family has, as the layer class it would build sizes
     * it — or empty when that class adds nothing the configuration does not already account.
     */
    default Optional<BatchPrefillWorkspace> batchPrefillWorkspace(
            DataType weights,
            Configuration config,
            int batchSize,
            ExecutionPolicy policy,
            boolean fp16Cache) {
        return Optional.empty();
    }

    /**
     * Why this family's batched prefill cannot be built for {@code c}, with either key/value cache,
     * or empty when it can. Asked only for a batched prefill on a device backend.
     */
    default Optional<String> batchPrefillUnsupported(Combination c) {
        return Optional.empty();
    }

    /**
     * What a batched prefill of this family needs from {@code policy} that it does not ask for, or
     * empty when nothing. Asked only for a batched prefill on a device backend.
     */
    default Optional<String> batchPrefillNeeds(Combination c, ExecutionPolicy policy) {
        return Optional.empty();
    }

    /**
     * Why this family's layers cannot store and read the key/value cache in half precision for
     * {@code c}, or empty when they can. Asked on a backend where the FP16 kernels were verified,
     * once the plan is known to build. A family keeps an FP32 cache unless it says otherwise.
     */
    default Optional<String> fp16KeyValueUnsupported(Combination c) {
        return Optional.of("the " + c.architecture() + " layers keep an FP32 cache");
    }

    /**
     * {@link #fp16KeyValueUnsupported} on Metal, once the plan is known to build. A family keeps an
     * FP32 cache there unless it says otherwise.
     */
    default Optional<String> metalFp16KeyValueUnsupported(Combination c) {
        return Optional.of("the " + c.architecture() + " layers keep an FP32 cache");
    }

    /**
     * Whether this family's batched prefill is the default on Metal for {@code weights}: its
     * SIMD-group kernels are tuned there. A family prefills one token at a time on Metal unless it
     * says otherwise.
     */
    default boolean metalBatchedPrefillDefault(DataType weights) {
        return false;
    }
}

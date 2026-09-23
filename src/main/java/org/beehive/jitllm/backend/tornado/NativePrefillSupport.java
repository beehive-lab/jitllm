package org.beehive.jitllm.backend.tornado;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BooleanSupplier;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.provider.CuBlasLibraryProvider;
import uk.ac.manchester.tornado.cudnn.CuDnn;
import uk.ac.manchester.tornado.cudnn.provider.CuDnnLibraryProvider;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.library.LibraryRegistry;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryContext;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider;

/**
 * Whether this host can run the batched prefill pass on the vendor libraries, and the one place
 * that decides it.
 *
 * <p>Two separate questions, because they are answered by two different libraries and either can be
 * absent on its own:
 *
 * <ul>
 *   <li>{@link #nativeProjections()} — cuBLAS is reachable, so the four prefill projections (QKV,
 *       gate/up, output and down) can be single FP16 GEMMs over stacked weights.
 *   <li>{@link #nativeAttention(SdpaShape)} — cuDNN's fused scaled-dot-product attention is
 *       reachable <b>and usable for the shape this session will ask it for</b>, so the first
 *       prefill chunk's attention can be one fused call.
 * </ul>
 *
 * <p><b>Both are gated on the tensor-core backend first</b>, and that is not a convenience. The
 * tensor-core gate is the same predicate {@code Qwen3FP16PlanComponents} uses to decide whether the
 * MMA batch-prefill family is the one being built at all, and the decode layer graphs ask {@link
 * #nativeProjections()} to work out which weights the prefill graphs stopped reading. If the two
 * sides could ever disagree, the decode graphs would consume a buffer that was never uploaded and
 * read zeros — silently, with no diagnostic.
 *
 * <p><b>Resolved lazily.</b> Nothing here runs unless something asks, and what asks is the
 * tensor-core batch-prefill planner and the decode family that pairs with it. A CPU run, an OpenCL
 * run, a Metal run or a single-token plan never reaches this class, so no optional native library
 * is dlopened on their behalf. The tensor-core check is evaluated first and short-circuits for the
 * same reason.
 *
 * <p><b>{@code -Djitllm.prefill.native=false} turns both off.</b> That is a supported operational
 * control, not an experiment: it is the escape hatch for a host whose vendor libraries load but
 * misbehave, and it is what the weight-handoff regression test flips to compare the native and JIT
 * paths in one process. There is deliberately no per-operation property — a configuration where
 * only some of these are on has never been validated end to end.
 */
public final class NativePrefillSupport {

    /**
     * The opt-in, {@code -Djitllm.nativeLibraries=true} ({@code --with-native-libraries}); the legacy
     * {@code -Djitllm.prefill.native} is honoured when set. Resolved into {@link
     * ExecutionPolicy#nativeLibraries()}, which is what the plan reads.
     */
    public static final String PROPERTY =
            org.beehive.jitllm.runtime.policy.ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY;

    /**
     * The execution-plan id the capability probe borrows.
     *
     * <p>{@code TornadoExecutionPlan} hands out ids from an {@code AtomicLong} that starts at zero
     * and only increments, so a negative id cannot collide with a real plan's — which matters,
     * because the probe destroys every library context registered under its id when it finishes.
     */
    private static final long PROBE_EXECUTION_PLAN_ID = Long.MIN_VALUE;

    private NativePrefillSupport() {}

    // @formatter:off
    /**
     * The shape and mask a session will ask cuDNN's fused attention for.
     *
     * <p>Every field reaches the plan cache key on the TornadoVM side and the graph the frontend
     * finalizes, so support is a property of the whole tuple and not of the library alone: cuDNN's
     * fused SDPA wants Ampere or newer, a head dimension that is a multiple of 8 and at most 256,
     * and it may still decline a particular combination. Asking for the exact tuple is the only way
     * to find out.
     */
    // @formatter:on
    public record SdpaShape(
            int batch, int heads, int seqQ, int seqKv, int headDim, float scale, boolean causal) {}

    /**
     * How the probe answers. Package-private and swappable so a test can simulate a host whose
     * fused SDPA is absent or present-but-unusable, which no property and no fixture can otherwise
     * produce on a machine whose toolkit supports it.
     */
    interface SdpaProbe {
        boolean usable(SdpaShape shape);
    }

    private static volatile SdpaProbe sdpaProbe = NativePrefillSupport::probeSdpaThroughTornado;

    /** Answers already established, so a second plan for the same shape does not rebuild a plan. */
    private static final Map<SdpaShape, Boolean> SDPA_CACHE = new ConcurrentHashMap<>();

    /** Installs a probe for a test; returns the previous one so the test can restore it. */
    static SdpaProbe setSdpaProbeForTesting(SdpaProbe probe) {
        SdpaProbe previous = sdpaProbe;
        sdpaProbe = probe;
        SDPA_CACHE.clear();
        return previous;
    }

    /**
     * Whether the four batch-prefill projections run as cuBLAS GEMMs over stacked weights, for a
     * session running {@code policy}.
     *
     * <p><b>Both sides of the weight hand-off ask this with the same policy.</b> The batch-prefill
     * graphs and the decode graphs that consume their weights are built in one pass; an answer that
     * differed between them would leave the decode side binding buffers the prefill side never
     * uploaded — the silent-zeros defect the weight-handoff regression tests exist to catch. The
     * policy is fixed for a plan's life, which is what makes the two answers agree.
     */
    public static boolean nativeProjections(
            org.beehive.jitllm.runtime.policy.ExecutionPolicy policy) {
        return policy.nativeLibraries()
                && TensorCoreSupport.isTensorCoreCapableBackend()
                && Probe.CUBLAS;
    }

    /** {@link #nativeProjections(ExecutionPolicy)} for the policy the properties resolve to. */
    public static boolean nativeProjections() {
        return nativeProjections(
                org.beehive.jitllm.runtime.policy.ExecutionPolicy.fromSystemProperties());
    }

    /** Whether cuBLAS can be reached from this process at all. */
    public static boolean cublasAvailable() {
        return Probe.CUBLAS;
    }

    // @formatter:off
    /**
     * Whether the first prefill chunk of a sequence can run its attention as one fused cuDNN call
     * at {@code shape}.
     *
     * <p>Implies {@link #nativeProjections()}: the two have only ever been measured and validated
     * together, and the fallback family that covers the later chunks is built from the primary's
     * stacked weights, which only exist when the projections are native. A {@code false} here
     * therefore <b>keeps the native projections</b> and selects the batched JIT paged attention for
     * every chunk, rather than giving up the whole path.
     *
     * <p>The answer comes from building a real plan for this exact shape, on the device this
     * session will run on, and throwing it away — see {@link #probeSdpaThroughTornado}. Nothing
     * cheaper distinguishes the three ways this can be unavailable: an absent shim, a shim that
     * loads but whose entry points are the CUDA-11 stubs that report unavailability by returning a
     * null plan, and a real implementation that declines this device or this tuple.
     */
    // @formatter:on
    public static boolean nativeAttention(
            org.beehive.jitllm.runtime.policy.ExecutionPolicy policy, SdpaShape shape) {
        if (!nativeProjections(policy) || !Probe.CUDNN) {
            return false;
        }
        return SDPA_CACHE.computeIfAbsent(shape, s -> sdpaProbe.usable(s));
    }

    // @formatter:off
    /**
     * Builds and immediately destroys a fused-SDPA plan for {@code shape}, through TornadoVM's own
     * provider.
     *
     * <p><b>Through the provider, not around it.</b> The fused path needs {@code libtornado-cudnn},
     * a JNI library, and a JNI library may be loaded by exactly one class loader — which has to be
     * TornadoVM's, because TornadoVM is what will call into it during execution. So this borrows
     * the provider's own {@code prepare} hook, the same call the real execution makes before
     * CUDA-graph capture, rather than loading anything itself.
     *
     * <p><b>What it allocates and how it is released.</b> {@code getOrCreateContext} creates a
     * cuDNN handle bound to a stream, and {@code prepare} finalizes an execution plan and grows a
     * device workspace for it. All three belong to the probe's execution-plan id, and {@code
     * destroyContexts} releases all three whether the preparation succeeded or threw — which it
     * must, because the plan build is exactly the step that can fail after the handle exists.
     *
     * <p><b>What it deliberately leaves.</b> Asking the device for the probe id's native stream
     * creates one CUDA command queue, and the only API that would remove it, {@code
     * TornadoDeviceContext.reset}, also unpins every host segment the device owns and flips a
     * device-wide flag. Tearing down a shared device to tidy one stream would be a worse defect
     * than the stream. One stream, once per process, is the cost.
     *
     * <p>The catch is broad on purpose <b>here and only here</b>: this is a probe, and every way
     * the library can decline is a legitimate answer of "no". No execution-path exception is
     * swallowed anywhere by this class.
     */
    // @formatter:on
    private static boolean probeSdpaThroughTornado(SdpaShape shape) {
        TornadoXPUDevice device;
        TornadoLibraryProvider provider;
        try {
            device =
                    (TornadoXPUDevice)
                            TornadoRuntimeProvider.getTornadoRuntime()
                                    .getBackend(0)
                                    .getDefaultDevice();
            provider = LibraryRegistry.findProvider(CuDnn.LIBRARY_NAME, device);
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
        // prepare() reads only the trailing descriptor arguments -- the extents, the scale and the
        // mask -- and never touches the tensors, so these are placeholders that allocate nothing
        // on the device.
        HalfFloatArray unused = new HalfFloatArray(1);
        LibraryTaskDescriptor descriptor =
                CuDnn.sdpaForward(
                        unused,
                        unused,
                        unused,
                        unused,
                        shape.batch(),
                        shape.heads(),
                        shape.seqQ(),
                        shape.seqKv(),
                        shape.headDim(),
                        shape.scale(),
                        shape.causal());
        boolean usable;
        try {
            LibraryContext context =
                    LibraryRegistry.getOrCreateContext(provider, device, PROBE_EXECUTION_PLAN_ID);
            provider.prepare(descriptor, context);
            usable = true;
        } catch (RuntimeException | LinkageError e) {
            usable = false;
        } finally {
            try {
                LibraryRegistry.destroyContexts(PROBE_EXECUTION_PLAN_ID);
            } catch (RuntimeException | LinkageError e) {
                // Nothing further to release, and a teardown failure is not evidence about
                // capability, so the answer already established stands.
            }
        }
        return usable;
    }

    /**
     * A one-line summary of what <b>this session</b> resolved to, for the plan's startup banner.
     *
     * @param fp16KeyValueCache whether this session holds its key/value cache in half precision,
     *     which the fused attention's adapters require
     * @param shape the attention shape this session would ask for
     */
    public static String describe(
            org.beehive.jitllm.runtime.policy.ExecutionPolicy policy,
            boolean fp16KeyValueCache,
            SdpaShape shape) {
        if (!policy.nativeLibraries()) {
            return "off (JIT kernels; --with-native-libraries to try cuBLAS/cuDNN)";
        }
        if (!TensorCoreSupport.isTensorCoreCapableBackend()) {
            return "off (no tensor-core backend)";
        }
        if (!Probe.CUBLAS) {
            return "off (cuBLAS not reachable)";
        }
        if (!fp16KeyValueCache) {
            return "cuBLAS projections; JIT attention (FP32 key/value cache)";
        }
        if (!Probe.CUDNN) {
            return "cuBLAS projections; JIT attention (cuDNN not reachable)";
        }
        return nativeAttention(policy, shape)
                ? "cuBLAS projections + cuDNN first-chunk attention"
                : "cuBLAS projections; JIT attention (cuDNN fused SDPA unusable at "
                        + shape.heads()
                        + " heads x "
                        + shape.seqQ()
                        + " tokens x "
                        + shape.headDim()
                        + ")";
    }

    /**
     * The library probes, in their own holder so that the class initializer above — which every
     * caller of {@link #nativeProjections()} triggers — does not dlopen anything. Only a caller
     * that gets past the tensor-core gate reaches these fields.
     */
    private static final class Probe {

        /** cuBLAS answers for itself: the provider dlopens the library and reports the result. */
        private static final boolean CUBLAS = probe(CuBlasLibraryProvider::isAvailable);

        /**
         * Whether cuDNN itself is reachable. Necessary and <b>not</b> sufficient for the fused
         * attention, which additionally needs a shim that may be absent, stubbed or unusable for
         * the shape — see {@link #nativeAttention(SdpaShape)}.
         */
        private static final boolean CUDNN = probe(CuDnnLibraryProvider::isAvailable);

        private static boolean probe(BooleanSupplier available) {
            try {
                return available.getAsBoolean();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }
    }
}

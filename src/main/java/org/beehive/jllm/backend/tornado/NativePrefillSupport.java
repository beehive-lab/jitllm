package org.beehive.jllm.backend.tornado;

import uk.ac.manchester.tornado.cublas.provider.CuBlasLibraryProvider;
import uk.ac.manchester.tornado.cudnn.provider.CuDnnLibraryProvider;

/**
 * Whether this host can run the batched prefill pass on the vendor libraries, and the one place
 * that decides it.
 *
 * <p>Two separate questions, because they are answered by two different libraries and either can be
 * absent on its own:
 *
 * <ul>
 *   <li>{@link #nativeProjections()} — cuBLAS is reachable, so the four prefill projections (QKV,
 *       gate/up, output and down) can be single FP16 GEMMs over stacked weights instead of the JIT
 *       MMA kernels.
 *   <li>{@link #nativeAttention()} — cuDNN <i>and</i> its fused scaled-dot-product-attention shim
 *       are reachable, so the first prefill chunk of a sequence can be one fused attention call.
 * </ul>
 *
 * <p><b>Both are gated on the tensor-core backend first</b>, and that is not a convenience. The
 * tensor-core gate is the same predicate {@code Qwen3FP16PlanComponents} uses to decide whether the
 * MMA batch-prefill family is the one being built at all, and the decode layer graphs ask {@link
 * #nativeProjections()} to work out which weights the prefill graphs stopped reading. If the two
 * sides could ever disagree, the decode graphs would consume a buffer that was never uploaded and
 * read zeros — silently, with no diagnostic. So this answer depends on the device and on the
 * libraries, and on nothing per-model, per-session or per-request.
 *
 * <p><b>Resolved lazily and cached.</b> Nothing here runs unless something asks, and what asks is
 * the tensor-core batch-prefill planner and the decode family that pairs with it. A CPU run, an
 * OpenCL run, a Metal run or a single-token plan never reaches this class, so no optional native
 * library is dlopened on their behalf. The tensor-core check is evaluated first and short-circuits
 * for the same reason.
 *
 * <p><b>{@code -Djllm.prefill.native=false} turns both off.</b> That is a supported operational
 * control, not an experiment: it is the escape hatch for a host whose vendor libraries load but
 * misbehave, and it is what the weight-handoff regression tests flip to compare the native and JIT
 * paths in one process. There is deliberately no per-operation property — a configuration where
 * only some of these are on has never been validated end to end.
 */
public final class NativePrefillSupport {

    /** The supported off-switch, {@code -Djllm.prefill.native=false}. */
    public static final String PROPERTY = "jllm.prefill.native";

    private NativePrefillSupport() {}

    /**
     * Read live rather than cached in a constant, so that a test can compare the native and JIT
     * paths in one process by setting it around two model loads.
     *
     * <p><b>It must be set before a plan is built and left alone until that plan is discarded.</b>
     * The batch-prefill graphs and the decode graphs that consume their weights are built in one
     * pass and both ask this; an answer that changed between them would leave the decode side
     * binding buffers the prefill side never uploaded — the silent-zeros defect the weight-handoff
     * regression tests exist to catch. Nothing in jllm writes this property.
     */
    private static boolean enabled() {
        return !"false".equalsIgnoreCase(System.getProperty(PROPERTY, "true"));
    }

    /** Whether the four batch-prefill projections run as cuBLAS GEMMs over stacked weights. */
    public static boolean nativeProjections() {
        return enabled() && TensorCoreSupport.isTensorCoreCapableBackend() && Probe.CUBLAS;
    }

    /**
     * Whether the first prefill chunk of a sequence runs its attention as one fused cuDNN call.
     *
     * <p>Implies {@link #nativeProjections()}: the two have only ever been measured and validated
     * together, and the fallback family that covers the later chunks is built from the primary's
     * stacked weights, which only exist when the projections are native.
     */
    public static boolean nativeAttention() {
        return nativeProjections() && Probe.CUDNN_SDPA;
    }

    /** A one-line summary for the plan's startup banner. */
    public static String describe() {
        if (!enabled()) {
            return "off (-Djllm.prefill.native=false)";
        }
        if (!TensorCoreSupport.isTensorCoreCapableBackend()) {
            return "off (no tensor-core backend)";
        }
        if (!Probe.CUBLAS) {
            return "off (cuBLAS not reachable)";
        }
        return Probe.CUDNN_SDPA
                ? "cuBLAS projections + cuDNN attention"
                : "cuBLAS projections (cuDNN attention not reachable)";
    }

    /**
     * The library probes, in their own holder so that the class initializer above — which every
     * caller of {@link #nativeProjections()} triggers — does not dlopen anything. Only a caller
     * that gets past the tensor-core gate reaches these fields.
     */
    private static final class Probe {

        /** cuBLAS answers for itself: the provider dlopens the library and reports the result. */
        private static final boolean CUBLAS = probeCuBlas();

        /**
         * cuDNN <b>and</b> the fused SDPA path. {@code CuDnnLibraryProvider.isAvailable()} covers
         * the library; the fused attention additionally needs {@code libtornado-cudnn}, the small
         * JNI shim TornadoVM builds only when the toolkit is new enough for cudnn-frontend. The
         * shim's absence is reported by the provider at execution time, which is far too late for a
         * dispatch decision, so its presence is established here instead — by looking for the file
         * on the JVM's library path rather than by loading it, because a JNI library may only be
         * loaded by one class loader and that one has to be TornadoVM's.
         */
        private static final boolean CUDNN_SDPA = probeCuDnn() && sdpaShimPresent();

        private static boolean probeCuBlas() {
            try {
                return CuBlasLibraryProvider.isAvailable();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }

        private static boolean probeCuDnn() {
            try {
                return CuDnnLibraryProvider.isAvailable();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }

        private static boolean sdpaShimPresent() {
            String fileName = System.mapLibraryName("tornado-cudnn");
            String path = System.getProperty("java.library.path", "");
            for (String entry : path.split(java.io.File.pathSeparator)) {
                if (!entry.isEmpty() && new java.io.File(entry, fileName).isFile()) {
                    return true;
                }
            }
            return false;
        }
    }
}

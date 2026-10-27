package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TileLayout;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/**
 * Where vendor native libraries ({@code --with-native-libraries}) have an implementation.
 *
 * <p>On by default where they are implemented and installed ({@link #defaultFor}); everywhere else
 * the JIT kernels run. On CUDA the native path replaces batched-prefill projections with cuBLAS
 * GEMMs over stacked weight copies (and the first chunk's attention with cuDNN where it is usable),
 * which costs a second copy of those projections on the device — for Qwen3-8B F16, 8.6 GB beside
 * 13.2 GB of weights. On Metal it runs the batched prefill's attention and projections on Apple MLX
 * ({@link MlxPrefill}). {@code --no-native-libraries} keeps the JIT kernels.
 *
 * <p>Which families and weights have a path on which backend is each provider's answer ({@code
 * TornadoPlanProvider.nativeLibraryWeights}); nothing here lists them.
 *
 * <p>Asking for it explicitly anywhere it is not implemented is refused, never ignored: a request
 * that quietly runs the JIT kernels would read, in a benchmark, as a measurement of the libraries.
 */
public final class NativeLibrarySupport {

    private NativeLibrarySupport() {}

    private static final System.Logger LOGGER =
            System.getLogger(NativeLibrarySupport.class.getName());

    private static final AtomicBoolean INSTALL_NOTICE = new AtomicBoolean();

    /**
     * The policy a model runs when its caller did not choose native libraries: on where this
     * combination has a native implementation and its libraries load, the given policy otherwise.
     *
     * <p>Where the combination is implemented but a library is missing, a notice naming what to
     * install is printed once and the JIT kernels run.
     */
    public static ExecutionPolicy defaultFor(Model model, ExecutionPolicy policy) {
        if (policy.nativeLibraries()) {
            return policy;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, true);
        if (unsupported(combination).isPresent() || losesTiles(combination, policy)) {
            return policy;
        }
        boolean metal = BackendId.METAL.equals(combination.backend());
        boolean available =
                metal
                        ? loads(MlxPrefill::available)
                        : NativePrefillSupport.cublasLtResolvable()
                                && loads(NativePrefillSupport::cublasAvailable);
        if (!available) {
            noteInstall(
                    metal ? "MLX (brew install mlx)" : "cuBLAS 12 and cuDNN 9",
                    "falling back to the JIT kernels");
            return policy;
        }
        if (!metal) {
            Optional<String> tooOld =
                    NativePrefillSupport.cublasTooOld(TornadoDevices.targetCudaComputeCapability());
            if (tooOld.isPresent()) {
                notice(tooOld.get() + "; falling back to the JIT kernels");
                return policy;
            }
        }
        if (!metal && !loads(NativePrefillSupport::cudnnAvailable)) {
            noteInstall("cuDNN 9", "prefill attention falls back to the JIT kernels");
        }
        return ExecutionPolicy.from(policy).nativeLibraries(true).build();
    }

    /**
     * Whether native libraries would give up tiles the JIT plan holds the weights in under {@code
     * policy} ({@link TileLayout#tiledTypes}): Gemma 4's cuBLAS projections read none, so its Q4_0
     * keeps the packed JIT kernels by default.
     */
    private static boolean losesTiles(Combination c, ExecutionPolicy policy) {
        ExecutionPolicy nativeLibraries =
                ExecutionPolicy.from(policy).nativeLibraries(true).build();
        return TornadoPlanRegistry.provider(c.architecture())
                .map(
                        provider ->
                                TileLayout.tiledTypes(provider, policy).contains(c.weights())
                                        && !TileLayout.tiledTypes(provider, nativeLibraries)
                                                .contains(c.weights()))
                .orElse(false);
    }

    /** Whether a library probe answers yes; a library whose classes cannot be linked does not. */
    private static boolean loads(java.util.function.BooleanSupplier probe) {
        try {
            return probe.getAsBoolean();
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }

    private static void noteInstall(String libraries, String fallback) {
        notice("For maximum performance please install " + libraries + "; " + fallback);
    }

    /** Prints a notice about the native libraries, once per process. */
    private static void notice(String message) {
        if (INSTALL_NOTICE.compareAndSet(false, true)) {
            LOGGER.log(System.Logger.Level.INFO, message + ".");
        }
    }

    /** Why native libraries cannot be used here, or empty when they can. */
    public static Optional<String> unsupported(Combination c) {
        if (BackendId.CPU.equals(c.backend())) {
            return Optional.of("the CPU path has no native-library implementation");
        }
        if (!BackendId.CUDA.equals(c.backend()) && !BackendId.METAL.equals(c.backend())) {
            return Optional.of("native libraries are implemented on CUDA and Metal only");
        }
        boolean implemented =
                TornadoPlanRegistry.provider(c.architecture())
                        .map(
                                provider ->
                                        provider.nativeLibraryWeights(c.backend())
                                                .contains(c.weights()))
                        .orElse(false);
        if (!implemented) {
            return Optional.of(
                    "no native-library path is implemented for "
                            + c.architecture()
                            + " / "
                            + c.weights()
                            + " on "
                            + c.backend()
                            + " yet ("
                            + TornadoPlanRegistry.nativeLibraryPaths(c.backend())
                            + " only)");
        }
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE) {
            return Optional.of(
                    "native libraries replace batched-prefill kernels only; add"
                            + " --with-prefill-decode --batch-prefill-size N");
        }
        if (BackendId.METAL.equals(c.backend())) {
            return Optional.empty();
        }
        if (!c.tensorCores()) {
            return Optional.of("the native path needs a tensor-core device");
        }
        return Optional.empty();
    }

    /**
     * Refuses a native-library request the resolved configuration does not implement.
     *
     * @throws UnsupportedOperationException naming the combination, the reason, and the way out
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!policy.nativeLibraries()) {
            return;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, gpu);
        Optional<String> reason = unsupported(combination);
        if (reason.isEmpty()) {
            if (BackendId.METAL.equals(combination.backend())) {
                if (!MlxPrefill.available()) {
                    reason = Optional.of("MLX's mlx.metallib could not be loaded in this process");
                }
            } else if (!NativePrefillSupport.cublasLtResolvable()) {
                reason =
                        Optional.of(
                                NativePrefillSupport.CUBLAS_LT_SONAME
                                        + " is not on the library path (LD_LIBRARY_PATH or the"
                                        + " loader cache)");
            } else if (!NativePrefillSupport.cublasAvailable()) {
                reason = Optional.of("cuBLAS could not be loaded in this process");
            } else {
                reason =
                        NativePrefillSupport.cublasTooOld(
                                TornadoDevices.targetCudaComputeCapability());
            }
        }
        reason.ifPresent(
                r -> {
                    throw new UnsupportedOperationException(refusal(combination, r));
                });
    }

    /** The refusal: the combination, why, and how to run with the JIT kernels instead. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "native libraries (--with-native-libraries, experimental) are not implemented for "
                        + combination
                        + ": "
                        + reason
                        + ". Drop --with-native-libraries (the JIT kernels run this"
                        + " configuration), or nativeLibraries(false) in the execution policy");
    }
}

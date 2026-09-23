package org.beehive.jllm.integration.cli;

import java.util.concurrent.atomic.AtomicBoolean;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/** The command-line warnings for experimental options, each printed once per process. */
public final class ExperimentalWarnings {

    private static final AtomicBoolean NATIVE_LIBRARIES = new AtomicBoolean();

    static final String NATIVE_LIBRARIES_WARNING =
            "WARNING: native libraries (--with-native-libraries) are experimental. They are"
                    + " implemented for Qwen3 F16 batched prefill on CUDA tensor-core devices only"
                    + " (cuBLAS projections, cuDNN attention) and keep stacked projection copies"
                    + " beside the weights, so they need more device memory than the JIT kernels;"
                    + " any other configuration is refused.";

    private ExperimentalWarnings() {}

    /** Prints the native-libraries warning if {@code policy} asks for them, once. */
    public static void nativeLibraries(ExecutionPolicy policy) {
        if (policy.nativeLibraries() && NATIVE_LIBRARIES.compareAndSet(false, true)) {
            System.err.println(NATIVE_LIBRARIES_WARNING);
        }
    }
}

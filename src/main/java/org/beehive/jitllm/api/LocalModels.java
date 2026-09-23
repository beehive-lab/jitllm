package org.beehive.jllm.api;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.backend.Device;
import org.beehive.jllm.runtime.backend.DeviceResolver;
import org.beehive.jllm.runtime.backend.DeviceResolvers;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.memory.MemoryPlan;

/**
 * Where a caller starts: load a model file, get a {@link LocalModel}.
 *
 * <p>Thread-safe; it holds no state.
 *
 * <pre>{@code
 * try (LocalModel model = LocalModels.load(Path.of("Llama-3.2-1B-Instruct-Q8_0.gguf"))) {
 *     TextGenerationModel generator = (TextGenerationModel) model;
 *     try (GenerationSession session = generator.newSession()) {
 *         System.out.println(session.generate(GenerationRequest.of("Why is the sky blue?")).text());
 *     }
 * }
 * }</pre>
 *
 * <p>No backend or device argument in v1 — see {@link ModelOptions}.
 */
public final class LocalModels {

    private LocalModels() {}

    /** Loads with the model file's own settings. */
    public static LocalModel load(Path modelFile) throws IOException {
        return load(modelFile, ModelOptions.defaults());
    }

    /**
     * Loads with the given options.
     *
     * @throws IOException if the file cannot be read or is not a model this build understands
     */
    public static LocalModel load(Path modelFile, ModelOptions options) throws IOException {
        Objects.requireNonNull(modelFile, "modelFile");
        Objects.requireNonNull(options, "options");

        // A context length of 0 means "whatever the model says"; the loader reads a negative
        // value as "no override", which is the same intent spelled differently.
        int contextLength = options.contextLength() > 0 ? options.contextLength() : -1;
        boolean gpu = useGpu(options);

        if (gpu) {
            MemoryPreflight.refuseIfOverCapacity(modelFile, contextLength, options);
        }
        Model model;
        try {
            model = ModelLoader.loadModel(modelFile, contextLength, true, gpu);
        } catch (RuntimeException | OutOfMemoryError exhaustion) {
            if (gpu && isDeviceMemoryExhaustion(exhaustion)) {
                throw deviceMemoryExhausted(modelFile, options, exhaustion);
            }
            throw exhaustion;
        }
        return new DelegatingModel(
                model,
                modelFile,
                gpu,
                options.executionPolicy(),
                options.storageOptions(),
                options.thinkingMode(),
                options.maxConcurrentSessions());
    }

    /**
     * Whether a load failure is the device running out of memory, in either form it arrives in.
     *
     * <p>Two distinct signatures, both observed on a real fixture:
     *
     * <ul>
     *   <li>{@code TornadoOutOfMemoryException} — the backend's configured device budget is
     *       exhausted while buffers are allocated;
     *   <li>{@code java.lang.OutOfMemoryError: Cannot reserve … direct buffer memory} — the host
     *       runs out of direct memory materializing the weights, which is where a representation
     *       the device has no kernel for is converted to Q8_0.
     * </ul>
     */
    private static boolean isDeviceMemoryExhaustion(Throwable failure) {
        for (Throwable t = failure; t != null; t = t.getCause()) {
            String type = t.getClass().getName();
            if (type.startsWith("uk.ac.manchester.tornado")
                    && type.toLowerCase(java.util.Locale.ROOT).contains("outofmemory")) {
                return true;
            }
            if (t instanceof OutOfMemoryError
                    && t.getMessage() != null
                    && t.getMessage().contains("direct buffer memory")) {
                return true;
            }
        }
        return false;
    }

    /**
     * The same failure, told in this project's own terms.
     *
     * <p>What the caller saw before was the backend's message — a byte count and an instruction to
     * raise {@code -Dtornado.device.memory}, a TornadoVM-internal property rather than the {@code
     * --gpu-memory} flag this project exposes. The prediction is included when it can still be
     * produced, because it names the dominant component, which a byte count does not.
     */
    private static InsufficientDeviceMemoryException deviceMemoryExhausted(
            Path modelFile, ModelOptions options, Throwable cause) {
        StringBuilder message =
                new StringBuilder(
                                org.beehive.jllm.runtime.diagnostics.DiagnosticCode
                                        .DEVICE_MEMORY_INSUFFICIENT
                                        .prefix())
                        .append("The device ran out of memory loading ")
                        .append(modelFile.getFileName())
                        .append(
                                ".\n  The preflight did not refuse this load beforehand: it reports")
                        .append(" CONSERVATIVE confidence on this backend, which is report-only.\n")
                        .append(
                                "  Raise --gpu-memory, reduce the context length, or select a smaller")
                        .append(" quantization.\n  Backend failure: ")
                        .append(cause);
        try {
            message.append("\n\n").append(preflight(modelFile, options).describe());
        } catch (RuntimeException | java.io.IOException unavailable) {
            // A prediction is an aid here, not the finding — the exhaustion is already established.
            message.append("\n  (no memory plan available: ").append(unavailable).append(')');
        }
        return new InsufficientDeviceMemoryException(message.toString(), cause);
    }

    /**
     * What this configuration is predicted to need on the device, <b>without loading it</b>.
     *
     * <p>Reads the model's descriptors, not its tensors, so it costs a metadata parse rather than a
     * multi-gigabyte upload — the point is to answer "will this fit" before finding out the
     * expensive way.
     *
     * <p><b>Three quantities, kept apart</b>, and the report says which is which: the logical bytes
     * the buffers occupy; the predicted budget consumption, which is larger whenever the backend
     * binds one buffer into more than one allocation domain; and physical free device memory, which
     * is <b>not reported</b> and is not part of this contract.
     *
     * <p>No backend or device type appears in the result. A {@link MemoryPlan} names components,
     * byte counts and a confidence — never a task graph.
     *
     * @throws IOException if the file cannot be read or is not a model this build understands
     */
    @Experimental
    public static MemoryPlan preflight(Path modelFile, ModelOptions options) throws IOException {
        Objects.requireNonNull(modelFile, "modelFile");
        Objects.requireNonNull(options, "options");
        int contextLength = options.contextLength() > 0 ? options.contextLength() : -1;
        return MemoryPreflight.plan(modelFile, contextLength, options);
    }

    /** Preflights with the model file's own settings. */
    @Experimental
    public static MemoryPlan preflight(Path modelFile) throws IOException {
        return preflight(modelFile, ModelOptions.defaults());
    }

    /** Whether to run on an accelerator. */
    private static boolean useGpu(ModelOptions options) {
        BackendId backend = options.resolvedBackend();
        if (backend == null) {
            return Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
        }
        if (BackendId.CPU.equals(backend)) {
            return false;
        }
        verifyAcceleratorHonoured(backend, DeviceResolvers.discovered().orElse(null));
        return true;
    }

    /**
     * Confirms an explicit accelerator request is the one this process actually resolves.
     *
     * <p>Package-private and taking the resolver as a parameter — rather than reading {@link
     * DeviceResolvers#discovered()} itself — so every mismatch shape (no resolver at all, a
     * resolved backend that disagrees) is exercised with a fake {@link DeviceResolver} in a unit
     * test, deterministically, with no real accelerator and no competing {@code ServiceLoader}
     * provider installed.
     *
     * @throws UnsupportedOperationException naming both identities, before any allocation — no
     *     silent fallback to whichever backend TornadoVM actually picked [this corrects a request
     *     that was never honoured, it does not change what already worked]
     */
    static void verifyAcceleratorHonoured(BackendId requested, DeviceResolver resolver) {
        if (resolver == null) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.DEVICE_SELECTOR_UNSUPPORTED.message(
                            "backend("
                                    + requested
                                    + ") was requested, but no accelerator backend is"
                                    + " available in this build — no device resolver was discovered. Requesting an"
                                    + " accelerator where none can be provided is rejected rather than silently run"
                                    + " on the host"));
        }
        Device resolved = resolver.resolve();
        BackendId resolvedBackend = resolved.id().backend();
        if (!requested.equals(resolvedBackend)) {
            throw new UnsupportedOperationException(
                    DiagnosticCode.DEVICE_SELECTOR_UNSUPPORTED.message(
                            "backend("
                                    + requested
                                    + ") was requested, but this process's TornadoVM"
                                    + " configuration resolves "
                                    + resolved.id()
                                    + " (\""
                                    + resolved.displayName()
                                    + "\"). Selecting a different backend requires configuring TornadoVM's own"
                                    + " priority/device properties before this process starts; a request that"
                                    + " disagrees with the already-resolved device is rejected rather than silently"
                                    + " run on it"));
        }
    }
}

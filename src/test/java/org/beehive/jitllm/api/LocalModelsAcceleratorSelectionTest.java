package org.beehive.jitllm.api;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceId;
import org.beehive.jitllm.runtime.backend.DeviceResolver;
import org.junit.Test;

/**
 * The full-path cases ({@link #explicitCudaFailsBeforeAnyFileIsTouched()} and its siblings) use a
 * deliberately nonexistent model path. The point is the exception <b>type</b>, not its message: an
 * accelerator mismatch is an {@link UnsupportedOperationException} raised inside {@code useGpu},
 * before {@code ModelLoader} ever opens the file, while a path that reaches the loader without
 * incident fails with a plain file-not-found {@link IOException} instead. That difference is what
 * proves validation ran before any allocation, on this project's own real entry point.
 */
public class LocalModelsAcceleratorSelectionTest {

    private static final Path NONEXISTENT =
            Path.of("/definitely/does-not-exist/no-such-model.gguf");

    private static Device device(BackendId backend, String handle) {
        DeviceId id = DeviceId.of(backend, handle);
        return new Device() {
            @Override
            public DeviceId id() {
                return id;
            }

            @Override
            public DeviceCapabilities capabilities() {
                return DeviceCapabilities.NONE;
            }

            @Override
            public String displayName() {
                return handle;
            }
        };
    }

    private static DeviceResolver resolverOf(Device device) {
        return () -> device;
    }

    // --- the package-private policy, exercised directly with fakes -------------------------

    @Test
    public void matchingBackendIsHonouredSilently() {
        // No exception is the assertion: requesting exactly what resolves must be a no-op.
        LocalModels.verifyAcceleratorHonoured(
                BackendId.CUDA, resolverOf(device(BackendId.CUDA, "NVIDIA GPU 0")));
    }

    @Test
    public void aRealMetalRequestAgainstAResolvedMetalDeviceIsHonoured() {
        LocalModels.verifyAcceleratorHonoured(
                BackendId.METAL, resolverOf(device(BackendId.METAL, "Apple Metal")));
    }

    @Test
    public void mismatchedBackendThrowsNamingBothIdentities() {
        UnsupportedOperationException thrown =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                LocalModels.verifyAcceleratorHonoured(
                                        BackendId.CUDA,
                                        resolverOf(device(BackendId.METAL, "Apple Metal"))));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("cuda"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("metal"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("Apple Metal"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("GPUL-CFG-001"));
    }

    @Test
    public void openclRequestAgainstAResolvedCudaDeviceThrows() {
        UnsupportedOperationException thrown =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                LocalModels.verifyAcceleratorHonoured(
                                        BackendId.OPENCL,
                                        resolverOf(device(BackendId.CUDA, "NVIDIA GPU 0"))));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("opencl"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("cuda"));
    }

    @Test
    public void noResolverAtAllThrowsANamedUnavailableDiagnostic() {
        UnsupportedOperationException thrown =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> LocalModels.verifyAcceleratorHonoured(BackendId.CUDA, null));
        assertTrue(
                thrown.getMessage(),
                thrown.getMessage().contains("no device resolver was discovered"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("GPUL-CFG-001"));
    }

    // --- the full LocalModels.load() path, proving order rather than just outcome ----------

    @Test
    public void explicitCudaFailsBeforeAnyFileIsTouched() {
        // On this machine only Metal resolves, so CUDA is always a mismatch — the point under test
        // is that the failure is UnsupportedOperationException, not a file-system error, which is
        // only possible if useGpu() ran (and threw) before ModelLoader ever opened NONEXISTENT.
        ModelOptions options = ModelOptions.builder().backend(BackendId.CUDA).build();
        assertThrows(
                UnsupportedOperationException.class, () -> LocalModels.load(NONEXISTENT, options));
    }

    @Test
    public void explicitCpuNeverReachesAcceleratorValidationAndFailsOnTheFileInstead()
            throws IOException {
        // CPU must not resolve an accelerator at all; the only way to observe that from outside is
        // that the failure is the ordinary "no such file" one, not an accelerator diagnostic.
        ModelOptions options = ModelOptions.builder().backend(BackendId.CPU).build();
        IOException thrown =
                assertThrows(IOException.class, () -> LocalModels.load(NONEXISTENT, options));
        assertTrue(
                thrown.getClass().getName(),
                thrown instanceof NoSuchFileException
                        || thrown.getMessage() != null
                                && thrown.getMessage().contains("does-not-exist"));
    }

    @Test
    public void unspecifiedBackendPreservesThePropertyDrivenDefaultAndSkipsValidation() {
        // No backend named: -Duse.tornadovm decides, exactly as before this change, and no
        // accelerator identity is ever compared — the failure is the file, not a mismatch.
        String previous = System.getProperty("use.tornadovm");
        System.clearProperty("use.tornadovm");
        try {
            ModelOptions options = ModelOptions.defaults();
            IOException thrown =
                    assertThrows(IOException.class, () -> LocalModels.load(NONEXISTENT, options));
            assertTrue(
                    thrown.getClass().getName(),
                    thrown instanceof NoSuchFileException
                            || thrown.getMessage() != null
                                    && thrown.getMessage().contains("does-not-exist"));
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }
}

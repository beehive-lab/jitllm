package org.beehive.jllm.model.loader;

import static org.junit.Assert.assertEquals;

import org.beehive.jllm.runtime.backend.BackendId;
import org.junit.Test;

public class ModelLoaderProviderBackendTest {

    @Test
    public void falseIsAlwaysCpu() {
        assertEquals(BackendId.CPU, ModelLoader.providerBackend(false));
    }

    @Test
    public void trueWithNoAcceleratorKeepsTheOldCudaPlaceholderRatherThanSilentlyBecomingCpu() {
        // Under the plain unit profile no accelerator resolves, so TornadoDevices.current() itself
        // falls back to the CPU placeholder. providerBackend must not report that placeholder here
        // -- doing so would flip a provider's `!BackendId.CPU.equals(backend)` check to false and
        // silently take the CPU path for a caller that asked for the GPU one.
        assertEquals(BackendId.CUDA, ModelLoader.providerBackend(true));
    }
}

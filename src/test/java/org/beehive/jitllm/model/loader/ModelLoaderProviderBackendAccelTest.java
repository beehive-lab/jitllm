package org.beehive.jllm.model.loader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeFalse;
import static org.junit.Assume.assumeTrue;

import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.runtime.backend.BackendId;
import org.junit.Test;

public class ModelLoaderProviderBackendAccelTest {

    @Test
    public void trueReportsTheRealResolvedBackendTruthfully() {
        var resolved = TornadoDevices.current();
        assumeFalse(
                "no accelerator resolved on this run",
                "unavailable".equals(resolved.displayName()));

        assertEquals(resolved.id().backend(), ModelLoader.providerBackend(true));
    }

    /**
     * Names the specific defect fixed: this project's own machine resolves Metal, and {@code
     * providerBackend} must not still claim CUDA for it — that was the exact mislabel every {@code
     * ModelProvider} implementation silently accepted before this task, because each one only ever
     * tested {@code !BackendId.CPU.equals(backend)}.
     */
    @Test
    public void aMetalRunIsNotRewrittenAsCuda() {
        var resolved = TornadoDevices.current();
        assumeTrue("not a Metal run", BackendId.METAL.equals(resolved.id().backend()));

        BackendId reported = ModelLoader.providerBackend(true);
        assertEquals(BackendId.METAL, reported);
        assertNotEquals(BackendId.CUDA, reported);
    }
}

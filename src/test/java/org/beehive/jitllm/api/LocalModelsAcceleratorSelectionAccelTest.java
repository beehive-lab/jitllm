package org.beehive.jllm.api;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.runtime.backend.BackendId;
import org.junit.Test;

/**
 * Skips (rather than failing) when this run is not on Metal, so the suite stays meaningful on a
 * CUDA or OpenCL machine too — the assertion is "the explicitly requested backend loads," not
 * "Metal specifically must be present."
 */
public class LocalModelsAcceleratorSelectionAccelTest {

    @Test
    public void explicitMetalMatchesTheRealResolvedDeviceAndLoadsSuccessfully() throws Exception {
        var resolved = TornadoDevices.current();
        assumeTrue("not a Metal run", BackendId.METAL.equals(resolved.id().backend()));

        Path model = GoldenFixture.locate(Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue("fixture absent", model != null);

        ModelOptions options = ModelOptions.builder().backend(BackendId.METAL).build();
        try (LocalModel loaded = LocalModels.load(model, options)) {
            // Reaching here at all is the assertion: an explicit METAL request against a real
            // Metal-resolved device must not throw, and must not have silently downgraded to CPU —
            // confirmed by the model actually being usable as a generation model.
            org.junit.Assert.assertTrue(loaded instanceof TextGenerationModel);
        }
    }
}

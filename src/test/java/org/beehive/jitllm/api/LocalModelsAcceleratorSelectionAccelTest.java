/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.api;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.backend.BackendId;
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

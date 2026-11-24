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
package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeFalse;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
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

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
import static org.junit.Assume.assumeFalse;
import static org.junit.Assume.assumeTrue;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.junit.Test;

public class ModelLoaderProviderBackendTest {

    @Test
    public void falseIsAlwaysCpu() {
        assertEquals(BackendId.CPU, ModelLoader.providerBackend(false));
    }

    @Test
    public void trueWithNoAcceleratorKeepsTheOldCudaPlaceholderRatherThanSilentlyBecomingCpu() {
        // With no accelerator, TornadoDevices.current() falls back to the CPU placeholder.
        // providerBackend must not report that placeholder here -- doing so would flip a provider's
        // `!BackendId.CPU.equals(backend)` check to false and silently take the CPU path for a
        // caller that asked for the GPU one.
        assumeTrue(
                "an accelerator resolves on this machine",
                BackendId.CPU.equals(TornadoDevices.current().backend()));
        assertEquals(BackendId.CUDA, ModelLoader.providerBackend(true));
    }

    @Test
    public void trueWithAnAcceleratorReportsTheResolvedBackend() {
        BackendId resolved = TornadoDevices.current().backend();
        assumeFalse("no accelerator resolves on this machine", BackendId.CPU.equals(resolved));
        assertEquals(resolved, ModelLoader.providerBackend(true));
    }
}

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
package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.DeviceDefaultPolicy;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;

/**
 * The device's default policy is taken only where it is predicted to fit: a GPU load that would run
 * out of memory on the default keeps the policy that was checked before the load.
 */
public class DeviceDefaultPolicyFitAccelTest {

    @Test
    public void aDefaultThatDoesNotFitIsSkipped() throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_Q8_0),
                file != null);
        Model model = KvPrecisionHarness.load(file, 1024, true);
        ExecutionPolicy chosen = ExecutionPolicy.fromSystemProperties();
        StorageOptions storage = StorageOptions.fromSystemProperties();

        ExecutionPolicy fitting =
                DeviceDefaultPolicy.resolve(model, chosen, true, true, storage, policy -> true);
        assertNotEquals("the batched default is taken where it fits", chosen, fitting);

        ExecutionPolicy none =
                DeviceDefaultPolicy.resolve(model, chosen, true, true, storage, policy -> false);
        assertEquals("nothing that does not fit is taken", chosen, none);
    }
}

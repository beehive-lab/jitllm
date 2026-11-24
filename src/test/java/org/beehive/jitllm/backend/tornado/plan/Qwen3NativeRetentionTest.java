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
package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/** A Qwen3 load keeps every tensor as the file stores it, and its plans admit what it reports. */
public class Qwen3NativeRetentionTest {

    /** Admission takes every quantized type a kept load reports, and F16. */
    @Test
    public void everyKeptTypeIsAdmitted() {
        assertEquals(
                java.util.EnumSet.of(
                        DataType.F16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q5_0,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K,
                        DataType.IQ4_XS),
                new Qwen3PlanProvider().supportedDataTypes());
    }
}

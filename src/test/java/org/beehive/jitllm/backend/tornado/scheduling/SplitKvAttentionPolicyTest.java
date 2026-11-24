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
package org.beehive.jitllm.backend.tornado.scheduling;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class SplitKvAttentionPolicyTest {

    @Test
    public void theLocalArraysMatchWhatTheKernelDeclares() {
        // The byte counts the device reports when it refuses the pipeline: Metal's message for the
        // 64-thread kernel reads "Threadgroup memory size (34052) exceeds ... (32768)".
        assertEquals(34052L, SplitKvAttentionPolicy.localBytes(64));
        assertEquals(17284L, SplitKvAttentionPolicy.localBytes(32));
    }

    @Test
    public void aDeviceWithThirtyTwoKilobytesRunsTheNarrowKernel() {
        assertEquals(SplitKvAttentionPolicy.NARROW_GROUP, SplitKvAttentionPolicy.groupSize(32768));
    }

    @Test
    public void aDeviceWithMoreLocalMemoryKeepsTheWideKernel() {
        assertEquals(SplitKvAttentionPolicy.WIDE_GROUP, SplitKvAttentionPolicy.groupSize(49152));
        assertEquals(SplitKvAttentionPolicy.WIDE_GROUP, SplitKvAttentionPolicy.groupSize(34052));
    }

    @Test
    public void anUnknownLimitKeepsTheWideKernel() {
        assertEquals(SplitKvAttentionPolicy.WIDE_GROUP, SplitKvAttentionPolicy.groupSize(0));
    }
}

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

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Whether the shuffle-reduced FP16 matrix-vector kernels run their unrolled variants, which keep
 * four independent loads in flight per lane.
 *
 * <p>A preference: both variants compute the same sums, in a different order. On an Apple GPU the
 * unrolled ones are faster: Qwen3-0.6B F16 decode on an M4 Pro went from 153.4 to 164.7 tok/s at
 * depth 0. On CUDA they are slower: Qwen3-0.6B F16 decode on an A100 drops from about 146 to 135
 * tok/s, so the other backends keep the original kernels.
 */
// @formatter:on
public final class Fp16GemvUnrollPolicy {

    private Fp16GemvUnrollPolicy() {}

    /** Whether the active device runs the unrolled FP16 matrix-vector kernels. */
    public static boolean unrolled() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }
}

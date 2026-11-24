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
package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.DecodedMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

/**
 * The tensor-core batched prefill on the JIT GEMM: JIT kernels decode the weights to FP16, then the
 * JIT tensor-core GEMM multiplies. Used on any GPU with tensor cores.
 */
public final class TensorCoreMatMul extends DecodedMatMul {

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.supports(DeviceCapability.TENSOR_CORE_MMA);
    }

    @Override
    protected boolean nativeGemm() {
        return false;
    }
}

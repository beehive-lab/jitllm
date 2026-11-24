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

import org.beehive.jitllm.backend.tornado.ops.DequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * The tensor-core batched prefill over quantized weights decoded one projection at a time, the GEMM
 * the JIT one. Used on any GPU with tensor cores.
 */
public final class DequantizingTensorCoreMatMul extends DequantizingMatMul {

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return tensorCoresOverDecoded(step, target);
    }

    @Override
    protected boolean nativeGemm() {
        return false;
    }
}

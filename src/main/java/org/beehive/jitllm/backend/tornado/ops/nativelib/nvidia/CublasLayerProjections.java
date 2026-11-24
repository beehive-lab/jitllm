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
package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * {@link LayerProjections} on cuBLAS: the projections the layer class has a cuBLAS form for.
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}), the GPU has tensor
 * cores and cuBLAS loads.
 */
public final class CublasLayerProjections extends LayerProjections {

    public CublasLayerProjections(Tasks tasks) {
        super(tasks);
    }

    @Override
    public String library() {
        return "cuBLAS";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.cublas() && tasks.cublasForm(step);
    }

    @Override
    protected boolean cublas() {
        return true;
    }
}

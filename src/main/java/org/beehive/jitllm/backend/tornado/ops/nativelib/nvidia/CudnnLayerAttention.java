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

import org.beehive.jitllm.backend.tornado.ops.LayerAttention;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * {@link LayerAttention} on cuDNN's fused attention: the layers the layer class has a cuDNN form
 * for, which it answers with the probe for its own attention shape ({@link
 * NativePrefillSupport#nativeAttention}).
 *
 * <p>Used when native libraries are allowed (no {@code --no-native-libraries}) and the GPU has
 * tensor cores.
 */
public final class CudnnLayerAttention extends LayerAttention {

    public CudnnLayerAttention(Tasks tasks) {
        super(tasks);
    }

    @Override
    public String library() {
        return "cuDNN";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.cublas() && tasks.cudnnForm(step);
    }

    @Override
    protected boolean cudnn() {
        return true;
    }
}

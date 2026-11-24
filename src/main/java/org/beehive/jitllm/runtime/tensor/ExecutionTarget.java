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
package org.beehive.jitllm.runtime.tensor;

/**
 * Where a model is being loaded to execute — the axis that decides which representation a tensor is
 * materialized in.
 *
 * <p>It exists because the answer genuinely differs. The host decodes K-quants and Q4_0 inside the
 * dot product and needs no conversion; the accelerator path has no kernel for them and materializes
 * {@link DataType#Q8_0} at load. A mapping without this parameter would have to pick one of those
 * answers and be wrong about the other, silently.
 */
public enum ExecutionTarget {

    /** Executed on the host CPU, which decodes quantized blocks during compute. */
    CPU,

    /**
     * Executed on an accelerator through TornadoVM, which needs a kernel for each representation.
     */
    GPU
}

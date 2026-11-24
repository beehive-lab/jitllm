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
package org.beehive.jitllm.inference.weights;

import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * jitllm utilizes two distinct weight types:
 *
 * <ul>
 *   <li><b>StandardWeights:</b> Designed for standard Java-based inference on the CPU.
 *   <li><b>TornadoWeights:</b> Optimized for GPU-accelerated inference using TornadoVM.
 * </ul>
 *
 * The packages <code>weights.standard</code> and <code>weights.tornado</code> define base classes
 * and model-specific implementations for weights in their respective formats.
 */
public interface Weights {

    /**
     * The representation these weights are stored in, in the runtime's vocabulary.
     *
     * <p>The type an operation over them is parameterized by. For GPU weights this is what was
     * <i>materialized</i>, which is not always what the file held: a K-quant or Q4_0 file becomes
     * {@link DataType#Q8_0} on the device.
     */
    DataType dataType();
}

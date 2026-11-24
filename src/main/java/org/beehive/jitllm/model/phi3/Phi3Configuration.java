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
package org.beehive.jitllm.model.phi3;

import org.beehive.jitllm.model.Configuration;

// @formatter:off
public record Phi3Configuration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int vocabularySize,
        int contextLength,
        float rmsNormEps,
        float ropeTheta)
        implements Configuration {

    /**
     * One: the decode graphs of a batched plan consume the weights the prefill graphs uploaded, so
     * the plan holds them once. Measured: Phi-3-mini F16 (7.1 GiB file) peaks at 7.9 GiB of GPU
     * memory batched on CUDA.
     */
    @Override
    public int weightBindingFamilies(int layerGraphFamilies) {
        return 1;
    }

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int numberOfHeadsKey() {
        // For Phi3, key heads are the same as key-value heads
        return numberOfKeyValueHeads;
    }

    @Override
    public int headSize() {
        // Calculate head size from dim and numberOfHeads
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        // Calculate key-value dimension
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    @Override
    public int kvMul() {
        // Calculate key-value multiplier for multi-query attention
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }
}

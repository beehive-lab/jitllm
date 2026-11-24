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
package org.beehive.jitllm.program.op;

/**
 * How a rotary embedding pairs the components it rotates.
 *
 * <p>The rotation is the same arithmetic in both cases — a complex multiply by a precomputed
 * frequency — and the layouts differ only in <b>which two components form a pair</b>. That is a
 * property of how a family stores its projections, which makes it configuration rather than a
 * second operation ({@code target-architecture.md}: operations are "parameterized by configuration,
 * not hard-coded per architecture").
 *
 * <p>Getting it wrong is silent: both layouts read valid floats and produce plausible activations,
 * and the model simply answers slightly wrong. That is the same failure shape as the hardcoded RoPE
 * bases this project already paid for, which is why the layout is named rather than implied by
 * which kernel happened to be called.
 */
public enum RopeLayout {

    /**
     * Adjacent components are paired: {@code (0,1)}, {@code (2,3)}, … Llama, Mistral, Devstral,
     * Granite and Phi3.
     */
    INTERLEAVED,

    /**
     * A component is paired with the one half a head away: {@code (i, i + headDim/2)}. The GPT-NeoX
     * arrangement, used by Qwen2, Qwen2-MoE, Qwen3 and Gemma.
     */
    NEOX_HALF
}

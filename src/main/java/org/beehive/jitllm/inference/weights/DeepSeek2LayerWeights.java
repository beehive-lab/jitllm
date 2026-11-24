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

/**
 * The per-block tensors of a {@code deepseek2} model, for either backend's tensor type.
 *
 * <p>Attention tensors are present at every block. The dense feed-forward ({@code ffnGate}, {@code
 * ffnUp}, {@code ffnDown}) is present only at the leading dense blocks and the mixture of experts
 * only after them; the slots a block does not carry are null.
 */
public record DeepSeek2LayerWeights<T>(
        T[] attnNorm,
        T[] qA,
        T[] qANorm,
        T[] qB,
        T[] kvAMqa,
        T[] kvANorm,
        T[] kB,
        T[] vB,
        T[] wo,
        T[] ffnNorm,
        T[] ffnGate,
        T[] ffnUp,
        T[] ffnDown,
        T[] router,
        T[] routerBias,
        T[] gateExperts,
        T[] upExperts,
        T[] downExperts,
        T[] sharedGate,
        T[] sharedUp,
        T[] sharedDown) {}

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
 * The mixture-of-experts tensors of a {@code qwen35moe} model, per layer, in the tensor type of the
 * backend that holds them ({@code FloatTensor} on the host, {@code TornadoTensor} on the device).
 *
 * <p>The routed experts stay stacked as the file stores them: {@link #gateExperts()}[l] holds every
 * expert's gate matrix, expert {@code e} at element offset {@code e * hiddenDim * dim}, and
 * likewise the up and down projections ({@code e * dim * hiddenDim}). Nothing is split or
 * converted.
 *
 * @param router {@code ffn_gate_inp}: one score row per expert, F32
 * @param gateExperts {@code ffn_gate_exps}
 * @param upExperts {@code ffn_up_exps}
 * @param downExperts {@code ffn_down_exps}
 * @param sharedGate {@code ffn_gate_shexp}
 * @param sharedUp {@code ffn_up_shexp}
 * @param sharedDown {@code ffn_down_shexp}
 * @param sharedGateInput {@code ffn_gate_inp_shexp}: the shared expert's gate vector, F32
 */
public record Qwen35ExpertWeights<T>(
        T[] router,
        T[] gateExperts,
        T[] upExperts,
        T[] downExperts,
        T[] sharedGate,
        T[] sharedUp,
        T[] sharedDown,
        T[] sharedGateInput) {}

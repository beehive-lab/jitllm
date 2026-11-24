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
package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Supplies the host forward pass for one architecture — discovered, never enumerated.
 *
 * <p>Families that share a routine still get one provider each. Mistral runs Llama's host pass and
 * DeepSeek runs Qwen2's, and "which routine" is a provider's answer to give rather than something a
 * caller should have to know.
 */
public interface CpuForwardProvider {

    /** The architecture this provider serves. */
    ArchitectureId architecture();

    /** The host forward pass for it. */
    ForwardPass create();
}

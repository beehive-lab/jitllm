package org.beehive.jllm.backend.cpu;

import org.beehive.jllm.inference.ForwardPass;
import org.beehive.jllm.runtime.model.ArchitectureId;

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

package org.beehive.jitllm.model.architecture;

import java.util.Objects;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * What an architecture needs to describe a program, and nothing more.
 *
 * <p>A description is a function of these four: the model's shape, how its weights were
 * materialized, how key/value entries are stored, and the execution choices that select components
 * (device sampling adds one, split-KV changes the policy descriptor). All four end up in the {@code
 * ProgramSignature}, which is why they are inputs rather than something the architecture looks up.
 *
 * @param configuration the model's shape; the architecture validates it
 * @param weights the materialized weight representation
 * @param keyValue how key/value entries are stored
 * @param policy the execution policy the program is described under
 */
public record ArchitectureInputs(
        Configuration configuration, DataType weights, DataType keyValue, ExecutionPolicy policy) {

    public ArchitectureInputs {
        Objects.requireNonNull(configuration, "configuration");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(keyValue, "keyValue");
        Objects.requireNonNull(policy, "policy");
    }

    /** Whether the described program ends with a device-resident sample. */
    public boolean deviceSample() {
        return policy.samplingResidency() == ExecutionPolicy.SamplingResidency.DEVICE;
    }

    /** Whether split-KV attention is part of the policy this program is described under. */
    public boolean splitKvAttention() {
        return policy.splitKvPartitions().isPresent();
    }
}

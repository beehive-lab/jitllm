package org.beehive.jitllm.model;

/**
 * The model families jitllm knows, as an internal identifier. Loading goes through the {@code
 * ModelProvider} that claims a file; this enum selects nothing.
 */
public enum ModelType {
    LLAMA_3,
    MISTRAL,
    DEVSTRAL_2,
    QWEN_2,
    QWEN_3,
    QWEN_2_MOE,
    DEEPSEEK_R1_DISTILL_QWEN,
    PHI_3,
    GRANITE,
    QWEN_3_5,
    GEMMA_4,
    DEEPSEEK_2,
    UNKNOWN;

    public boolean isDeepSeekR1() {
        return this == DEEPSEEK_R1_DISTILL_QWEN;
    }
}

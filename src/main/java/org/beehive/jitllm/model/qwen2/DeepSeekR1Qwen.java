package org.beehive.jllm.model.qwen2;

import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.tokenizer.Tokenizer;

public class DeepSeekR1Qwen extends Qwen2 {

    public DeepSeekR1Qwen(
            Qwen2Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(configuration, tokenizer, weights, chatFormat);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.DEEPSEEK_R1_DISTILL_QWEN;
    }

    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jllm.runtime.model.ArchitectureId.of("deepseek-r1-distill-qwen");
    }
}

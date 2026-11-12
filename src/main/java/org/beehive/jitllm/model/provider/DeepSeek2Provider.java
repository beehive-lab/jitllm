package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.DeepSeek2ModelLoader;

/** The {@code deepseek2} family: latent attention with a mixture of experts (GLM-4.7-Flash). */
public final class DeepSeek2Provider extends FamilyProvider {

    public DeepSeek2Provider() {
        super("deepseek2");
    }

    @Override
    protected DeepSeek2ModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new DeepSeek2ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.DeepSeek4ModelLoader;

/** The {@code deepseek4} family: DeepSeek-V4. */
public final class DeepSeek4Provider extends FamilyProvider {

    public DeepSeek4Provider() {
        super("deepseek4");
    }

    @Override
    protected DeepSeek4ModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new DeepSeek4ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

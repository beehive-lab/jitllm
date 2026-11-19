package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.GptOssModelLoader;

/** The {@code gpt-oss} architecture: OpenAI's gpt-oss-20b and gpt-oss-120b. */
public final class GptOssProvider extends FamilyProvider {

    public GptOssProvider() {
        super("gpt-oss");
    }

    @Override
    protected GptOssModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new GptOssModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

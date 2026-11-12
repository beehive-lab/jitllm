package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.Qwen35MoeModelLoader;

/**
 * The {@code qwen35moe} architecture: the {@code qwen35} hybrid stack with a mixture-of-experts
 * feed-forward (Qwen3.6-35B-A3B). The same loader reads it; the configuration it builds carries the
 * experts.
 */
public final class Qwen35MoeProvider extends FamilyProvider {

    public Qwen35MoeProvider() {
        super("qwen35moe");
    }

    @Override
    protected Qwen35MoeModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new Qwen35MoeModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

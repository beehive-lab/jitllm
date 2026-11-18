package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.KimiLinearModelLoader;

/** The {@code kimi-linear} family: Kimi-Linear-48B-A3B. */
public final class KimiLinearProvider extends FamilyProvider {

    public KimiLinearProvider() {
        super("kimi-linear");
    }

    @Override
    protected KimiLinearModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new KimiLinearModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

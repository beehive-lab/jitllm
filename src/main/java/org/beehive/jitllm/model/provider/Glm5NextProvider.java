package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.Glm5NextModelLoader;

/** The {@code glm5next} family: GLM-5.3-Flash's text backbone. */
public final class Glm5NextProvider extends FamilyProvider {

    public Glm5NextProvider() {
        super("glm5next");
    }

    @Override
    protected Glm5NextModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new Glm5NextModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

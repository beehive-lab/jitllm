package org.beehive.jitllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Glm5NextModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/** The {@code glm5next} family: GLM-5.3-Flash's text backbone. */
public final class Glm5NextProvider extends FamilyProvider {

    public Glm5NextProvider() {
        super("glm5next");
    }

    @Override
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        FileChannel channel = source.gguf().getFileChannel();
        return new Glm5NextModelLoader(
                        channel, source.gguf(), contextLength, !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

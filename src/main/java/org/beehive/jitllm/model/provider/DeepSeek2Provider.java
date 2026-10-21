package org.beehive.jitllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.DeepSeek2ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/** The {@code deepseek2} family: latent attention with a mixture of experts (GLM-4.7-Flash). */
public final class DeepSeek2Provider extends FamilyProvider {

    public DeepSeek2Provider() {
        super("deepseek2");
    }

    @Override
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        FileChannel channel = source.gguf().getFileChannel();
        return new DeepSeek2ModelLoader(
                        channel, source.gguf(), contextLength, !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

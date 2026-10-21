package org.beehive.jitllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Qwen35MoeModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

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
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        FileChannel channel = source.gguf().getFileChannel();
        return new Qwen35MoeModelLoader(
                        channel, source.gguf(), contextLength, !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

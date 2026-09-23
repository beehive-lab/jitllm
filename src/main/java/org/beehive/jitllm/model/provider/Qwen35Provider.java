package org.beehive.jitllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Qwen35ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/**
 * The {@code qwen35} architecture: Qwen3.5, Qwen3.6 and Qwen3.8, which share one hybrid stack.
 *
 * <p>Recognition needs no case in {@link GgufRecognition}. These files declare {@code
 * general.architecture = qwen35} and nothing else claims that name, so the default branch — an
 * architecture that simply means itself — already resolves it. That is what adding a family with
 * one class and one service line is supposed to look like.
 */
public final class Qwen35Provider extends FamilyProvider {

    public Qwen35Provider() {
        super("qwen35");
    }

    @Override
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        FileChannel channel = source.gguf().getFileChannel();
        return new Qwen35ModelLoader(
                        channel, source.gguf(), contextLength, !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

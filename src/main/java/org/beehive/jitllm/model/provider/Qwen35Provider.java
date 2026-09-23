package org.beehive.jllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jllm.format.ModelSource;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.Qwen35ModelLoader;
import org.beehive.jllm.runtime.backend.BackendId;

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

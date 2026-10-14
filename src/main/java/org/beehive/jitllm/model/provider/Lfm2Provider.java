package org.beehive.jitllm.model.provider;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Lfm2ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/**
 * The {@code lfm2moe} architecture: LFM2-8B-A1B and LFM2.5-24B-A2B.
 *
 * <p>Recognition needs no case in {@link GgufRecognition}: the files declare {@code
 * general.architecture = lfm2moe}, and nothing else claims that name.
 */
public final class Lfm2Provider extends FamilyProvider {

    public Lfm2Provider() {
        super("lfm2moe");
    }

    @Override
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        FileChannel channel = source.gguf().getFileChannel();
        return new Lfm2ModelLoader(
                        channel, source.gguf(), contextLength, !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

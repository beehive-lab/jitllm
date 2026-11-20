package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.Lfm2ModelLoader;

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
    protected Lfm2ModelLoader loader(ModelSource source, int contextLength, boolean useTornadovm) {
        return new Lfm2ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}

package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Gemma4ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/**
 * Gemma-4. Declares {@code gemma4} in {@code general.architecture} (verified against
 * gemma-4-E2B-it-Q8_0.gguf, which also names itself "Gemma-4-E2B-It"), and its weight set —
 * per-layer token embeddings, dual RoPE tables for sliding-window and full attention — is its own,
 * so no other loader can read it.
 */
public final class Gemma4Provider extends FamilyProvider {
    public Gemma4Provider() {
        super("gemma4");
    }

    @Override
    public Model load(ModelSource source, BackendId backend, int contextLength) {
        return new Gemma4ModelLoader(
                        source.gguf().getFileChannel(),
                        source.gguf(),
                        contextLength,
                        !BackendId.CPU.equals(backend))
                .loadModel();
    }
}

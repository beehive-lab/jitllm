package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.AbstractModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * The shape every built-in family provider has: one identity, one loader, and recognition that asks
 * the file what it is.
 *
 * <p>Every provider asks {@link GgufRecognition} the same question and claims the source only when
 * the answer is its own identity. Two providers therefore cannot both claim a file — the ambiguity
 * check in {@link ModelProviders} guards against a future provider that decides for itself, not
 * against these.
 *
 * <p>Extending this is a convenience, not a requirement. The SPI is {@link ModelProvider}; a
 * provider that recognizes its files some other way implements that directly.
 */
public abstract class FamilyProvider implements ModelProvider {

    private final ArchitectureId architecture;

    protected FamilyProvider(String architecture) {
        this.architecture = ArchitectureId.of(architecture);
    }

    @Override
    public boolean supports(ModelSource source) {
        return architecture.equals(GgufRecognition.architectureOf(source));
    }

    @Override
    public final ArchitectureId architecture(ModelSource source) {
        return architecture;
    }

    @Override
    public final String name() {
        return architecture + " provider";
    }

    /**
     * This family's loader for {@code source}.
     *
     * @param useTornadovm whether the weights are materialized for the accelerator
     */
    protected abstract AbstractModelLoader<?, ?> loader(
            ModelSource source, int contextLength, boolean useTornadovm);

    @Override
    public final Model load(ModelSource source, BackendId backend, int contextLength) {
        return loader(source, contextLength, !BackendId.CPU.equals(backend)).loadModel();
    }

    /** From the metadata alone: no tokenizer, tensor or weight is built. */
    @Override
    public final Configuration configuration(ModelSource source, int contextLength) {
        return loader(source, contextLength, false).loadConfiguration();
    }
}

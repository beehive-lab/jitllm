package org.beehive.jitllm.model.provider;

import java.io.IOException;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Recognizes a model source and loads it.
 *
 * <p>Providers are discovered with {@link java.util.ServiceLoader}, so adding an architecture means
 * adding a provider and one registration line, not editing a central switch (<a
 * href="./././././././docs/architecture/architecture.md">Rule 15</a>).
 *
 * <h2>Recognition</h2>
 *
 * <p>{@link #supports} is asked first and must be cheap and certain: it sees the metadata, not the
 * weights. Two providers claiming one source is an <b>error</b> that names both, not a race decided
 * by classpath order — see {@link ModelProviders#select}.
 */
public interface ModelProvider {

    /** Whether this provider handles the source. Metadata only; must not read tensor data. */
    boolean supports(ModelSource source);

    /**
     * The architecture identity for a source this provider supports — chosen here, once, and used
     * downstream instead of being re-derived.
     */
    ArchitectureId architecture(ModelSource source);

    /**
     * Loads the model.
     *
     * @param backend where it will execute. {@link BackendId#CPU} is a real choice, not a fallback
     *     any other backend means the accelerator path
     * @param contextLength tokens, or a negative value for the model's own — the loaders' existing
     *     convention, kept rather than reinvented
     * @throws IOException if the file cannot be read
     */
    Model load(ModelSource source, BackendId backend, int contextLength) throws IOException;

    /**
     * The configuration a {@link #load} would give the model, without loading it: what the memory
     * preflight needs before anything is materialized.
     *
     * <p>The default loads the model on the host and keeps its configuration, which is correct and
     * slow. A provider that can read the configuration from the metadata alone overrides it.
     *
     * @param contextLength as for {@link #load}
     * @throws IOException if the file cannot be read
     */
    default Configuration configuration(ModelSource source, int contextLength) throws IOException {
        return load(source, BackendId.CPU, contextLength).configuration();
    }

    /** For diagnostics: which provider answered, in a message a user can act on. */
    default String name() {
        return getClass().getSimpleName();
    }
}

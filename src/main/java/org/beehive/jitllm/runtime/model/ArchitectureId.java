package org.beehive.jitllm.runtime.model;

import java.util.Locale;
import java.util.Objects;

/**
 * Which architecture a loaded model is — backend-neutral, and chosen exactly once.
 *
 * <p>The provider that recognizes a source selects the identity, and everything downstream uses
 * <i>that</i> rather than re-deriving it from metadata. Two registries guessing independently is
 * the failure this prevents: a loader that decides "llama" and a program factory that decides
 * "qwen2" from the same file produce a model that loads and computes nonsense.
 *
 * <p>A value, not an enum: adding an architecture must not mean editing a central list ({@link
 * org.beehive.jitllm.model.ModelType} is that list, and Rule 15 is why it is going).
 *
 * <h2>Why it lives in {@code runtime}</h2>
 */
public final class ArchitectureId {

    private final String name;

    private ArchitectureId(String name) {
        this.name = name;
    }

    /**
     * @param name lower-case identifier, e.g. {@code "llama"}, {@code "qwen3"}, {@code "phi3"}
     */
    public static ArchitectureId of(String name) {
        Objects.requireNonNull(name, "name");
        String normalized = name.trim().toLowerCase(Locale.ROOT);
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("an architecture identity needs a name");
        }
        return new ArchitectureId(normalized);
    }

    public String name() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof ArchitectureId id && name.equals(id.name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return name;
    }
}

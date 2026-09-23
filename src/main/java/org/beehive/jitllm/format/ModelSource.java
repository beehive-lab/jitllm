package org.beehive.jitllm.format;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * A model file, opened far enough to be recognized but not yet loaded.
 *
 * <p>What a provider is offered when it is asked {@code supports(.)}: the metadata, and the handle
 * needed to read the rest if it says yes. Recognition therefore costs one metadata parse for the
 * whole discovery pass rather than one per provider — providers are asked in turn, and a source
 * that re-read the file for each of them would make adding a provider cost a file read.
 *
 * <p>Lives in the format layer because that is what it is: a GGUF file. When a second format
 * arrives, this is the type that grows a sibling, not the SPI.
 */
public final class ModelSource {

    private final Path path;
    private final GGUF gguf;
    private final Map<String, Object> metadata;

    private ModelSource(Path path, GGUF gguf, Map<String, Object> metadata) {
        this.path = path;
        this.gguf = gguf;
        this.metadata = metadata;
    }

    /** Reads the metadata; the tensor data is not touched. */
    public static ModelSource ofFile(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        GGUF gguf = GGUF.loadGGUFMetadata(path);
        return new ModelSource(path, gguf, gguf.getMetadata());
    }

    /** For callers that already hold a parsed file. */
    public static ModelSource of(Path path, GGUF gguf) {
        Objects.requireNonNull(gguf, "gguf");
        return new ModelSource(Objects.requireNonNull(path, "path"), gguf, gguf.getMetadata());
    }

    /**
     * A source that carries metadata and no file.
     *
     * <p>Recognition needs only the metadata, so this is what a test — or a caller probing a
     * catalogue of models — works with. {@link #gguf()} refuses rather than returning null: a
     * provider that tried to load one of these has made a mistake worth hearing about.
     */
    public static ModelSource ofMetadata(Path path, Map<String, Object> metadata) {
        return new ModelSource(
                Objects.requireNonNull(path, "path"),
                null,
                Map.copyOf(Objects.requireNonNull(metadata, "metadata")));
    }

    public Path path() {
        return path;
    }

    /** The file's metadata — what a provider recognizes a model by. */
    public Map<String, Object> metadata() {
        return metadata;
    }

    /** Whether this source can be loaded, or only recognized. */
    public boolean isLoadable() {
        return gguf != null;
    }

    /**
     * The parsed file, for the provider that goes on to load it.
     *
     * @throws IllegalStateException if this source carries metadata only
     */
    public GGUF gguf() {
        if (gguf == null) {
            throw new IllegalStateException(
                    path
                            + " carries metadata only and cannot be loaded;"
                            + " it was created for recognition");
        }
        return gguf;
    }

    /** A metadata string, or {@code null} — the common shape of a recognition check. */
    public String metadataString(String key) {
        Object value = metadata().get(key);
        return value instanceof String text ? text : null;
    }

    @Override
    public String toString() {
        return path.getFileName() + " (" + metadata().size() + " metadata entries)";
    }
}

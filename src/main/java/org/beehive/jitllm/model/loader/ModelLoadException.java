package org.beehive.jllm.model.loader;

/** Exception thrown when model loading fails. */
public class ModelLoadException extends RuntimeException {

    public ModelLoadException(String message) {
        super(message);
    }

    public ModelLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}

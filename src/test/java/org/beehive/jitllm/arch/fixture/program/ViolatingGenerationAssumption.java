package org.beehive.jllm.arch.fixture.program;

import org.beehive.jllm.tokenizer.Tokenizer;

/**
 * Rule 14's fixture: a core-layer type that requires a tokenizer.
 *
 * <p>A program component that cannot be constructed without one has decided that every model
 * generates text, which is the assumption Rule 14 exists to keep out of the core.
 */
public final class ViolatingGenerationAssumption {

    private final Tokenizer tokenizer;

    public ViolatingGenerationAssumption(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }
}

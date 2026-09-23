package org.beehive.jitllm.arch.fixture.runtime;

import org.beehive.jitllm.format.GGMLType;

/**
 * Deliberate Rule 4 violation of the <i>format</i> kind: a runtime-layer type naming a file-format
 * type. The backend half of the rule is covered by {@code fixture.model.ViolatingModel}, which
 * names TornadoVM types; without this one, the format half would never be seen to fail.
 */
public class ViolatingRuntimeVocabulary {

    private final GGMLType fileType;

    public ViolatingRuntimeVocabulary(GGMLType fileType) {
        this.fileType = fileType;
    }

    public GGMLType fileType() {
        return fileType;
    }
}

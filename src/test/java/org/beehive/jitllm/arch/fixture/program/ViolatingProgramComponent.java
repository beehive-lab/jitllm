package org.beehive.jllm.arch.fixture.program;

import uk.ac.manchester.tornado.api.TaskGraph;

/** Rule 3's fixture: a "program description" that names a TornadoVM type. */
public final class ViolatingProgramComponent {

    private final TaskGraph graph;

    public ViolatingProgramComponent(TaskGraph graph) {
        this.graph = graph;
    }

    public TaskGraph graph() {
        return graph;
    }
}

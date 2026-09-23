package org.beehive.jitllm.program;

import java.util.List;
import java.util.Objects;

/**
 * A backend-neutral description of one forward pass.
 *
 * <p>Which operations, over which weights, in what order, producing which outputs — and nothing
 * else. <b>No device handles, no task graphs, no TornadoVM types, no GGUF types.</b> It is data: it
 * can be inspected, logged and compared.
 *
 * <p>A backend lowers it. That lowering is <b>many-to-many and backend-owned</b>: one operation may
 * become several tasks, several operations may fuse into one, and one operation may become
 * different task sets depending on compile options or device capability. The TornadoVM builders
 * already do all three, which is why lowering is written per backend rather than derived from rules
 *
 * @param signature what this program is, as a value — and its cache key
 * @param components the components, in order; the same list the signature carries
 */
public record InferenceProgram(ProgramSignature signature, List<ProgramComponent> components) {

    public InferenceProgram {
        Objects.requireNonNull(signature, "signature");
        components = List.copyOf(Objects.requireNonNull(components, "components"));
        if (!components.equals(signature.components())) {
            throw new IllegalArgumentException(
                    "a program's components must be the ones its" + " signature describes");
        }
    }

    /** Builds a program from a signature, taking its components. */
    public static InferenceProgram of(ProgramSignature signature) {
        return new InferenceProgram(signature, signature.components());
    }

    /**
     * The components {@code phase} runs, in order.
     *
     * @throws IllegalArgumentException if this program does not declare that phase
     */
    public List<ProgramComponent> componentsFor(PhaseId phase) {
        Objects.requireNonNull(phase, "phase");
        for (PhaseSelection selection : signature.phases()) {
            if (selection.phase() == phase) {
                return selection.componentIndices().stream().map(components::get).toList();
            }
        }
        throw new IllegalArgumentException(
                "this program has no "
                        + phase
                        + " phase; it declares "
                        + signature.phases().stream().map(PhaseSelection::phase).toList());
    }
}

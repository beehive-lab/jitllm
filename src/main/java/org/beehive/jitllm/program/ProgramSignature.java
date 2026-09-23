package org.beehive.jitllm.program;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * What a program is, as a value — and therefore what makes two compiled programs the same one.
 *
 * <p>It is the compiled-program cache key's largest component, which turns value-comparability from
 * a style preference into a correctness requirement: an accidental identity comparison is a silent
 * cache miss that recompiles the same program, or worse a claimed match that is not one. Three
 * rules make it real, all enforced at construction:
 *
 * <ul>
 *   <li><b>No Java arrays</b>, at any depth. Arrays compare by identity.
 *   <li><b>Defensive copies</b> of every collection, so a signature handed a caller's mutable list
 *       does not change when that list does.
 *   <li><b>Canonical ordering</b>, validated here — bindings by index, phases in {@link PhaseId}
 *       order, component indices ascending within a phase. Two semantically identical signatures
 *       cannot differ because one collection was built in a different iteration order.
 * </ul>
 *
 * <h2>What is deliberately absent</h2>
 *
 * <p><b>No device handles, no TornadoVM types, no format types.</b> A signature that held handles
 * could not be compared, logged, or used as a key (Rule 3).
 *
 * @param architecture the architecture identity. Held as a {@link ArchitectureId}: it was a bare
 *     {@code String} only because the type lived in {@code model.provider}, which Rule 9 does not
 *     let {@code program.**} reach. Moving it to {@code runtime.model} removed the reason, and with
 *     it the chance of two layers disagreeing about how an architecture is spelled
 * @param policy the execution policy the program was described under, as a stable descriptor; a
 *     value type will replace this
 * @param capacity what the device arrays were sized from
 * @param components the program's components, in order
 * @param phases one selection per phase, in {@link PhaseId} order
 * @param bindings the binding surface, in index order
 */
public record ProgramSignature(
        ArchitectureId architecture,
        String policy,
        CapacityShape capacity,
        List<ProgramComponent> components,
        List<PhaseSelection> phases,
        List<BindingEntry> bindings) {

    public ProgramSignature {
        Objects.requireNonNull(architecture, "architecture");
        Objects.requireNonNull(policy, "policy");
        Objects.requireNonNull(capacity, "capacity");
        components = List.copyOf(Objects.requireNonNull(components, "components"));
        phases = List.copyOf(Objects.requireNonNull(phases, "phases"));
        bindings = List.copyOf(Objects.requireNonNull(bindings, "bindings"));

        if (components.isEmpty()) {
            throw new IllegalArgumentException("a program must declare at least one component");
        }
        if (phases.isEmpty()) {
            throw new IllegalArgumentException("a program must declare at least one phase");
        }

        // Canonical form: bindings in index order, indices dense from zero.
        for (int i = 0; i < bindings.size(); i++) {
            if (bindings.get(i).index() != i) {
                throw new IllegalArgumentException(
                        "bindings must be in index order and dense from"
                                + " zero; entry "
                                + i
                                + " declares index "
                                + bindings.get(i).index());
            }
        }

        // Canonical form: phases in PhaseId order, each declared once.
        PhaseId previous = null;
        for (PhaseSelection selection : phases) {
            if (previous != null && selection.phase().compareTo(previous) <= 0) {
                throw new IllegalArgumentException(
                        "phases must be declared once each, in PhaseId"
                                + " order; found "
                                + selection.phase()
                                + " after "
                                + previous);
            }
            previous = selection.phase();
        }

        // Every selection names components this program declared, and every component participates.
        for (PhaseSelection selection : phases) {
            for (Integer index : selection.componentIndices()) {
                if (index >= components.size()) {
                    throw new IllegalArgumentException(
                            "phase "
                                    + selection.phase()
                                    + " selects"
                                    + " component "
                                    + index
                                    + ", but the program declares only "
                                    + components.size());
                }
                ProgramComponent component = components.get(index);
                if (!component.phases().contains(selection.phase())) {
                    throw new IllegalArgumentException(
                            "phase "
                                    + selection.phase()
                                    + " selects "
                                    + component.name()
                                    + ", which does not declare that phase");
                }
            }
        }
    }

    /**
     * The bindings whose identity is fixed for the life of the compiled program.
     *
     * <p>Everything a captured graph holds — weights, key/value storage, workspace, control and
     * result arrays alike.
     */
    public List<BindingEntry.ProgramFixed> programFixed() {
        return bindings.stream()
                .filter(BindingEntry.ProgramFixed.class::isInstance)
                .map(BindingEntry.ProgramFixed.class::cast)
                .toList();
    }

    /** The scalars an invocation supplies, by writing into a carrier. */
    public List<BindingEntry.InvocationValue> invocationValues() {
        return bindings.stream()
                .filter(BindingEntry.InvocationValue.class::isInstance)
                .map(BindingEntry.InvocationValue.class::cast)
                .toList();
    }

    /** What the caller may read once an invocation completes. */
    public List<BindingEntry.HostVisibleResult> results() {
        return bindings.stream()
                .filter(BindingEntry.HostVisibleResult.class::isInstance)
                .map(BindingEntry.HostVisibleResult.class::cast)
                .toList();
    }
}

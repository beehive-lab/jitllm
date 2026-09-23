package org.beehive.jitllm.backend.tornado.lowering;

import java.util.List;
import java.util.Set;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.op.EmbeddingLookup;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.OperationKind;
import org.beehive.jitllm.program.op.VocabProjection;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The checks every single-token decoder program shares, and nothing a family owns.
 *
 * <p>The skeleton is genuinely common: an embedding, some number of layer composites, a final norm,
 * a vocabulary projection and optionally a device sample; prefill skipping the tail that decode
 * runs; one weight representation throughout. <b>The per-layer sequence is not here</b>, and must
 * not be — that is the part where families actually differ, and a shared version of it would be a
 * rule table by another name.
 */
final class ProgramShape {

    /** The representations this backend has single-token task graphs for. */
    static final Set<DataType> SUPPORTED_WEIGHTS = Set.of(DataType.F16, DataType.Q8_0);

    /**
     * What follows the vocabulary projection.
     *
     * <p>An enumeration of the two tails that exist, not a general grammar: Granite scales its
     * logits and nothing else does yet. A third family adds a case, and if a fourth needs something
     * this cannot say, that is the point at which a shared skeleton has stopped paying for itself.
     */
    enum TailShape {
        /** Projection, then optionally the device sample. */
        PLAIN,
        /** Projection, a {@code Scale} on the logits, then optionally the device sample. */
        SCALED_LOGITS
    }

    private ProgramShape() {}

    /** The common case: no scale after the embedding, no scale on the logits. */
    static int validateSkeleton(
            String family,
            InferenceProgram program,
            ArchitectureId architecture,
            LayerValidator layerValidator) {
        return validateSkeleton(
                family, program, architecture, layerValidator, TailShape.PLAIN, false);
    }

    /**
     * Checks the skeleton and returns the program's weight representation.
     *
     * @param layerValidator applied to each layer composite, with the weight representation
     * @return how many layer composites were found
     */
    static int validateSkeleton(
            String family,
            InferenceProgram program,
            ArchitectureId architecture,
            LayerValidator layerValidator,
            TailShape tail,
            boolean scaledEmbedding) {
        var signature = program.signature();
        if (!architecture.equals(signature.architecture())) {
            throw new UnsupportedProgramException(
                    family, "architecture", architecture.name(), signature.architecture().name());
        }
        if (!signature.policy().startsWith("single-token")) {
            throw new UnsupportedProgramException(
                    family, "policy", "single-token…", signature.policy());
        }

        List<ProgramComponent> components = program.components();
        if (components.size() < 3) {
            throw new UnsupportedProgramException(
                    family, "component count", "at least 3", String.valueOf(components.size()));
        }

        // Prefill must skip the tail; decode must run everything. Anything else is not this shape.
        List<ProgramComponent> prefill = program.componentsFor(PhaseId.PREFILL);
        List<ProgramComponent> decode = program.componentsFor(PhaseId.DECODE);
        if (decode.size() != components.size()) {
            throw new UnsupportedProgramException(
                    family,
                    "decode phase",
                    "every component",
                    decode.size() + " of " + components.size());
        }
        if (prefill.size() >= decode.size()) {
            throw new UnsupportedProgramException(
                    family,
                    "prefill phase",
                    "fewer components than decode — it skips the projection",
                    prefill.size() + " against " + decode.size());
        }

        expectLeaf(
                family, components.getFirst(), OperationKind.EMBEDDING_LOOKUP, "first component");
        DataType weights = ((EmbeddingLookup) leafOperation(components.getFirst())).dataType();
        if (!SUPPORTED_WEIGHTS.contains(weights)) {
            throw new UnsupportedProgramException(
                    family,
                    "embedding representation",
                    "one of " + SUPPORTED_WEIGHTS,
                    weights.name());
        }

        int firstLayer = 1;
        if (scaledEmbedding) {
            expectLeaf(family, components.get(1), OperationKind.SCALE, "embedding scale");
            firstLayer = 2;
        }

        int layers = 0;
        for (int i = firstLayer; i < components.size(); i++) {
            ProgramComponent component = components.get(i);
            if (component instanceof ProgramComponent.Composite composite) {
                layerValidator.validate(composite, weights);
                layers++;
            } else {
                break;
            }
        }
        if (layers == 0) {
            throw new UnsupportedProgramException(
                    family, "layers", "at least one layer composite", "none");
        }

        int tailIndex = firstLayer + layers;
        expectLeaf(family, components.get(tailIndex), OperationKind.RMS_NORM, "final norm");
        expectLeaf(
                family,
                components.get(tailIndex + 1),
                OperationKind.VOCAB_PROJECTION,
                "vocabulary projection");
        expectWeightRepresentation(
                family,
                "vocabulary projection",
                ((VocabProjection) leafOperation(components.get(tailIndex + 1))).dataType(),
                weights);
        int next = tailIndex + 2;
        if (tail == TailShape.SCALED_LOGITS) {
            if (components.size() <= next) {
                throw new UnsupportedProgramException(
                        family, "logit scale", "a SCALE after the projection", "nothing");
            }
            expectLeaf(family, components.get(next), OperationKind.SCALE, "logit scale");
            next++;
        }
        if (components.size() > next) {
            expectLeaf(family, components.get(next), OperationKind.ARG_MAX, "device sampling");
            if (components.size() > next + 1) {
                throw new UnsupportedProgramException(
                        family,
                        "trailing components",
                        "none after sampling",
                        String.valueOf(components.size() - next - 1));
            }
        }
        return layers;
    }

    /**
     * One representation for the whole program: a mixed one names a program with no task graphs.
     */
    static void expectWeightRepresentation(
            String family, String what, DataType found, DataType expected) {
        if (found != expected) {
            throw new UnsupportedProgramException(
                    family, what + " weight representation", expected.name(), found.name());
        }
    }

    static void expectLeaf(
            String family, ProgramComponent component, OperationKind kind, String what) {
        if (!(component instanceof ProgramComponent.Leaf leaf)) {
            throw new UnsupportedProgramException(
                    family, what, kind.name(), "a composite named " + component.name());
        }
        if (leaf.operation().kind() != kind) {
            throw new UnsupportedProgramException(
                    family, what, kind.name(), leaf.operation().kind().name());
        }
    }

    /** The children of a layer, checked against the exact kinds a family performs, in order. */
    static void expectLayerSequence(
            String family, ProgramComponent.Composite layer, OperationKind[] expected) {
        List<ProgramComponent> inner = layer.children();
        if (inner.size() != expected.length) {
            throw new UnsupportedProgramException(
                    family,
                    layer.name() + " component count",
                    String.valueOf(expected.length),
                    String.valueOf(inner.size()));
        }
        for (int i = 0; i < expected.length; i++) {
            expectLeaf(family, inner.get(i), expected[i], layer.name() + " component " + i);
        }
    }

    static Operation leafOperation(ProgramComponent component) {
        return ((ProgramComponent.Leaf) component).operation();
    }

    @FunctionalInterface
    interface LayerValidator {
        void validate(ProgramComponent.Composite layer, DataType weights);
    }
}

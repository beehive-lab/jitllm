package org.beehive.jllm.model.architecture;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Both failures are the kind that otherwise produce a working-looking model that computes the wrong
 * thing, so both are asserted to fail <b>by name</b> rather than merely to fail.
 */
public class ModelArchitecturesTest {

    /** Two implementations claiming one identity is a defect, and the message names both. */
    @Test
    public void duplicateIdentitiesFailNamingBothImplementations() {
        List<ModelArchitecture> duplicated = List.of(new Stub("stub"), new OtherStub("stub"));
        try {
            ModelArchitectures.select(ArchitectureId.of("stub"), duplicated);
            fail("two architectures claiming one identity must be refused");
        } catch (IllegalStateException expected) {
            assertTrue(
                    "the message must name the identity: " + expected.getMessage(),
                    expected.getMessage().contains("'stub'"));
            assertTrue(
                    "and both implementations: " + expected.getMessage(),
                    expected.getMessage().contains(Stub.class.getName())
                            && expected.getMessage().contains(OtherStub.class.getName()));
        }
    }

    /** A missing architecture fails by identity, and says what is registered. */
    @Test
    public void aMissingArchitectureFailsByIdentity() {
        try {
            ModelArchitectures.select(ArchitectureId.of("gemma4"), List.of(new Stub("stub")));
            fail("an unregistered identity must be refused");
        } catch (IllegalStateException expected) {
            assertTrue(
                    "the message must name what was asked for: " + expected.getMessage(),
                    expected.getMessage().contains("'gemma4'"));
            assertTrue(
                    "and what is registered: " + expected.getMessage(),
                    expected.getMessage().contains("stub"));
        }
    }

    /** Asking whether something is described must not throw — four families are not. */
    @Test
    public void absenceIsAnAnswerNotAnError() {
        List<ModelArchitecture> discovered = ModelArchitectures.discover();
        assertTrue(
                "llama is described",
                ModelArchitectures.isDescribed(ArchitectureId.of("llama"), discovered));
        assertFalse(
                "gemma4 is not, and asking is not an error",
                ModelArchitectures.isDescribed(ArchitectureId.of("gemma4"), discovered));
        assertFalse(
                "nor is qwen2-moe",
                ModelArchitectures.isDescribed(ArchitectureId.of("qwen2-moe"), discovered));
        assertFalse(
                "nor devstral",
                ModelArchitectures.isDescribed(ArchitectureId.of("devstral"), discovered));
    }

    /** The seven registered architectures are discovered, and none collides. */
    @Test
    public void theRegisteredArchitecturesAreDiscoveredAndDistinct() {
        List<ModelArchitecture> discovered = ModelArchitectures.discover();
        Set<ArchitectureId> ids = new java.util.LinkedHashSet<>();
        for (ModelArchitecture architecture : discovered) {
            assertTrue("two architectures claim " + architecture.id(), ids.add(architecture.id()));
        }
        assertTrue(
                "expected the seven described architectures, found " + ids,
                ids.containsAll(
                        List.of(
                                ArchitectureId.of("llama"),
                                ArchitectureId.of("mistral"),
                                ArchitectureId.of("qwen2"),
                                ArchitectureId.of("deepseek-r1-distill-qwen"),
                                ArchitectureId.of("qwen3"),
                                ArchitectureId.of("granite"),
                                ArchitectureId.of("phi3"))));
    }

    /**
     * An alias shares the computation and keeps its own identity — so the two are not one program.
     */
    @Test
    public void anAliasSharesTheComputationButNotTheIdentity() {
        var config = new LlamaConfiguration("FP16", 64, 128, 2, 4, 2, 48, 32, 1e-5f, 500000f);
        var inputs =
                new ArchitectureInputs(
                        config, DataType.F16, DataType.F32, ExecutionPolicy.builder().build());

        InferenceProgram llama = new LlamaArchitecture().describe(inputs);
        InferenceProgram mistral =
                new MistralArchitecture()
                        .describe(
                                new ArchitectureInputs(
                                        mistralConfig(),
                                        DataType.F16,
                                        DataType.F32,
                                        ExecutionPolicy.builder().build()));

        assertEquals("the same operations", kinds(llama), kinds(mistral));
        assertNotEquals("but not the same program", llama.signature(), mistral.signature());
        assertEquals(ArchitectureId.of("llama"), llama.signature().architecture());
        assertEquals(ArchitectureId.of("mistral"), mistral.signature().architecture());
    }

    /** An architecture refuses a configuration that is not its shape, naming what it got. */
    @Test
    public void theConfigurationShapeIsValidated() {
        try {
            new Qwen3Architecture()
                    .validateConfiguration(
                            new LlamaConfiguration(
                                    "FP16", 64, 128, 2, 4, 2, 48, 32, 1e-5f, 500000f));
            fail("qwen3 must refuse a Llama configuration");
        } catch (IllegalArgumentException expected) {
            assertTrue(
                    expected.getMessage(),
                    expected.getMessage().contains("Qwen3Configuration")
                            && expected.getMessage().contains("LlamaConfiguration"));
        }
    }

    private static org.beehive.jllm.model.mistral.MistralConfiguration mistralConfig() {
        return new org.beehive.jllm.model.mistral.MistralConfiguration(
                "FP16", 64, 128, 2, 4, 2, 48, 32, false, 1e-5f, 500000f);
    }

    private static List<String> kinds(InferenceProgram program) {
        return program.components().stream()
                .map(c -> c.getClass().getSimpleName() + ":" + c.name())
                .toList();
    }

    private record Stub(String name) implements ModelArchitecture {
        @Override
        public ArchitectureId id() {
            return ArchitectureId.of(name);
        }

        @Override
        public void validateConfiguration(Configuration configuration) {}

        @Override
        public Set<PhaseId> logicalPhases() {
            return EnumSet.allOf(PhaseId.class);
        }

        @Override
        public InferenceProgram describe(ArchitectureInputs inputs) {
            throw new UnsupportedOperationException("a stub describes nothing");
        }
    }

    private record OtherStub(String name) implements ModelArchitecture {
        @Override
        public ArchitectureId id() {
            return ArchitectureId.of(name);
        }

        @Override
        public void validateConfiguration(Configuration configuration) {}

        @Override
        public Set<PhaseId> logicalPhases() {
            return EnumSet.allOf(PhaseId.class);
        }

        @Override
        public InferenceProgram describe(ArchitectureInputs inputs) {
            throw new UnsupportedOperationException("a stub describes nothing");
        }
    }
}

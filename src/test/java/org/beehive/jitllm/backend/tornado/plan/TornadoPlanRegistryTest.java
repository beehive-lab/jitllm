package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Set;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Families that have not migrated must be <b>absent</b>, not present-and-empty: absence is what
 * makes {@code ForwardPlanFactory} keep answering for them, and a provider declaring no modes would
 * silently take that answer away.
 */
public class TornadoPlanRegistryTest {

    @Test
    public void theMigratedProvidersDeclareTheInventoriedMatrix() {
        var llama = provider("llama");
        assertEquals(
                "both representations, plus the Q4_0 it retains rather than materializes",
                Set.of(DataType.F16, DataType.Q8_0, DataType.Q4_0),
                llama.supportedDataTypes());
        // Declared for the family, not per representation: Q4_0 has single-token kernels only, and
        // the registry refuses the other two modes for it by name rather than on a cast.
        assertEquals(
                "all three plan shapes", Set.of(ExecutionMode.values()), llama.supportedModes());

        var mistral = provider("mistral");
        assertEquals(
                "both representations",
                Set.of(DataType.F16, DataType.Q8_0),
                mistral.supportedDataTypes());
        assertEquals(
                "STANDARD only, as the factory has always enforced",
                Set.of(ExecutionMode.STANDARD),
                mistral.supportedModes());
    }

    /**
     * A STANDARD-only family declares exactly that, for every such family rather than one of them.
     *
     * <p>Devstral's {@code PREFILL_DECODE not yet supported for DEVSTRAL_2 + Q4_K} was read as a
     * possible policy-resolution or diagnostic defect — a request for PREFILL_DECODE coming back
     * labelled BATCH_PREFILL_DECODE. It is not: each mode reaches the registry through its own
     * factory entry point and the refusal names the mode that was actually requested. This pins
     * that, so a future change cannot quietly make one mode answer for another.
     *
     * <p>It also pins the scope. STANDARD_ONLY is a family-level declaration shared by Devstral,
     * Qwen2 and Mistral; it is not specific to Devstral, and not specific to Q4_K — the dtype
     * appears in the message text only because the message names the whole tuple.
     */
    @Test
    public void everyStandardOnlyFamilyDeclaresStandardOnly() {
        for (String architecture : Set.of("devstral", "qwen2", "mistral")) {
            assertEquals(
                    architecture + " is STANDARD-only",
                    Set.of(ExecutionMode.STANDARD),
                    provider(architecture).supportedModes());
        }
    }

    /** Every architecture the factory used to dispatch on now has a provider. */
    @Test
    public void everyArchitectureInTheMatrixIsRegistered() {
        var registered = TornadoPlanRegistry.registered();
        for (String architecture :
                new String[] {
                    "llama",
                    "mistral",
                    "devstral",
                    "qwen2",
                    "deepseek-r1-distill-qwen",
                    "qwen2-moe",
                    "qwen3",
                    "gemma4",
                    "phi3",
                    "granite",
                    "qwen35"
                }) {
            assertTrue(
                    architecture
                            + " must resolve through a provider now that the factory's"
                            + " switches are gone",
                    registered.contains(ArchitectureId.of(architecture)));
        }
        assertEquals("the matrix has eleven architectures", 11, registered.size());
    }

    /**
     * Qwen2-MoE: registered, {@code Q8_0} only, and {@code F16} is a refusal rather than a gap.
     *
     * <p>The shape that made the registry's protocol matter. If an unsupported dtype returned an
     * empty result, this family would have looked unmigrated and — while the switch still existed —
     * the switch would have answered for it. Now there is no switch, so an empty result here would
     * surface as "no provider registered", which is a different and false statement.
     */
    @Test
    public void aRegisteredProviderWithAnUnsupportedDtypeIsNotAnUnmigratedArchitecture() {
        var moe = provider("qwen2-moe");
        assertEquals("Q8_0 only", Set.of(DataType.Q8_0), moe.supportedDataTypes());
        assertEquals(
                "STANDARD and batch prefill, but not sequential prefill",
                Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE),
                moe.supportedModes());
        assertTrue(
                "it is registered, which is what distinguishes refusal from absence",
                TornadoPlanRegistry.registered().contains(ArchitectureId.of("qwen2-moe")));
    }

    /**
     * Admission and per-tensor native support are separate declarations.
     *
     * <p>For a family whose model is one representation throughout they agree, and the default says
     * so. They stop agreeing the moment a model is mixed: it reports one representation and holds
     * several, so a memory prediction built from the admission set counts every tensor that is not
     * the model's representation at the wrong size. This pins that the preflight reads the
     * per-tensor declaration and not the admission one.
     */
    @Test
    public void perTensorNativeSupportIsDeclaredApartFromAdmission() {
        for (TornadoPlanProvider provider : TornadoPlanRegistry.discover()) {
            assertEquals(
                    provider.architecture() + " retains what it declares per tensor",
                    provider.nativeTensorTypes(),
                    TornadoPlanRegistry.nativeDeviceTypes(provider.architecture()));
        }

        TornadoPlanProvider uniform =
                new TornadoPlanProvider() {
                    @Override
                    public ArchitectureId architecture() {
                        return ArchitectureId.of("uniform-test");
                    }

                    @Override
                    public Set<DataType> supportedDataTypes() {
                        return Set.of(DataType.Q8_0);
                    }

                    @Override
                    public Set<ExecutionMode> supportedModes() {
                        return Set.of(ExecutionMode.STANDARD);
                    }

                    @Override
                    public org.beehive.jitllm.backend.tornado.plan.components
                                    .SingleTokenForwardPlanComponents
                            components(
                                    DataType weights,
                                    org.beehive.jitllm.inference.state.State state,
                                    org.beehive.jitllm.model.Model model) {
                        throw new UnsupportedOperationException("declaration only");
                    }
                };
        assertEquals(
                "a uniform family answers both questions the same way, without restating it",
                uniform.supportedDataTypes(),
                uniform.nativeTensorTypes());

        TornadoPlanProvider mixed =
                new TornadoPlanProvider() {
                    @Override
                    public ArchitectureId architecture() {
                        return ArchitectureId.of("mixed-test");
                    }

                    @Override
                    public Set<DataType> supportedDataTypes() {
                        return Set.of(DataType.Q4_0);
                    }

                    @Override
                    public Set<DataType> nativeTensorTypes() {
                        return Set.of(DataType.Q4_0, DataType.Q5_K, DataType.F32);
                    }

                    @Override
                    public Set<ExecutionMode> supportedModes() {
                        return Set.of(ExecutionMode.STANDARD);
                    }

                    @Override
                    public org.beehive.jitllm.backend.tornado.plan.components
                                    .SingleTokenForwardPlanComponents
                            components(
                                    DataType weights,
                                    org.beehive.jitllm.inference.state.State state,
                                    org.beehive.jitllm.model.Model model) {
                        throw new UnsupportedOperationException("declaration only");
                    }
                };
        assertTrue(
                "a mixed family reads representations it does not admit a plan for",
                mixed.nativeTensorTypes().containsAll(mixed.supportedDataTypes())
                        && mixed.nativeTensorTypes().size() > mixed.supportedDataTypes().size());
    }

    /**
     * {@code qwen35} resolves in all three modes, and reads eight representations per tensor.
     *
     * <p>The mode set is a claim the selection layer acts on: a mode declared without graphs fails
     * from inside TornadoVM, and a mode implemented but not declared is unreachable. Both are
     * pinned here because this family gained the other two modes after its single-token path
     * shipped.
     */
    @Test
    public void qwen35ResolvesInEveryMode() {
        var qwen35 = provider("qwen35");
        assertEquals(
                "single token, sequential prefill and batched prefill",
                Set.of(ExecutionMode.values()),
                qwen35.supportedModes());
        assertEquals(
                "admitted on the representation its trunk projections share",
                Set.of(DataType.Q4_0),
                qwen35.supportedDataTypes());
        assertTrue(
                "and reads five block layouts and F32 per tensor without materializing any",
                qwen35.nativeTensorTypes()
                        .containsAll(
                                Set.of(
                                        DataType.F32,
                                        DataType.Q4_0,
                                        DataType.Q4_1,
                                        DataType.Q5_K,
                                        DataType.Q6_K)));
    }

    private static TornadoPlanProvider provider(String architecture) {
        ArchitectureId id = ArchitectureId.of(architecture);
        return TornadoPlanRegistry.discover().stream()
                .filter(p -> p.architecture().equals(id))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no plan provider for " + id));
    }
}

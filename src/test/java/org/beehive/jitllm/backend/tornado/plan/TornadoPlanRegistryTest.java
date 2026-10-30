package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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
                "both representations, the Q4_0 of its Q4_0 layers, and the types a file kept on"
                        + " CUDA or Metal reports",
                Set.of(
                        DataType.F16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                llama.supportedDataTypes());
        // Declared for the family, not per representation: Q4_0 has single-token kernels only, and
        // the registry refuses the other two modes for it by name rather than on a cast.
        assertEquals(
                "all three plan shapes", Set.of(ExecutionMode.values()), llama.supportedModes());

        var mistral = provider("mistral");
        assertEquals(
                "both representations, and the types a file kept on CUDA or Metal reports",
                Set.of(
                        DataType.F16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                mistral.supportedDataTypes());
        assertEquals(
                "single-token and batched prefill/decode (Llama's layer graphs)",
                Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE),
                mistral.supportedModes());
    }

    /** Families with batched prefill but no sequential prefill/decode plan. */
    @Test
    public void batchedFamiliesDeclareStandardAndBatch() {
        for (String architecture :
                Set.of(
                        "qwen2",
                        "deepseek-r1-distill-qwen",
                        "mistral",
                        "phi3",
                        "granite",
                        "devstral")) {
            assertEquals(
                    architecture + " is STANDARD and BATCH_PREFILL_DECODE",
                    Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE),
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
                    "qwen35",
                    "qwen35moe",
                    "deepseek2"
                }) {
            assertTrue(
                    architecture
                            + " must resolve through a provider now that the factory's"
                            + " switches are gone",
                    registered.contains(ArchitectureId.of(architecture)));
        }
        assertEquals("the matrix has thirteen architectures", 13, registered.size());
    }

    /**
     * Qwen2-MoE: registered, quantized representations only, and {@code F16} is a refusal rather
     * than a gap.
     *
     * <p>The shape that made the registry's protocol matter. If an unsupported dtype returned an
     * empty result, this family would have looked unmigrated and — while the switch still existed —
     * the switch would have answered for it. Now there is no switch, so an empty result here would
     * surface as "no provider registered", which is a different and false statement.
     */
    @Test
    public void aRegisteredProviderWithAnUnsupportedDtypeIsNotAnUnmigratedArchitecture() {
        var moe = provider("qwen2-moe");
        assertEquals(
                "the quantized representations a kept load holds",
                org.beehive.jitllm.backend.tornado.lowering.TornadoSupportSets.NATIVE_QUANTIZED,
                moe.supportedDataTypes());
        assertFalse("F16 is refused", moe.supportedDataTypes().contains(DataType.F16));
        assertEquals(
                "STANDARD and batch prefill, but not sequential prefill",
                Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE),
                moe.supportedModes());
        assertTrue(
                "it is registered, which is what distinguishes refusal from absence",
                TornadoPlanRegistry.registered().contains(ArchitectureId.of("qwen2-moe")));
    }

    /**
     * {@code qwen35} resolves in all three modes.
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
                Set.of(DataType.Q4_0, DataType.Q8_0),
                qwen35.supportedDataTypes());
    }

    private static TornadoPlanProvider provider(String architecture) {
        ArchitectureId id = ArchitectureId.of(architecture);
        return TornadoPlanRegistry.discover().stream()
                .filter(p -> p.architecture().equals(id))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no plan provider for " + id));
    }
}

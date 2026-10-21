package org.beehive.jitllm.backend.tornado.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * {@link PackedWeights#loading}: a load chooses its representation for the policy it is given, not
 * for the process's properties, and neither a later load nor one on another thread sees it.
 */
public class PackedWeightsTest {

    private static final String WIDTH = "jitllm.prefillBatchSize";
    private static final String PREFILL_DECODE = "jitllm.withPrefillDecode";

    private static ExecutionPolicy batched(int width) {
        return ExecutionPolicy.builder()
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(width)
                .build();
    }

    @Test
    public void aScopedLoadSeesItsOwnPolicyOverConflictingProperties() throws Exception {
        String width = System.getProperty(WIDTH);
        String prefillDecode = System.getProperty(PREFILL_DECODE);
        System.setProperty(PREFILL_DECODE, "true");
        System.setProperty(WIDTH, "96");
        try {
            ExecutionPolicy policy = batched(128);
            assertSame(policy, PackedWeights.loading(policy, PackedWeights::loadPolicy));
            assertEquals(Optional.empty(), PackedWeights.unsupported(policy));
            // Outside a scope the properties are the policy, as for the CLI.
            assertEquals(96, PackedWeights.loadPolicy().prefillBatchSize());
            assertTrue(PackedWeights.unsupported(PackedWeights.loadPolicy()).isPresent());
        } finally {
            restore(WIDTH, width);
            restore(PREFILL_DECODE, prefillDecode);
        }
    }

    @Test
    public void scopesNestAndDoNotCrossThreads() throws Exception {
        ExecutionPolicy outer = batched(256);
        ExecutionPolicy inner = batched(128);
        PackedWeights.loading(
                outer,
                () -> {
                    assertSame(inner, PackedWeights.loading(inner, PackedWeights::loadPolicy));
                    assertSame(outer, PackedWeights.loadPolicy());
                    ExecutionPolicy elsewhere =
                            CompletableFuture.supplyAsync(PackedWeights::loadPolicy).get();
                    assertTrue(elsewhere != outer && elsewhere != inner);
                    return null;
                });
    }

    @Test
    public void onlyAnUntiledBatchedPrefillWidthIsRefused() {
        assertEquals(
                Optional.empty(), PackedWeights.unsupported(ExecutionPolicy.builder().build()));
        assertEquals(Optional.empty(), PackedWeights.unsupported(batched(1)));
        assertEquals(Optional.empty(), PackedWeights.unsupported(batched(384)));
        assertTrue(PackedWeights.unsupported(batched(64)).isPresent());
    }

    private static void restore(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }
}

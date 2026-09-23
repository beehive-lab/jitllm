package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's Q8_0 logits against the CPU reference with the prompt ingested through the batched
 * prefill graphs, then decoded.
 *
 * <p>A width of seven: the prompt becomes several whole chunks and a partial final one, so the
 * compared decode rows depend on every chunk boundary the prefill crossed — the positions each row
 * believes it is at, and what the key/value cache holds when the next chunk starts.
 *
 * <p>Its own measured envelope, {@link CpuGpuParity#GEMMA4_Q8_0_BATCHED}: the prompt rows go
 * through FP16-operand GEMMs, which the single-token bounds were never set for.
 *
 * <p>Its own class, and therefore its own JVM: device memory a closed session frees returns to
 * TornadoVM's buffer provider rather than to the driver.
 */
public class Gemma4Q8_0BatchedPrefillParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    @Test
    public void gemma4E2bQ80BatchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.GEMMA_4_E2B_Q8_0, GEMMA4_Q8_0_BATCHED, 7);
    }
}

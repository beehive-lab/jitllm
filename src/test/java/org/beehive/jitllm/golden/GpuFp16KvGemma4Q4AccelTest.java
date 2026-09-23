package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * FP16 against FP32 key/value cache on the GPU for Gemma 4; see {@link GpuFp16KvPrecision}. The
 * family has no sequential prefill/decode plan, so its two modes are the single-token and the
 * batched one.
 */
public class GpuFp16KvGemma4Q4AccelTest {

    // @formatter:off
    /**
     * The packed-integer decode off, through its exact-comparison switch, so this measures what the
     * half-precision cache does and not what an eight-bit activation quantizer does to it.
     *
     * <p>With it on, the two runs differ by relative L2 0.027 on this file — every decode
     * activation is rounded to eight bits, and a cache difference far below that step still flips
     * some roundings, which then propagate. With it off they differ by 2.2e-4 (single-token) and
     * 3.6e-4 (batched), what the Q8_0 file measures. The packed path over the FP16 cache is scored
     * where its arithmetic can be judged: the long-prefix NLL screen.
     */
    // @formatter:on
    static {
        System.setProperty("jitllm.gemma4.packedIntegerDot", "false");
    }

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.GEMMA_4_E2B_Q4_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }

    @Test
    public void batchedPrefill() throws Exception {
        GpuFp16KvPrecision.check(Fixture.GEMMA_4_E2B_Q4_0, GpuFp16KvPrecision.Mode.BATCHED);
    }
}

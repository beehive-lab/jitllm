package org.beehive.jllm.golden;

import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.junit.Test;

/**
 * {@link Qwen35SequenceResetStandardAccelTest} with a half-precision key/value cache, which is the
 * path that decodes with split-KV attention.
 *
 * <p>The reset case is the one that matters for a split plan: the per-head partials live in scratch
 * that is reused across tokens and across sequences, so a reset that left a longer run's partials
 * behind would show here as the second sequence disagreeing with the first.
 */
public class Qwen35Fp16KvSequenceResetStandardAccelTest extends Qwen35SequenceReset {

    static {
        System.setProperty("jllm.kvcache.fp16", "true");
    }

    @Override
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {
        PlanDispatchEvidence.assertQwen35SplitKvAttention(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
    }

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceStandardOnFp16Kv() throws Exception {
        assertResetRestoresTheSequence(1);
    }
}

package org.beehive.jllm.golden;

import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.junit.Test;

/**
 * The reset case for split-KV in the <b>sequential</b> prefill/decode mode.
 *
 * <p>The third of the three plan shapes that reach this family's decode attention. A batch width of
 * one with the phase strategy on is prefill/decode rather than batched prefill/decode, and it
 * builds its decode attention through the same path, so the split pair has to be there too.
 */
public class Qwen35Fp16KvSequenceResetPrefillDecodeAccelTest extends Qwen35SequenceReset {

    static {
        System.setProperty("jllm.kvcache.fp16", "true");
        System.setProperty("jllm.withPrefillDecode", "true");
    }

    @Override
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {
        PlanDispatchEvidence.assertQwen35SplitKvAttention(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
    }

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequencePrefillDecodeOnFp16Kv() throws Exception {
        assertResetRestoresTheSequence(1);
    }
}

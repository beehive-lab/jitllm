package org.beehive.jllm.api;

import static org.junit.Assert.assertNotNull;

import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;

/**
 * {@link Qwen35BatchedFp16KvLifecycleAccelTest} with the tensor-core path selected.
 *
 * <p>The FP16 key/value lifecycle and its session reset over the MMA batched prefill. The property
 * selects the path; {@link #verifyDispatch} checks it against the scheduler of the plan this
 * session generated with, and fails if that evidence is missing.
 */
public class Qwen35MmaBatchedFp16KvLifecycleAccelTest
        extends Qwen35BatchedFp16KvLifecycleAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
    }

    @Override
    protected void verifyDispatch(DelegatingSession session, int batch, int dim) {
        var plan = session.planIfBuilt();
        assertNotNull("the session generated without building a plan", plan);
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan), batch, dim);
    }
}

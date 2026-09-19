package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's Q4_0 file against the CPU reference.
 *
 * <p>Its own class, because surefire forks per class and this family's three fixtures cannot be
 * loaded in one JVM: 9.3 GB in BF16 and 5.0 GB in Q8_0, and device memory a closed session frees
 * returns to TornadoVM's buffer provider rather than to the driver, so the third load exhausts a 24
 * GB card and the failure lands on whichever fixture ran last rather than on whichever is wrong.
 *
 * <p>The device runs this file <b>materialized as Q8_0</b>: {@code Gemma4PlanProvider} admits F16
 * and Q8_0, so the loader decodes its Q4_0 and Q4_1 blocks at load time. The host reference decodes
 * the same file natively, so this scores the materialization as well as the kernels — which is what
 * the device actually executes until a retained Q4_0 path exists.
 */
public class Gemma4Q4_0ParityAccelTest extends CpuGpuParity {

    @Test
    public void gemma4E2bQ4_0CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_Q4_0, Q4_0_PACKED_ACTIVATION);
    }
}

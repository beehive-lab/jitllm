package org.beehive.jitllm.backend.tornado.plan;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/** A Qwen3 load keeps every tensor as the file stores it, and its plans admit what it reports. */
public class Qwen3NativeRetentionTest {

    /** Admission takes every quantized type a kept load reports, and F16. */
    @Test
    public void everyKeptTypeIsAdmitted() {
        assertEquals(
                java.util.EnumSet.of(
                        DataType.F16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_1,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                new Qwen3PlanProvider().supportedDataTypes());
    }
}

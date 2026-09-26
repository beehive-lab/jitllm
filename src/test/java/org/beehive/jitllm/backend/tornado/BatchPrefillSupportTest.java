package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/** Where a family's batched prefill cannot be built, as a pure function of the configuration. */
public class BatchPrefillSupportTest {

    private static Optional<String> check(
            String arch, DataType weights, ExecutionMode mode, BackendId backend, boolean tc) {
        return BatchPrefillSupport.unsupported(
                new Combination(arch, weights, mode, backend, true, tc, true));
    }

    @Test
    public void onlyTheBatchedModeIsEverRefused() {
        for (ExecutionMode mode :
                new ExecutionMode[] {ExecutionMode.STANDARD, ExecutionMode.PREFILL_DECODE}) {
            for (String arch : new String[] {"gemma4", "qwen35", "llama"}) {
                assertEquals(
                        Optional.empty(),
                        check(arch, DataType.Q4_0, mode, BackendId.OPENCL, false));
            }
        }
    }

    @Test
    public void gemma4NeedsTensorCores() {
        assertEquals(
                Optional.empty(),
                check(
                        "gemma4",
                        DataType.Q8_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertTrue(
                check(
                                "gemma4",
                                DataType.Q8_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.OPENCL,
                                false)
                        .isPresent());
    }

    @Test
    public void qwen35IsRefusedOnOpenClOnly() {
        assertEquals(
                Optional.empty(),
                check(
                        "qwen35",
                        DataType.Q4_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.CUDA,
                        true));
        assertTrue(
                check(
                                "qwen35",
                                DataType.Q4_0,
                                ExecutionMode.BATCH_PREFILL_DECODE,
                                BackendId.OPENCL,
                                false)
                        .isPresent());
    }

    @Test
    public void theRefusalSaysEitherCacheAndTheWayOut() {
        var combination =
                new Combination(
                        "qwen35",
                        DataType.Q4_0,
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        BackendId.OPENCL,
                        true,
                        false,
                        true);
        String message = BatchPrefillSupport.refusal(combination, "reason");
        assertTrue(message, message.startsWith("[GPUL-CFG-002]"));
        assertTrue(message, message.contains("qwen35 / Q4_0 / BATCH_PREFILL_DECODE on opencl"));
        assertTrue(message, message.contains("either key/value cache"));
        assertTrue(message, message.contains("--batch-prefill-size"));
        assertFalse(message, message.contains("--fp32-kv-cache"));
    }
}

package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.OptionalInt;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.Overrides;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.SamplingResidency;
import org.junit.Test;

public class ExecutionPolicyTest {

    @Test
    public void structuralEqualityOverEveryField() {
        ExecutionPolicy base = ExecutionPolicy.builder().build();

        assertEquals(
                "two default policies are the same policy",
                base,
                ExecutionPolicy.builder().build());
        assertEquals(base.hashCode(), ExecutionPolicy.builder().build().hashCode());

        assertNotEquals(
                "phase strategy",
                base,
                ExecutionPolicy.builder().phaseStrategy(PhaseStrategy.PREFILL_DECODE).build());
        assertNotEquals(
                "prefill batch",
                ExecutionPolicy.builder()
                        .phaseStrategy(PhaseStrategy.PREFILL_DECODE)
                        .prefillBatchSize(4)
                        .build(),
                ExecutionPolicy.builder().phaseStrategy(PhaseStrategy.PREFILL_DECODE).build());
        assertNotEquals(
                "sampling residency",
                base,
                ExecutionPolicy.builder().samplingResidency(SamplingResidency.DEVICE).build());
        assertNotEquals(
                "split-KV partitions",
                base,
                ExecutionPolicy.builder().splitKvPartitions(8).build());
    }

    @Test
    public void anOverrideChangesOnlyTheFieldsItNames() {
        ExecutionPolicy model =
                ExecutionPolicy.builder()
                        .phaseStrategy(PhaseStrategy.PREFILL_DECODE)
                        .prefillBatchSize(8)
                        .samplingResidency(SamplingResidency.DEVICE)
                        .splitKvPartitions(4)
                        .build();

        ExecutionPolicy session =
                Overrides.builder()
                        .samplingResidency(SamplingResidency.HOST)
                        .build()
                        .applyTo(model);

        assertEquals(SamplingResidency.HOST, session.samplingResidency());
        assertEquals(
                "the phase strategy is the model's",
                PhaseStrategy.PREFILL_DECODE,
                session.phaseStrategy());
        assertEquals("so is the batch", 8, session.prefillBatchSize());
        assertEquals("and the partition count", OptionalInt.of(4), session.splitKvPartitions());
        assertNotEquals("and the session is not running the model's policy", model, session);
    }

    /** An empty override is not "a policy equal to the model's" — it is the model's, unchanged. */
    @Test
    public void anEmptyOverrideReturnsTheBaseItself() {
        ExecutionPolicy model =
                ExecutionPolicy.builder().samplingResidency(SamplingResidency.DEVICE).build();
        assertSame(model, Overrides.none().applyTo(model));
        assertTrue(Overrides.none().isEmpty());
    }

    /**
     * Switching to single-token drops the inherited batch rather than inheriting a contradiction.
     *
     * <p>Without this the override would build a policy that construction rejects, and the caller
     * would see a validation error for a field they never set.
     */
    @Test
    public void switchingToSingleTokenDropsTheInheritedPrefillBatch() {
        ExecutionPolicy model =
                ExecutionPolicy.builder()
                        .phaseStrategy(PhaseStrategy.PREFILL_DECODE)
                        .prefillBatchSize(8)
                        .build();

        ExecutionPolicy session =
                Overrides.builder()
                        .phaseStrategy(PhaseStrategy.SINGLE_TOKEN)
                        .build()
                        .applyTo(model);

        assertEquals(PhaseStrategy.SINGLE_TOKEN, session.phaseStrategy());
        assertEquals(1, session.prefillBatchSize());
    }

    @Test
    public void aPolicyThatCouldNotBeExecutedIsRejectedAtConstruction() {
        try {
            ExecutionPolicy.builder().prefillBatchSize(4).build(); // single-token by default
            fail("a single-token policy naming a prefill batch must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("prefill batch"));
        }
        try {
            ExecutionPolicy.builder().prefillBatchSize(0).build();
            fail("a prefill batch below one must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("at least 1"));
        }
    }

    @Test
    public void theDefaultsMatchTodaysPropertyDerivedValues() {
        String[] properties = {
            "jitllm.withPrefillDecode",
            "jitllm.prefillBatchSize",
            "jitllm.deviceSample",
            "jitllm.attention.splitKv",
            "jitllm.attention.splitKv.count"
        };
        String[] saved = new String[properties.length];
        for (int i = 0; i < properties.length; i++) {
            saved[i] = System.getProperty(properties[i]);
            System.clearProperty(properties[i]);
        }
        try {
            ExecutionPolicy fromProperties = ExecutionPolicy.fromSystemProperties();
            assertEquals(
                    "with no properties set, the default is the plain single-token policy",
                    ExecutionPolicy.builder().build(),
                    fromProperties);

            System.setProperty("jitllm.withPrefillDecode", "true");
            System.setProperty("jitllm.prefillBatchSize", "8");
            System.setProperty("jitllm.deviceSample", "true");
            System.setProperty("jitllm.attention.splitKv", "true");
            System.setProperty("jitllm.attention.splitKv.count", "16");
            assertEquals(
                    ExecutionPolicy.builder()
                            .phaseStrategy(PhaseStrategy.PREFILL_DECODE)
                            .prefillBatchSize(8)
                            .samplingResidency(SamplingResidency.DEVICE)
                            .splitKvPartitions(16)
                            .build(),
                    ExecutionPolicy.fromSystemProperties());

            // And it is read per call, not folded: a second read sees the change. That is the
            // defect this whole task removes, so reproducing it in the resolver would be
            // self-defeating.
            System.setProperty("jitllm.deviceSample", "false");
            assertEquals(
                    SamplingResidency.HOST,
                    ExecutionPolicy.fromSystemProperties().samplingResidency());
        } finally {
            for (int i = 0; i < properties.length; i++) {
                if (saved[i] == null) {
                    System.clearProperty(properties[i]);
                } else {
                    System.setProperty(properties[i], saved[i]);
                }
            }
        }
    }

    /** The model default is resolved at load, and the session's override applies on top of it. */
    @Test
    public void modelOptionsCarryTheDefaultAndSessionOptionsCarryTheOverride() {
        ExecutionPolicy policy =
                ExecutionPolicy.builder().samplingResidency(SamplingResidency.DEVICE).build();

        assertEquals(
                policy, ModelOptions.builder().executionPolicy(policy).build().executionPolicy());
        assertEquals(
                "an unset policy resolves at build, never later",
                ExecutionPolicy.fromSystemProperties(),
                ModelOptions.builder().build().executionPolicy());

        Overrides overrides = Overrides.builder().splitKvPartitions(2).build();
        assertEquals(
                overrides,
                SessionOptions.builder().executionPolicy(overrides).build().executionPolicy());
        assertTrue(SessionOptions.defaults().executionPolicy().isEmpty());
    }
}

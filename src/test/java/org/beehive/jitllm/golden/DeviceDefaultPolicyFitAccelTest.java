package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.DeviceDefaultPolicy;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.junit.Test;

/**
 * The device's default policy is taken only where it is predicted to fit: a GPU load that would run
 * out of memory on the default keeps the policy that was checked before the load.
 */
public class DeviceDefaultPolicyFitAccelTest {

    @Test
    public void aDefaultThatDoesNotFitIsSkipped() throws Exception {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_Q8_0);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_Q8_0),
                file != null);
        Model model = KvPrecisionHarness.load(file, 1024, true);
        ExecutionPolicy chosen = ExecutionPolicy.fromSystemProperties();
        StorageOptions storage = StorageOptions.fromSystemProperties();

        ExecutionPolicy fitting =
                DeviceDefaultPolicy.resolve(model, chosen, true, true, storage, policy -> true);
        assertNotEquals("the batched default is taken where it fits", chosen, fitting);

        ExecutionPolicy none =
                DeviceDefaultPolicy.resolve(model, chosen, true, true, storage, policy -> false);
        assertEquals("nothing that does not fit is taken", chosen, none);
    }
}

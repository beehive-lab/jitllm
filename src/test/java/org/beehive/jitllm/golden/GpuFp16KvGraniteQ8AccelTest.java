package org.beehive.jitllm.golden;

import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvGraniteQ8AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.GRANITE_3_2_2B_Q8_0, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }
}

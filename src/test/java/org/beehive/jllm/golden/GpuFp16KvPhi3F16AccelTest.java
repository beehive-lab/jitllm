package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvPhi3F16AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check(Fixture.PHI3_MINI_4K_F16, GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }
}

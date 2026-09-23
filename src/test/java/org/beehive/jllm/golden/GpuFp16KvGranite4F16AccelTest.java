package org.beehive.jllm.golden;

import org.junit.Test;

/** FP16 against FP32 key/value cache on the GPU; see {@link GpuFp16KvPrecision}. */
public class GpuFp16KvGranite4F16AccelTest {

    @Test
    public void singleToken() throws Exception {
        GpuFp16KvPrecision.check("granite-4.0-1b-F16.gguf", GpuFp16KvPrecision.Mode.SINGLE_TOKEN);
    }
}

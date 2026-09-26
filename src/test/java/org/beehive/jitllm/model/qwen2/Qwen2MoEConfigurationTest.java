package org.beehive.jitllm.model.qwen2;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class Qwen2MoEConfigurationTest {

    @Test
    public void batchDecodeReusesPrefillWeights() {
        Qwen2MoEConfiguration config =
                new Qwen2MoEConfiguration(
                        "Q8_0", 2048, 5632, 24, 16, 16, 128, 128, 151936, 32768, 200, 60, 4,
                        1408, 5632, false, 1e-6f, 1e6f);

        assertEquals(1, config.weightBindingFamilies(1));
        assertEquals(1, config.weightBindingFamilies(2));
    }
}

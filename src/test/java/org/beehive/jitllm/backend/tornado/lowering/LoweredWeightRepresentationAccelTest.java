package org.beehive.jitllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.program.ProgramSignature;
import org.junit.Test;

/**
 * Needs the accelerator because the representation asserted is the <b>materialized device</b> one,
 * which is a property of the GPU weights rather than of the file.
 */
public class LoweredWeightRepresentationAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    @Test
    public void twoRepresentationsOfOneArchitectureAreNotOneProgram() throws Exception {
        Path f16 = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        Path q8 = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (f16 == null || q8 == null) {
            assumeTrue(
                    "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0),
                    false);
        }
        String previousGpu = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try {
            ProgramSignature fromF16 = describe(f16);
            ProgramSignature fromQ8 = describe(q8);

            assertEquals(
                    "the component sequence is a property of the family, not of the"
                            + " representation — Q8_0 adds no operation",
                    fromF16.components().size(),
                    fromQ8.components().size());
            assertNotEquals(
                    "an F16 and a Q8_0 Llama are not the same program, and a signature that"
                            + " cannot tell them apart would let one compiled program serve both",
                    fromF16,
                    fromQ8);
        } finally {
            if (previousGpu == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previousGpu);
            }
        }
    }

    private static ProgramSignature describe(Path modelFile) throws Exception {
        Model loaded = ModelLoader.loadModel(modelFile, 256, true, true);
        return LoweredPlanSelection.describe(
                        loaded,
                        org.beehive.jitllm.runtime.policy.ExecutionPolicy.builder().build(),
                        org.beehive.jitllm.runtime.tensor.DataType.F32)
                .signature();
    }
}

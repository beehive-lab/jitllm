package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * The attention-output and down projections of Llama 3.2 1B F16's tensor-core batched prefill at
 * chunks that are not whole 128-row tiles, read off the printed task-graph chain.
 *
 * <p>The JIT GEMMs compute whole tiles, so they take the padded rows and split their depth on them:
 * a chunk of 16 is one tile, 16 output tiles and four depth slices. cuBLAS multiplies only the
 * chunk's tokens. Example: at a chunk of 129, the JIT GEMMs run 256 rows ({@code 512×16×4}) and
 * cuBLAS {@code n=129}.
 */
public class LlamaF16NarrowPrefillDispatchAccelTest {

    @Test
    public void jitGemmsSplitThePaddedRows() throws Exception {
        String chain16 = chain(16, false);
        assertTrue(
                chain16, chain16.matches("(?s).*task +woProj +gemmMMASplitK +256×16×4 / 256\n.*"));
        assertTrue(
                chain16, chain16.matches("(?s).*task +w2Proj +gemmMMASplitK +256×16×4 / 256\n.*"));
        assertTrue(
                chain16,
                chain16.matches("(?s).*task +w2ProjReduce +splitKReduce +262144 / 256\n.*"));
        String chain129 = chain(129, false);
        assertTrue(
                chain129,
                chain129.matches("(?s).*task +woProj +gemmMMASplitK +512×16×4 / 256\n.*"));
        assertTrue(
                chain129,
                chain129.matches("(?s).*task +w2Proj +gemmMMASplitK +512×16×4 / 256\n.*"));
    }

    @Test
    public void cublasTakesTheChunkTokens() throws Exception {
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        for (int width : new int[] {16, 129}) {
            String chain = chain(width, true);
            assertTrue(
                    chain,
                    chain.matches(
                            "(?s).*libraryTask +woProj +cuBLAS cublasGemmExFP16FP32 +m=2048 n="
                                    + width
                                    + " k=2048\n.*"));
            assertTrue(
                    chain,
                    chain.matches(
                            "(?s).*libraryTask +w2Proj +cuBLAS cublasGemmExFP16FP32 +m=2048 n="
                                    + width
                                    + " k=8192\n.*"));
        }
    }

    private static String chain(int width, boolean nativeLibraries) throws Exception {
        Path file = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16),
                file != null);
        assumeTrue(
                "the tensor-core prefill needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        return TaskGraphChainLlamaAccelTest.capture(
                file,
                Map.of(
                        "jitllm.withPrefillDecode",
                        "true",
                        "jitllm.prefillBatchSize",
                        String.valueOf(width),
                        ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY,
                        String.valueOf(nativeLibraries)));
    }
}

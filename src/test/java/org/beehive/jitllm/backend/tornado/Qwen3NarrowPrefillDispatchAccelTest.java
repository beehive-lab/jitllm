package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.NativePrefillSupport;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * Qwen3-0.6B F16's tensor-core batched prefill 512 wide builds a narrow family 128 wide, read off
 * the printed task-graph chain: seven more graphs, run for a chunk of up to 128 tokens, whose
 * projections and attention take 128 rows while the primary family's keep 512.
 */
public class Qwen3NarrowPrefillDispatchAccelTest {

    @Test
    public void cublasNarrowFamilyMultipliesOneTile() throws Exception {
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        String chain = chain(true, null);
        assertNarrowFamily(chain);
        String narrow = narrowSection(chain);
        assertTrue(
                narrow,
                narrow.matches(
                        "(?s).*libraryTask +qkvProj +cuBLAS cublasGemmExFP16FP32 +m=4096 n=128"
                                + " k=1024\n.*"));
        assertTrue(
                narrow,
                narrow.matches(
                        "(?s).*libraryTask +w2Proj +cuBLAS cublasGemmExFP16FP32 +m=1024 n=128"
                                + " k=3072\n.*"));
        assertTrue(
                narrow,
                narrow.matches("(?s).*task +attn_flash +flashAttentionHead128 +4096 / 128\n.*"));
        assertFalse("no cuDNN in the narrow family: " + narrow, narrow.contains("cudnn"));
        assertTrue(
                chain,
                chain.matches(
                        "(?s).*libraryTask +qkvProj +cuBLAS cublasGemmExFP16FP32 +m=4096 n=512"
                                + " k=1024\n.*"));
    }

    @Test
    public void jitNarrowFamilyRunsOneRowTile() throws Exception {
        String chain = chain(false, null);
        assertNarrowFamily(chain);
        String narrow = narrowSection(chain);
        assertTrue(narrow, narrow.matches("(?s).*task +qkvProj +gemmMMAQKV +256×32 / 256\n.*"));
        assertTrue(narrow, narrow.matches("(?s).*task +woProj +gemmMMASplitK +256×8×4 / 256\n.*"));
        assertTrue(
                narrow,
                narrow.matches("(?s).*task +attn_flash +flashAttentionHead128 +4096 / 128\n.*"));
        assertTrue(chain, chain.matches("(?s).*task +qkvProj +gemmMMAQKV +1024×32 / 256\n.*"));
    }

    @Test
    public void theNarrowFamilyCanBeTurnedOff() throws Exception {
        String chain = chain(true, "false");
        assertTrue(chain, chain.startsWith("Plan  QWEN_3 F16 · batch-prefill-decode (batch 512)"));
        assertFalse(chain, chain.contains("narrow"));
    }

    private static void assertNarrowFamily(String chain) {
        assertTrue(chain, chain.startsWith("Plan  QWEN_3 F16 · batch-prefill-decode (batch 512)"));
        assertTrue(chain, chain.contains("(prefill narrow layers, 7 graphs, 4 layers each)"));
        assertTrue(chain, chain.contains("per chunk of up to 128 prompt tokens"));
    }

    /** The narrow family's block of the chain, up to the next graph block. */
    private static String narrowSection(String chain) {
        int start = chain.indexOf("batchPrefillNarrowLayer_0 ");
        assertTrue("no narrow family:\n" + chain, start >= 0);
        int end = chain.indexOf("\n[", start);
        return chain.substring(start, end < 0 ? chain.length() : end + 1);
    }

    private static String chain(boolean nativeLibraries, String narrowChunks) throws Exception {
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                file != null);
        assumeTrue(
                "the tensor-core prefill needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        Map<String, String> properties = new HashMap<>();
        properties.put("jitllm.withPrefillDecode", "true");
        properties.put("jitllm.prefillBatchSize", "512");
        properties.put(ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, String.valueOf(nativeLibraries));
        String previous =
                System.getProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
        if (narrowChunks != null) {
            System.setProperty(
                    BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY, narrowChunks);
        }
        try {
            return TaskGraphChainLlamaAccelTest.capture(file, properties);
        } finally {
            if (previous == null) {
                System.clearProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
            } else {
                System.setProperty(
                        BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY, previous);
            }
        }
    }
}

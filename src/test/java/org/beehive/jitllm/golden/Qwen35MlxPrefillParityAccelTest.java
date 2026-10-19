package org.beehive.jitllm.golden;

import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

/**
 * Qwen3.5-0.8B against the CPU reference with its batched prefill projections on MLX; see {@link
 * MlxPrefillParity}. The attention, the delta rule and the norms stay on the Metal kernels, so the
 * MLX-only tasks checked are the projections' glue.
 *
 * <p>Uses {@code Qwen3.5-0.8B-Q8_0.gguf} and {@code Qwen3.5-0.8B-Q4_0.gguf} under the fixture
 * directory, by name; each skips when its file is absent.
 */
public class Qwen35MlxPrefillParityAccelTest extends MlxPrefillParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final Fp32KeyValueCache FP32_KEY_VALUE_CACHE = new Fp32KeyValueCache();

    @Override
    String[] mlxTasks() {
        return new String[] {".mlx_attn_q_split", ".mlx_ssm_qkv_split", ".mlx_ffn_residual"};
    }

    @Test
    public void q8_0Fp16Copies() throws Exception {
        assertMlxParity(model("Qwen3.5-0.8B-Q8_0.gguf"), "Q8_0", FP16, Weights.FP16_COPIES);
    }

    @Test
    public void q8_0Affine() throws Exception {
        assertMlxParity(model("Qwen3.5-0.8B-Q8_0.gguf"), "Q8_0", FP16, Weights.AFFINE);
    }

    @Test
    public void q4_0Fp16Copies() throws Exception {
        assertMlxParity(model("Qwen3.5-0.8B-Q4_0.gguf"), "Q4_0", FP16, Weights.FP16_COPIES);
    }

    private static Path model(String name) {
        Path file = GoldenFixture.modelsRoot().resolve(name);
        assumeTrue("environment absent: " + file, Files.isRegularFile(file));
        return file;
    }
}

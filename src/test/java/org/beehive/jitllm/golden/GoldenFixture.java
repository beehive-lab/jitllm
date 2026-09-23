package org.beehive.jitllm.golden;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HexFormat;

/**
 * Locates and verifies the pinned model fixtures for the golden and parity gates.
 *
 * <p>The GGUF files are far too large to commit, so only their SHA-256 is pinned here. The file
 * itself is resolved from {@code $JITLLM_TEST_MODELS} or {@code ~/.jitllm/test-models/} (falling
 * back to the pre-rename cache, see {@link #modelsRoot()}), and an absent fixture produces a fetch
 * instruction rather than a mysterious failure.
 *
 * <p>Per {@code verification-gates.md}, a missing fixture or absent accelerator causes the Class B
 * tests to <b>skip with an explicit marker</b> — never to pass.
 */
public final class GoldenFixture {

    public enum Fixture {
        LLAMA_3_2_1B_F16(
                "Llama-3.2-1B-Instruct-F16.gguf",
                "F16",
                "d4efb14e1eee8d5d9de41211cabd6e81030f79e8070176a3843f6e4e9ecc84da"),
        LLAMA_3_2_1B_Q8_0(
                "Llama-3.2-1B-Instruct-Q8_0.gguf",
                "Q8_0",
                "3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1"),

        /**
         * <b>No recorded golden logits.</b> The checks that use it compare the lowered path against
         * the legacy one <i>in one process on one file</i>, which is a stronger statement than a
         * recorded row and needs no committed data. Recording goldens is a reviewed action; adding
         * a fixture is not the same thing, and conflating them would have made this slice wait on a
         * decision it does not need.
         */
        /**
         * Quantized locally from {@link #LLAMA_3_2_1B_F16} with llama.cpp's {@code llama-quantize},
         * because the corpus had no Q4_0 file that any family could run on a device — and Q4_0
         * device residency could not be verified without one.
         *
         * <p>Its {@code blk.*} weights are all Q4_0; {@code token_embd}, which is also the output
         * projection here, is Q6_K. That mix is the normal shape of a Q4_0 file and is why the
         * layers read Q4_0 while the logits layer reads a Q8_0 materialization.
         */
        LLAMA_3_2_1B_Q4_0(
                "Llama-3.2-1B-Instruct-Q4_0.gguf",
                "Q4_0",
                "4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad"),

        QWEN2_5_0_5B_F16(
                "Qwen2.5-0.5B-Instruct-f16.gguf",
                "F16",
                "f1ad9d1174ce6ab47b584d522634c47e411b75bffdffd9a4e106e21e882392e5",
                "qwen2.5-0.5b"),
        QWEN2_5_0_5B_Q8_0(
                "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                "Q8_0",
                "25130a98aa782284a7dabea0c23245b2fd371ed47244e79d78b8ec23245fdf96",
                "qwen2.5-0.5b"),

        MISTRAL_7B_Q8_0(
                "Mistral-7B-Instruct-v0.3.Q8_0.gguf",
                "Q8_0",
                "24df553dc0e725196fe8a3c7be1edfe6ff17a0fe855f508b3f4a0e444e2e4281",
                "mistral-7b"),

        GRANITE_3_2_2B_F16(
                "granite-3.2-2b-instruct-f16.gguf",
                "F16",
                "000535b376c11e1eeb27231d85f19523db42f66d175b0e7cccb704610ae129ce",
                "granite-3.2-2b"),
        GRANITE_3_2_2B_Q8_0(
                "granite-3.2-2b-instruct-Q8_0.gguf",
                "Q8_0",
                "7ffbd0fe17ac37775c3758464aa9a09773a3a162b9459eb9094278a7a809682a",
                "granite-3.2-2b"),

        QWEN3_0_6B_F16(
                "Qwen3-0.6B-f16.gguf",
                "F16",
                "ab9004daf660cd6a6ba1c07556e74fcceb2b756063ccce3f9c69d3a637b361cc",
                "qwen3-0.6b"),
        QWEN3_0_6B_Q8_0(
                "Qwen3-0.6B-Q8_0.gguf",
                "Q8_0",
                "84c0dbe606526d5907251d88ea88b41457f46ce456e9a333d5d2b6245a95cafe",
                "qwen3-0.6b"),

        PHI3_MINI_4K_F16(
                "Phi-3-mini-4k-instruct-fp16.gguf",
                "F16",
                "5d99003e395775659b0dde3f941d88ff378b2837a8dc3a2ea94222ab1420fad3",
                "phi3-mini-4k"),
        PHI3_MINI_4K_Q8_0(
                "Phi-3-mini-4k-instruct-Q8_0.gguf",
                "Q8_0",
                "0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47",
                "phi3-mini-4k"),

        /**
         * The {@code qwen35} hybrid architecture, and the only fixture here that is not a small
         * model: 27B in 16 GB. Host-only, because no accelerator claims the architecture and the
         * device would need roughly 28 GB once its Q4_0 weights were materialized as Q8_0.
         *
         * <p>Quantization is recorded as the file's own {@code Q4_0} rather than the {@code Q8_0}
         * every quantized model reports for its activations, because what distinguishes this
         * fixture is what the weights are, and it also mixes Q4_1, Q5_K, Q6_K and Q8_0 tensors.
         */
        /**
         * Gemma 4 E2B, the family's first device fixture. Three representations of one checkpoint,
         * from {@code unsloth/gemma-4-E2B-it-GGUF}.
         *
         * <p>The fp16 leg is the file's own <b>BF16</b>: upstream publishes no F16 build of this
         * model, and {@code general.file_type} 32 reports FP16 activations for it, so it runs the
         * FP16 plan against BF16 weights.
         *
         * <p>The Q4_0 leg is mixed, which is the normal shape of a Q4_0 file and the reason this
         * family cannot use Llama's all-or-nothing retention: 241 Q4_0 tensors, {@code ffn_down}
         * Q4_1 on blocks 0-3 only, a Q4_K {@code token_embd} that is also the output projection,
         * and a Q5_K {@code per_layer_token_embd} that never reaches the device.
         *
         * <p>Both quantized files are pinned at upstream revision {@code 0314792d}. The Q8_0 hash
         * moved there from {@code 0a8488b1} with the repository's chat-template update; the Q4_0
         * file at that revision is the one pinned before.
         */
        GEMMA_4_E2B_BF16(
                "gemma-4-E2B-it-BF16.gguf",
                "BF16",
                "1eafd61d010ce8ca09db38f370aadd64c6d792db269c365ad0d9ea2709701890",
                "gemma-4-e2b"),
        GEMMA_4_E2B_Q8_0(
                "gemma-4-E2B-it-Q8_0.gguf",
                "Q8_0",
                "605d3c2647d7c58c1e4b5375ccb5702acf94c2611b4c8d4877812f8fdd32d053",
                "gemma-4-e2b"),
        GEMMA_4_E2B_Q4_0(
                "gemma-4-E2B-it-Q4_0.gguf",
                "Q4_0",
                "31d3a3c630d4e71a7416498c42660dd3805066948acaec76a47e1ffac7010132",
                "gemma-4-e2b"),

        QWEN3_8_27B_Q4_0(
                "Qwen3.8-27B-Q4_0.gguf",
                "Q4_0",
                "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d",
                "qwen3.8-27b");

        public final String fileName;
        public final String quantization;
        public final String sha256;
        private final String modelDirName;

        Fixture(String fileName, String quantization, String sha256) {
            this(fileName, quantization, sha256, "llama-3.2-1b");
        }

        Fixture(String fileName, String quantization, String sha256, String modelDirName) {
            this.fileName = fileName;
            this.quantization = quantization;
            this.sha256 = sha256;
            this.modelDirName = modelDirName;
        }

        /**
         * Directory name used for this fixture's committed goldens.
         *
         * <p>Derived from the model rather than hardcoded to Llama's, which is what it was until a
         * second model needed a fixture. A fixture with no recorded goldens still answers this —
         * the name is well-defined whether or not the directory exists.
         */
        public String goldenDirName() {
            return modelDirName + "-" + quantization.toLowerCase();
        }
    }

    private GoldenFixture() {}

    /**
     * Root of the local fixture cache.
     *
     * <p>The renames from GPULlama3.java to jllm to jitllm moved this from {@code
     * $GPULLAMA_TEST_MODELS} / {@code ~/.gpullama3/test-models}, then {@code $JLLM_TEST_MODELS} /
     * {@code ~/.jllm/test-models}, to {@code $JITLLM_TEST_MODELS} / {@code ~/.jitllm/test-models}.
     * Falling back to the old locations matters more than it looks: an unresolved fixture makes the
     * Class B gates <b>skip</b>, not fail, so a developer or runner that still has the old cache
     * would silently stop running every golden and accelerator correctness check while the build
     * stayed green. The fallback closes that window; it can go once no machine has the old cache.
     */
    public static Path modelsRoot() {
        String env = System.getenv("JITLLM_TEST_MODELS");
        if (env != null && !env.isBlank()) {
            return Paths.get(env);
        }
        for (String legacyVar : new String[] {"JLLM_TEST_MODELS", "GPULLAMA_TEST_MODELS"}) {
            String legacyEnv = System.getenv(legacyVar);
            if (legacyEnv != null && !legacyEnv.isBlank()) {
                return Paths.get(legacyEnv);
            }
        }
        Path home = Paths.get(System.getProperty("user.home"));
        Path current = home.resolve(".jitllm").resolve("test-models");
        if (Files.isDirectory(current)) {
            return current;
        }
        for (String legacyDir : new String[] {".jllm", ".gpullama3"}) {
            Path legacy = home.resolve(legacyDir).resolve("test-models");
            if (Files.isDirectory(legacy)) {
                return legacy;
            }
        }
        return current;
    }

    /**
     * @return the fixture path, or {@code null} when it is not present locally.
     */
    public static Path locate(Fixture fixture) {
        Path p = modelsRoot().resolve(fixture.fileName);
        return Files.isRegularFile(p) ? p : null;
    }

    public static String absentMessage(Fixture fixture) {
        return "Model fixture absent: "
                + fixture.fileName
                + "\n  expected under: "
                + modelsRoot()
                + "\n  sha256: "
                + fixture.sha256
                + "\n  Set JITLLM_TEST_MODELS to a directory containing it, or place/symlink the"
                + " file there. It is intentionally not committed.";
    }

    /** Full SHA-256 of the fixture; used to prove the golden was produced from this exact file. */
    public static String sha256(Path file) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 20];
            try (InputStream in = Files.newInputStream(file)) {
                int n;
                while ((n = in.read(buf)) > 0) {
                    md.update(buf, 0, n);
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }
}

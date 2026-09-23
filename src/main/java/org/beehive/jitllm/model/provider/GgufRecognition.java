package org.beehive.jitllm.model.provider;

import java.util.Locale;
import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Which architecture a GGUF file declares itself to be.
 *
 * <p>Replaces the substring match over {@code general.name} that recognition used to be. The file
 * states its architecture in {@code general.architecture}; reading that is both more reliable and
 * cheaper to reason about than matching words in a display name that anyone may set.
 *
 * <h2>Why the name is still consulted</h2>
 *
 * <p>Because {@code general.architecture} does not always discriminate, and the corpus says so:
 *
 * <pre>
 *   Mistral-7B-Instruct-v0.3.Q8_0.gguf              arch=llama     name=models--mistralai--Mistral-7B-Instruct-v0.3
 *   DeepSeek-R1-Distill-Qwen-1.5B.gguf              arch=qwen2     name=DeepSeek R1 Distill Qwen 1.5B
 *   Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf  arch=mistral3  name=Devstral-Small-2-24B-Instruct-2512
 * </pre>
 *
 * <p>Mistral and older Devstral builds ship as {@code llama}; the DeepSeek distill ships as {@code
 * qwen2}; Devstral's newer builds ship as {@code mistral3}. They need different tokenizers, chat
 * formats or RoPE bases, so the name is the only thing that tells them apart — but it is consulted
 * <b>within</b> a declared architecture, not across all of them. That is the difference from the
 * old chain, where "llama" anywhere in a name could win outright.
 *
 * <p>A file with no declared architecture falls back to the old name matching, so models produced
 * by tools that omit the key keep loading.
 */
final class GgufRecognition {

    private GgufRecognition() {}

    /**
     * The architecture this source is, or {@code null} when nothing here recognizes it.
     *
     * <p>Single-valued by construction: every provider asks this one question and claims the source
     * only if the answer is its own identity, so two providers cannot both claim a file.
     */
    static ArchitectureId architectureOf(ModelSource source) {
        String declared = lower(source.metadataString("general.architecture"));
        String name = lower(source.metadataString("general.name"));

        if (declared != null) {
            // Only the architectures that do not simply mean themselves are listed. Everything
            // else is its own identity, so adding a family needs no case here — which is what
            // makes a new family a new file plus one service line (Rule 15).
            return switch (declared) {
                case "llama" -> withinLlama(name);
                case "qwen2" ->
                        contains(name, "deepseek r1 distill")
                                ? ArchitectureId.of("deepseek-r1-distill-qwen")
                                : ArchitectureId.of("qwen2");
                case "granitemoehybrid" -> ArchitectureId.of("granite");
                case "mistral3" -> withinMistral3(name);
                default -> ArchitectureId.of(declared);
            };
        }

        if (source.metadata().containsKey("granite.block_count")) {
            return ArchitectureId.of("granite");
        }
        return byNameAlone(name);
    }

    /**
     * Devstral's newer builds declare {@code mistral3} rather than {@code llama}.
     *
     * <p>{@code Devstral-Small-2-24B-Instruct-2512} ships {@code general.architecture=mistral3} and
     * a {@code mistral3.*} metadata block — which is exactly what {@code DevstralModelLoader}
     * already reads: it hardcodes the {@code "mistral3"} prefix for every configuration key and
     * already handles that block's YaRN RoPE scaling ({@code rope.scaling.type=yarn}, factor,
     * beta_fast, beta_slow, log_multiplier, original_context_length). The retained Devstral
     * components were written for this metadata shape; only recognition had no case for it, so the
     * file reached no provider and failed with {@code [GPUL-MO2]}.
     *
     * <p>Scoped by name rather than claiming the whole architecture, for the same reason {@link
     * #withinLlama} is: {@code mistral3} is Mistral's architecture identifier, not Devstral's, and
     * a non-Devstral {@code mistral3} model has no provider here. Those keep falling through to
     * their own identity and are claimed by nobody — recognized as themselves and rejected with a
     * named diagnostic, which is the behaviour this class's contract calls for.
     */
    private static ArchitectureId withinMistral3(String name) {
        return contains(name, "devstral")
                ? ArchitectureId.of("devstral")
                : ArchitectureId.of("mistral3");
    }

    /** Mistral and Devstral declare themselves {@code llama}; only the name separates them. */
    private static ArchitectureId withinLlama(String name) {
        if (contains(name, "devstral")) {
            return ArchitectureId.of("devstral");
        }
        if (contains(name, "mistral")) {
            return ArchitectureId.of("mistral");
        }
        return ArchitectureId.of("llama");
    }

    /** The old chain, in its old order, for files that declare no architecture. */
    private static ArchitectureId byNameAlone(String name) {
        if (name == null) {
            return null;
        }
        if (contains(name, "granite")) {
            return ArchitectureId.of("granite");
        }
        if (contains(name, "devstral")) {
            return ArchitectureId.of("devstral");
        }
        if (contains(name, "mistral")) {
            return ArchitectureId.of("mistral");
        }
        if (contains(name, "llama")) {
            return ArchitectureId.of("llama");
        }
        if (contains(name, "deepseek r1 distill")) {
            return ArchitectureId.of("deepseek-r1-distill-qwen");
        }
        if (contains(name, "qwen2")) {
            return ArchitectureId.of("qwen2");
        }
        if (contains(name, "qwen3")) {
            return ArchitectureId.of("qwen3");
        }
        if (contains(name, "phi3") || contains(name, "phi-3")) {
            return ArchitectureId.of("phi3");
        }
        return null;
    }

    private static String lower(String value) {
        return value == null ? null : value.toLowerCase(Locale.ROOT);
    }

    private static boolean contains(String name, String marker) {
        return name != null && name.contains(marker);
    }
}

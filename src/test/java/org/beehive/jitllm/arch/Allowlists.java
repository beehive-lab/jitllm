package org.beehive.jllm.arch;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Enumerated allowlists for the rules that cannot pass on today's code.
 *
 * <p>Policy ({@code dependency-rules.md} §Allowlist policy), enforced by {@link
 * DependencyRulesTest}:
 *
 * <ol>
 *   <li>fully qualified names only — <b>never</b> a wildcard or a package;
 *   <li>every entry names the milestone that removes it;
 *   <li>an allowlist may shrink in any PR; it may not grow without an ADR or a recorded maintainer
 *       decision;
 *   <li>a rule with an empty allowlist has its allowlist deleted, not left empty — which is why
 *       Rules 7 and 11 have no list here;
 *   <li>CI fails both on a new violation and on a stale entry that no longer violates.
 * </ol>
 */
public final class Allowlists {

    /**
     * Rule 1 — classes outside the Tornado backend that import {@code uk.ac.manchester.tornado}.
     * Removal order per the rule text: sampler, then inference, then loaders, then state, then the
     * tornado tensors.
     */
    public static final Set<String> RULE_1 = frozen();

    /**
     * Rule 2 — model packages depending on TornadoVM or on the Tornado backend package. The
     * concrete model types are here for {@code TornadoVMMasterPlan} in {@code generateTokensGPU};
     * that leaves with the session split.
     */
    public static final Set<String> RULE_2 =
            frozen(
                    "org.beehive.jllm.model.Model",
                    "org.beehive.jllm.model.devstral.Devstral",
                    "org.beehive.jllm.model.gemma4.Gemma4",
                    "org.beehive.jllm.model.granite.Granite",
                    "org.beehive.jllm.model.llama.Llama",
                    "org.beehive.jllm.model.mistral.Mistral",
                    "org.beehive.jllm.model.phi3.Phi3",
                    "org.beehive.jllm.model.qwen2.Qwen2",
                    "org.beehive.jllm.model.qwen2.Qwen2MoE",
                    "org.beehive.jllm.model.qwen3.Qwen3",
                    "org.beehive.jllm.model.qwen35.Qwen35",
                    "org.beehive.jllm.model.loader.AbstractModelLoader",
                    "org.beehive.jllm.model.loader.DevstralModelLoader",
                    "org.beehive.jllm.model.loader.Gemma4ModelLoader",
                    "org.beehive.jllm.model.loader.GraniteLoader",
                    "org.beehive.jllm.model.loader.LlamaModelLoader",
                    "org.beehive.jllm.model.loader.MistralModelLoader",
                    "org.beehive.jllm.model.loader.ModelLoader",
                    "org.beehive.jllm.model.loader.Phi3ModelLoader",
                    "org.beehive.jllm.model.loader.Qwen2ModelLoader",
                    "org.beehive.jllm.model.loader.Qwen2MoEModelLoader",
                    "org.beehive.jllm.model.loader.Qwen35ModelLoader",
                    "org.beehive.jllm.model.loader.Qwen3ModelLoader");

    /** Rule 5 — loaded-model types with non-final fields. */
    public static final Set<String> RULE_5 =
            frozen(
                    "org.beehive.jllm.model.devstral.Devstral",
                    "org.beehive.jllm.model.gemma4.Gemma4",
                    "org.beehive.jllm.model.llama.Llama",
                    "org.beehive.jllm.model.mistral.Mistral",
                    "org.beehive.jllm.model.phi3.Phi3",
                    "org.beehive.jllm.model.qwen2.Qwen2",
                    "org.beehive.jllm.model.qwen2.Qwen2MoE",
                    "org.beehive.jllm.model.qwen3.Qwen3");

    /**
     * Rule 8a — lower layers reaching generation policy (the CLI, the options record, the server).
     *
     * <p>Nothing under {@code tornadovm} is listed, and {@link DependencyRulesTest} asserts that
     * stays true: on-device sampling is an <b>operation</b>, not generation policy (Rule 8b), so a
     * future device sampler must not be waved through here.
     */
    public static final Set<String> RULE_8A =
            frozen(
                    "org.beehive.jllm.bench.JllmBench",
                    "org.beehive.jllm.inference.sampler.Sampler",
                    "org.beehive.jllm.model.loader.ModelLoader",
                    "org.beehive.jllm.tensor.standard.Q4_0FloatTensor",
                    "org.beehive.jllm.tensor.standard.Q4_1FloatTensor");

    /**
     * Rule 15 — classes outside the provider package that still dispatch on {@code ModelType}.
     *
     * <p>What remains is the legacy load path, kept selectable with {@code -Djllm.providers=false}
     * for one release. Adding a <i>new</i> family no longer lands here: it is a new provider file
     * plus one service line. The CLI left this list when it moved onto the public facade — it now
     * reads {@code ModelInfo.architecture()} rather than naming a {@code ModelType}.
     */
    public static final Set<String> RULE_15 = frozen("org.beehive.jllm.model.loader.ModelLoader");

    /**
     * Rule 16 — console I/O in library code.
     *
     * <p>{@code JllmApp} and {@code Options} are today's CLI integration and are excluded by the
     * rule rather than listed here: printing is the CLI's job. The rule document counts 20 printing
     * files; the two CLI types account for the difference.
     */
    public static final Set<String> RULE_16 =
            frozen(
                    "org.beehive.jllm.auxiliary.RunMetrics",
                    "org.beehive.jllm.auxiliary.Timer$1",
                    "org.beehive.jllm.bench.JllmBench",
                    "org.beehive.jllm.inference.TokenGenerationLoop",
                    "org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA",
                    "org.beehive.jllm.backend.tornado.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillMMA",
                    "org.beehive.jllm.backend.tornado.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA",
                    "org.beehive.jllm.backend.tornado.layers.type.q8_0.prefill.Qwen3Q8_0LayersBatchPrefillMMA");

    /**
     * Rule 4 — classes outside the format layer and the loaders that still name GGUF's types.
     *
     * <ul>
     *   <li>{@code ModelType} and its enum-constant bodies take a {@code GGUF} in {@code
     *   <li>The {@code tensor.standard} and {@code tensor.tornado} classes decode GGML block
     * </ul>
     */
    public static final Set<String> RULE_4 =
            frozen(
                    "org.beehive.jllm.model.ModelType",
                    "org.beehive.jllm.model.ModelType$1",
                    "org.beehive.jllm.model.ModelType$2",
                    "org.beehive.jllm.model.ModelType$3",
                    "org.beehive.jllm.model.ModelType$4",
                    "org.beehive.jllm.model.ModelType$5",
                    "org.beehive.jllm.model.ModelType$6",
                    "org.beehive.jllm.model.ModelType$7",
                    "org.beehive.jllm.model.ModelType$8",
                    "org.beehive.jllm.model.ModelType$9",
                    "org.beehive.jllm.model.ModelType$10",
                    "org.beehive.jllm.model.ModelType$11",
                    "org.beehive.jllm.model.ModelType$12",
                    "org.beehive.jllm.tensor.standard.ArrayFloatTensor",
                    "org.beehive.jllm.tensor.standard.ArrayHalfFloatTensor",
                    "org.beehive.jllm.tensor.standard.BF16FloatTensor",
                    "org.beehive.jllm.tensor.standard.FloatTensor",
                    "org.beehive.jllm.tensor.standard.FP16FloatTensor",
                    "org.beehive.jllm.tensor.standard.FP32FloatTensor",
                    "org.beehive.jllm.tensor.standard.Q4_0FloatTensor",
                    "org.beehive.jllm.tensor.standard.Q4_1FloatTensor",
                    "org.beehive.jllm.tensor.standard.Q4_KFloatTensor",
                    "org.beehive.jllm.tensor.standard.Q5_KFloatTensor",
                    "org.beehive.jllm.tensor.standard.Q6_KFloatTensor",
                    "org.beehive.jllm.tensor.standard.Q8_0FloatTensor",
                    "org.beehive.jllm.backend.tornado.tensor.TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.FP16TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.FP32TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q4_0TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q8_0TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q4_1TornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q4_KTornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q5_KTornadoTensor",
                    "org.beehive.jllm.backend.tornado.tensor.Q6_KTornadoTensor");

    // Rule 7 and Rule 11 have no allowlist: they pass on today's code (policy item 4).

    private Allowlists() {}

    private static Set<String> frozen(String... names) {
        return Set.copyOf(new LinkedHashSet<>(Set.of(names)));
    }
}

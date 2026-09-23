package org.beehive.jllm.examples;

import java.nio.file.Path;
import org.beehive.jllm.api.GenerationRequest;
import org.beehive.jllm.api.GenerationResult;
import org.beehive.jllm.api.GenerationSession;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.api.TextGenerationModel;
import org.beehive.jllm.runtime.memory.MemoryPlan;

/**
 * Choosing where and how a model runs, and finding out before paying for it.
 *
 * <p>{@link LocalModels#preflight} answers "will this fit" without loading the weights, which is
 * the cheap way to fail. Then {@link ModelOptions} fixes what cannot change for the model's life:
 * the context budget, the backend, the device, the execution policy and the KV storage.
 *
 * <p>A device selector this build cannot honour throws instead of quietly running somewhere else.
 * That is deliberate — a silent fall back to the CPU looks like a slow GPU, not like a mistake.
 */
public final class DeviceAndPolicy {

    private DeviceAndPolicy() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: DeviceAndPolicy <model.gguf>");
            System.exit(2);
        }
        Path path = Path.of(args[0]);

        ModelOptions options = ModelOptions.builder().contextLength(2048).build();

        // Ask first: no weights are read, so this costs almost nothing.
        MemoryPlan plan = LocalModels.preflight(path, options);
        System.out.printf(
                "preflight: %.2f GiB of weights, %.2f GiB predicted budget, fits=%b%n",
                plan.logicalBytes() / (double) (1 << 30),
                plan.predictedBudgetBytes() / (double) (1 << 30),
                plan.fitsConfiguredBudget());
        plan.dominantComponents()
                .forEach(component -> System.out.println("  dominant: " + component));

        try (LocalModel model = LocalModels.load(path, options)) {
            System.out.printf(
                    "loaded %s (%s), context %d, compute type %s%n",
                    model.info().name(),
                    model.info().architecture(),
                    model.configuration().maxContextLength(),
                    model.info().computeType());

            try (GenerationSession session = ((TextGenerationModel) model).newSession()) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .prompt("Say hello in five words.")
                                        .maxNewTokens(32)
                                        .temperature(0.0f)
                                        .build());
                System.out.println(result.text());
                System.out.printf(
                        "prefill %s, decode %s%n",
                        result.timings().prefill(), result.timings().decode());
            }
        }
    }
}

package org.beehive.jitllm;

import java.io.IOException;
import java.util.Locale;
import java.util.Scanner;
import org.beehive.jitllm.api.FinishReason;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.integration.cli.ModelRunConfig;
import org.beehive.jitllm.integration.cli.StartupDiagnostics;

/**
 * The command-line integration.
 *
 * <p>It enters through the public facade, like any other caller: load a {@code LocalModel}, open a
 * {@code GenerationSession}, send {@code GenerationRequest}s. The chat template, the conversation
 * history, the stop tokens and the streaming decode all belong to the session, so what is left here
 * is what a CLI is actually for — parsing arguments and writing to the console.
 */
public class JitllmApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API =
            Boolean.parseBoolean(
                    System.getProperty(
                            "jitllm.VectorAPI",
                            "true")); // Enable Java Vector API for CPU acceleration
    public static final boolean SHOW_PERF_INTERACTIVE =
            Boolean.parseBoolean(
                    System.getProperty(
                            "jitllm.ShowPerfInteractive",
                            "true")); // Show performance metrics in interactive mode

    /**
     * On-device greedy sampling ({@code -Djitllm.deviceSample=true}) keeps the logits on the GPU
     * and returns only the argmax token id. It is only valid on the GPU FP16 greedy path for the
     * models whose decode loop reads {@code state.workspace.sampledToken} (Llama / Mistral /
     * Qwen3). For any other configuration the host still needs the full logits row, so the flag is
     * cleared here.
     *
     * <p>Must run after the model is loaded and before a session is opened: the property is read
     * when the session builds its execution plan.
     */
    private static void guardDeviceSample(LocalModel model, Options options) {
        if (!Boolean.getBoolean("jitllm.deviceSample")) {
            return;
        }
        boolean greedy = options.temperature() == 0.0f;
        boolean fp16 = "FP16".equals(model.info().computeType().name());
        String architecture = model.info().architecture();
        boolean wiredLoop =
                architecture.equals("llama")
                        || architecture.equals("mistral")
                        || architecture.equals("qwen3");
        if (!(options.useTornadovm() && greedy && fp16 && wiredLoop)) {
            System.err.println(
                    "[deviceSample] ignored — requires GPU + greedy (temperature 0) + FP16 + Llama/Mistral/Qwen3");
            System.clearProperty("jitllm.deviceSample");
        }
    }

    /** The request shape both modes share; only the prompt and system prompt differ per turn. */
    private static GenerationRequest.Builder request(Options options) {
        return GenerationRequest.builder()
                .maxNewTokens(options.maxNewTokens())
                .temperature(options.temperature())
                .topP(options.topp())
                .seed(options.seed());
    }

    private static void runSingleInstruction(GenerationSession session, Options options) {
        GenerationRequest.Builder builder =
                request(options).prompt(options.prompt()).systemPrompt(options.systemPrompt());
        if (options.stream()) {
            builder.onEvent(event -> System.out.print(event.text()));
        }
        GenerationResult result = session.generate(builder.build());
        if (options.stream()) {
            System.out.println();
        } else {
            System.out.println(result.text());
        }
        if (result.finishReason() == FinishReason.CONTEXT_FULL && result.generatedTokens() == 0) {
            // The prompt alone filled the capacity --max-tokens sized: nothing was generated,
            // and printing a zero-token metrics block was the only sign of it.
            throw new IllegalArgumentException(
                    contextFullMessage(result.promptTokens(), options.maxTokens()));
        }
        if (SHOW_PERF_INTERACTIVE) {
            RunMetrics.printMetrics();
        }
    }

    /**
     * The diagnostic for a prompt that leaves no room to generate: {@code --max-tokens} is the
     * capacity in positions this run was sized for, prompt included.
     */
    static String contextFullMessage(int promptTokens, int maxTokens) {
        return "the prompt is "
                + promptTokens
                + " tokens and --ctx-size "
                + maxTokens
                + " is the whole capacity (prompt plus generated tokens), so nothing could be"
                + " generated; pass --ctx-size larger than the prompt";
    }

    /**
     * The chat loop. The session carries the conversation, so each turn sends only the new user
     * text; the system prompt goes with the first turn and is retained from there.
     */
    private static void runInteractive(GenerationSession session, Options options) {
        Scanner in = new Scanner(System.in);
        boolean firstTurn = true;
        while (true) {
            System.out.print("> ");
            System.out.flush();
            if (!in.hasNextLine()) {
                break;
            }
            String userText = in.nextLine();
            if (userText.equals("quit") || userText.equals("exit")) {
                break;
            }

            GenerationRequest.Builder builder = request(options).prompt(userText);
            if (firstTurn) {
                builder.systemPrompt(options.systemPrompt());
                firstTurn = false;
            }
            if (options.stream()) {
                builder.onEvent(event -> System.out.print(event.text()));
            }

            GenerationResult result = session.generate(builder.build());
            if (options.stream()) {
                System.out.println();
            } else {
                System.out.println(result.text());
            }

            if (result.finishReason() == FinishReason.CONTEXT_FULL) {
                System.err.println(
                        "\n"
                                + contextFullMessage(result.promptTokens(), options.maxTokens())
                                + " (or start a new session)");
                break;
            }
            if (SHOW_PERF_INTERACTIVE) {
                RunMetrics.printMetrics();
            }
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * @param args command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException if model loading or file operations fail.
     */
    static void main(String[] args) throws IOException {
        org.beehive.jitllm.integration.cli.CliErrors.reportDiagnostics(() -> run(args));
    }

    private static void run(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        long startedNs = System.nanoTime();
        ModelOptions modelOptions =
                new ModelRunConfig(
                                options.modelPath(),
                                options.contextLength(),
                                options.useTornadovm())
                        .modelOptions();

        try (LocalModel model = LocalModels.load(options.modelPath(), modelOptions)) {
            long modelLoadNs = System.nanoTime() - startedNs;
            guardDeviceSample(model, options);
            try (GenerationSession session = ((TextGenerationModel) model).newSession()) {
                if (StartupDiagnostics.verbose()) {
                    String sampling =
                            options.temperature() == 0
                                    ? "greedy"
                                    : String.format(
                                            Locale.ROOT,
                                            "temperature %.3f / top-p %.3f / seed %d",
                                            options.temperature(),
                                            options.topp(),
                                            options.seed());
                    System.err.print(
                            StartupDiagnostics.render(
                                    model,
                                    session.prepare(),
                                    sampling,
                                    modelOptions,
                                    modelLoadNs,
                                    startedNs));
                }
                if (options.interactive()) {
                    runInteractive(session, options);
                } else {
                    runSingleInstruction(session, options);
                }
            }
        }
    }
}

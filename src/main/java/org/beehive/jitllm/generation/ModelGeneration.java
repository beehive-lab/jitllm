package org.beehive.jllm.generation;

import static org.beehive.jllm.JllmApp.SHOW_PERF_INTERACTIVE;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import org.beehive.jllm.Options;
import org.beehive.jllm.auxiliary.RunMetrics;
import org.beehive.jllm.auxiliary.metrics.RunMetricsSink;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;

/**
 * The generation loops that used to live on {@code Model} as default methods.
 *
 * <h2>Why they moved (Rule 8a)</h2>
 *
 * <p>Generation <i>policy</i> — the token loop, stop conditions, streaming, prompt construction and
 * console I/O — is not forward execution. A model that owns a generation loop cannot be an
 * embedding, classification or reranking model, and it drags a CLI options record and {@code
 * System.out} into the one interface every backend implements. The allowed direction is {@code
 * generation.**} → {@code model.**}, never the reverse, which is why this class knows about {@code
 * Options} and {@code Model} knows nothing about this class.
 *
 * <p><b>The bodies moved verbatim.</b> {@code this} became a {@code model} parameter and the
 * interface calls were qualified; nothing else changed, so the behaviour is the behaviour the
 * goldens already pin.
 *
 * <h2>What to use instead</h2>
 *
 * <p>These are <b>transitional</b>. New code uses the facade, which owns a session rather than a
 * loop:
 *
 * <pre>
 *   try (LocalModel model = LocalModels.load(path, ModelOptions.defaults());
 *        GenerationSession session = ((TextGenerationModel) model).newSession()) {
 *       GenerationResult result = session.generate(GenerationRequest.builder()
 * .prompt(prompt)
 * .maxNewTokens(256)
 * .onToken(token -> …)      // the streaming callback runInstructOnceLangChain4J takes
 * .build());
 *   }
 * </pre>
 *
 * <p>The mapping is exact: {@code options.prompt()} → {@code prompt}, {@code
 * options.systemPrompt()} → {@code systemPrompt}, {@code options.maxTokens()} → {@code
 * maxNewTokens}, {@code options.temperature()}/{@code topp()}/{@code seed()} → the request's own
 * fields, and the {@code Consumer<String>} → {@code onToken}. What the facade does <b>not</b>
 * reproduce is console echo: it returns text and streams tokens, and printing is the caller's
 * business, which is the point of the rule.
 */
public final class ModelGeneration {

    private ModelGeneration() {}

    /**
     * Model agnostic default implementation for interactive mode.
     *
     * @param sampler
     * @param options
     */
    public static void runInteractive(Model model, Sampler sampler, Options options) {
        // Even though might be expensive, create state here for smoother interaction later
        State state = model.createNewState();
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = model.chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        if (model.shouldAddBeginOfText()) {
            conversationTokens.add(chatFormat.getBeginOfText());
        }

        if (model.shouldAddSystemPrompt() && options.systemPrompt() != null) {
            conversationTokens.addAll(
                    chatFormat.encodeMessage(
                            new ChatFormat.Message(
                                    ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        int startPosition = 0;
        Scanner in = new Scanner(System.in);

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (options.useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(
                            state, model, RunMetricsSink.installedOrDisabled());
        }

        try {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = in.nextLine();
                if (List.of("quit", "exit").contains(userText)) {
                    break;
                }

                conversationTokens.addAll(
                        chatFormat.encodeMessage(
                                new ChatFormat.Message(ChatFormat.Role.USER, userText)));
                conversationTokens.addAll(
                        chatFormat.encodeHeader(
                                new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

                // Include reasoning for Deepseek-R1-Distill-Qwen
                if (model.shouldIncludeReasoning()) {
                    List<Integer> thinkStartTokens =
                            model.tokenizer()
                                    .encode(
                                            "<think>\n",
                                            model.tokenizer().getSpecialTokens().keySet());
                    conversationTokens.addAll(thinkStartTokens);

                    // If streaming, immediately output the think start
                    if (options.stream()) {
                        System.out.print("<think>\n");
                    }
                }

                Set<Integer> stopTokens = chatFormat.getStopTokens();

                List<Integer> responseTokens;
                IntConsumer tokenConsumer =
                        token -> {
                            if (options.stream()) {
                                if (model.tokenizer().shouldDisplayToken(token)) {
                                    System.out.print(model.tokenizer().decode(List.of(token)));
                                }
                            }
                        };

                // Choose between GPU and CPU path based on configuration
                if (options.useTornadovm()) {
                    // GPU path using TornadoVM
                    responseTokens =
                            model.generateTokensGPU(
                                    state,
                                    startPosition,
                                    conversationTokens.subList(
                                            startPosition, conversationTokens.size()),
                                    stopTokens,
                                    options.maxTokens(),
                                    sampler,
                                    options.echo(),
                                    options.stream() ? tokenConsumer : null,
                                    tornadoVMPlan);
                } else {
                    // CPU path
                    responseTokens =
                            model.generateTokens(
                                    state,
                                    startPosition,
                                    conversationTokens.subList(
                                            startPosition, conversationTokens.size()),
                                    stopTokens,
                                    options.maxTokens(),
                                    sampler,
                                    options.echo(),
                                    tokenConsumer);
                }

                // Include stop token in the prompt history, but not in the response displayed to
                // the user.
                conversationTokens.addAll(responseTokens);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                    stopToken = responseTokens.getLast();
                    responseTokens.removeLast();
                }
                if (!options.stream()) {
                    String responseText = model.tokenizer().decode(responseTokens);
                    // Add the forced <think>\n prefix for non-streaming output
                    if (model.shouldIncludeReasoning()) {
                        responseText = "<think>\n" + responseText;
                    }
                    System.out.println(responseText);
                }
                if (stopToken == null) {
                    System.err.println(
                            "\n Ran out of context length...\n Increase context length with by passing to jllm --max-tokens XXX");
                    break;
                }
                System.out.print("\n");

                // Optionally print performance metrics after each response
                if (SHOW_PERF_INTERACTIVE) {
                    RunMetrics.printMetrics();
                }
            }
        } finally {
            // Clean up TornadoVM resources when exiting the chat loop
            if (options.useTornadovm() && tornadoVMPlan != null) {
                try {
                    tornadoVMPlan.freeTornadoExecutionPlan();
                } catch (Exception e) {
                    System.err.println(
                            "Error while cleaning up TornadoVM resources: " + e.getMessage());
                }
            }
        }
    }

    /**
     * Model agnostic default implementation for instruct mode.
     *
     * @param sampler
     * @param options
     */
    public static String runInstructOnce(Model model, Sampler sampler, Options options) {
        State state = model.createNewState();
        ChatFormat chatFormat = model.chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        List<Integer> promptTokens = new ArrayList<>();

        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }

        if (model.shouldAddSystemPrompt() && options.systemPrompt() != null) {
            promptTokens.addAll(
                    chatFormat.encodeMessage(
                            new ChatFormat.Message(
                                    ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (options.useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(
                            state, model, RunMetricsSink.installedOrDisabled());
        }

        promptTokens.addAll(
                chatFormat.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        // Include reasoning for Deepseek-R1-Distill-Qwen
        if (model.shouldIncludeReasoning()) {
            List<Integer> thinkStartTokens =
                    model.tokenizer()
                            .encode("<think>\n", model.tokenizer().getSpecialTokens().keySet());
            promptTokens.addAll(thinkStartTokens);

            // If streaming, immediately output the think start
            if (options.stream()) {
                System.out.print("<think>\n");
            }
        }

        List<Integer> responseTokens;

        IntConsumer tokenConsumer =
                token -> {
                    if (options.stream()) {
                        if (model.tokenizer().shouldDisplayToken(token)) {
                            System.out.print(model.tokenizer().decode(List.of(token)));
                        }
                    }
                };

        Set<Integer> stopTokens = chatFormat.getStopTokens();

        if (options.useTornadovm()) {
            // GPU path using TornadoVM - Call generateTokensGPU without the token consumer
            // parameter
            responseTokens =
                    model.generateTokensGPU(
                            state,
                            0,
                            promptTokens,
                            stopTokens,
                            options.maxTokens(),
                            sampler,
                            options.echo(),
                            options.stream() ? tokenConsumer : null,
                            tornadoVMPlan);
        } else {
            // CPU path
            responseTokens =
                    model.generateTokens(
                            state,
                            0,
                            promptTokens,
                            stopTokens,
                            options.maxTokens(),
                            sampler,
                            options.echo(),
                            tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }

        String responseText = "";
        if (!options.stream()) {
            responseText = model.tokenizer().decode(responseTokens);
            // Add the forced <think>\n prefix for non-streaming output
            if (model.shouldIncludeReasoning()) {
                responseText = "<think>\n" + responseText;
            }
        }

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }

        return responseText;
    }

    public static String runInstructOnceLangChain4J(
            Model model, Sampler sampler, Options options, Consumer<String> tokenCallback) {
        State state = model.createNewState();
        ChatFormat chatFormat = model.chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        List<Integer> promptTokens = new ArrayList<>();

        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }

        if (model.shouldAddSystemPrompt() && options.systemPrompt() != null) {
            promptTokens.addAll(
                    chatFormat.encodeMessage(
                            new ChatFormat.Message(
                                    ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (options.useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(
                            state, model, RunMetricsSink.installedOrDisabled());
        }

        promptTokens.addAll(
                chatFormat.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        if (model.shouldIncludeReasoning()) {
            List<Integer> thinkStartTokens =
                    model.tokenizer()
                            .encode("<think>\n", model.tokenizer().getSpecialTokens().keySet());
            promptTokens.addAll(thinkStartTokens);

            // If streaming, immediately output the think start
            if (options.stream()) {
                System.out.print("<think>\n");
            }
        }

        List<Integer> responseTokens;

        IntConsumer tokenConsumer =
                token -> {
                    if (model.tokenizer().shouldDisplayToken(token)) {
                        String piece = model.tokenizer().decode(List.of(token));
                        if (options.stream() && tokenCallback != null) {
                            tokenCallback.accept(piece); // ✅ send to LangChain4j handler
                        }
                    }
                };

        Set<Integer> stopTokens = chatFormat.getStopTokens();

        if (options.useTornadovm()) {
            // GPU path using TornadoVM Call generateTokensGPU without the token consumer parameter
            responseTokens =
                    model.generateTokensGPU(
                            state,
                            0,
                            promptTokens,
                            stopTokens,
                            options.maxTokens(),
                            sampler,
                            options.echo(),
                            options.stream() ? tokenConsumer : null,
                            tornadoVMPlan);
        } else {
            // CPU path
            responseTokens =
                    model.generateTokens(
                            state,
                            0,
                            promptTokens,
                            stopTokens,
                            options.maxTokens(),
                            sampler,
                            options.echo(),
                            tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }

        String responseText = model.tokenizer().decode(responseTokens);

        if (!options.stream()) {
            responseText = model.tokenizer().decode(responseTokens);
            if (model.shouldIncludeReasoning()) {
                responseText = "<think>\n" + responseText;
            }
        }

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }

        return responseText;
    }
}

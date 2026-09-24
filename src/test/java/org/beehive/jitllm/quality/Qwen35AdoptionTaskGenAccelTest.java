package org.beehive.jitllm.quality;

import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.TornadoBatchPrefillPass;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.golden.TupleInfo;
import org.beehive.jitllm.inference.Logits;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

// @formatter:off
/**
 * Greedy generation for the adoption evaluation's objectively scored tasks: one JSONL of prompts in
 * ({@code jitllm.eval.tasks}), one JSONL of completions out ({@code jitllm.eval.out}). The model is
 * loaded once; every task is an independent sequence (reset between), the prompt goes through the
 * batched prefill at {@code jitllm.eval.batch} (default 512) with the chat template, thinking
 * disabled, then greedy decode until a stop token or {@code jitllm.eval.maxTokens} (default 384).
 * Identical prompts, tokenization, limits and decoding for whichever build runs it; the scoring is
 * done outside by {@code score_tasks.py}.
 */
// @formatter:on
public class Qwen35AdoptionTaskGenAccelTest {

    static {
        System.setProperty("jitllm.kvcache.fp16", "true");
    }

    private static final int CONTEXT = 2048;

    /**
     * The model under evaluation: {@code jitllm.eval.fixture} names a {@link Fixture}, the Qwen3.8
     * file by default. The harness is the frozen protocol's, so another family runs the same
     * prompts, lengths and scoring.
     */
    static Fixture evalFixture() {
        return Fixture.valueOf(System.getProperty("jitllm.eval.fixture", "QWEN3_8_27B_Q4_0"));
    }

    @Test
    public void theTasksAreGenerated() throws Exception {
        Path modelPath = GoldenFixture.locate(evalFixture());
        assumeTrue("environment absent", modelPath != null);
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        String tasksFile = System.getProperty("jitllm.eval.tasks");
        String outFile = System.getProperty("jitllm.eval.out");
        assumeTrue(
                "jitllm.eval.tasks and jitllm.eval.out select this run",
                tasksFile != null && outFile != null);
        int batch = Integer.getInteger("jitllm.eval.batch", 512);
        int maxTokens = Integer.getInteger("jitllm.eval.maxTokens", 384);
        String systemFile = System.getProperty("jitllm.eval.system");
        String systemText = systemFile == null ? null : Files.readString(Paths.get(systemFile));

        // The FP16 control: the same code and runtime with the int8 pairs excluded through the
        // test seam, so the projections run the FP16 dequantize-then-GEMM pair the int8 pair
        // replaced. The meta line records which kernels actually ran in either case.
        if ("fp16".equals(System.getProperty("jitllm.eval.control"))) {
            org.beehive.jitllm.backend.tornado.layers.Qwen35BatchPrefillLayers
                            .int8TaskFilterForTests =
                    task -> false;
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(batch));
        Model model = ModelLoader.loadModel(modelPath, CONTEXT, true, true);
        State state = State.withPrefillBatchSize(batch, model::createNewState);
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        var batched = (TornadoVMMasterPlanBatchPrefillDecode) plan;
        ChatFormat chat = model.chatFormat();
        Set<Integer> stops = chat.getStopTokens();
        List<String> lines = Files.readAllLines(Paths.get(tasksFile), StandardCharsets.UTF_8);
        StringBuilder out = new StringBuilder();
        out.append("{\"meta\": {\"batch\": ")
                .append(batch)
                .append(", \"maxTokens\": ")
                .append(maxTokens)
                .append(", \"systemPromptChars\": ")
                .append(systemText == null ? 0 : systemText.length())
                .append(", \"gateUpKernel\": \"")
                .append(
                        org.beehive.jitllm.backend.tornado.PlanDispatchEvidence
                                .batchedTaskKernelsIfAny(plan, "ffn_gate_proj"))
                .append("\", \"downKernel\": \"")
                .append(
                        org.beehive.jitllm.backend.tornado.PlanDispatchEvidence
                                .batchedTaskKernelsIfAny(plan, "ffn_down_proj"))
                .append("\", \"attnOutputKernel\": \"")
                .append(
                        org.beehive.jitllm.backend.tornado.PlanDispatchEvidence
                                .batchedTaskKernelsIfAny(plan, "attn_output_proj"))
                .append("\"}}\n");
        try {
            int done = 0;
            for (String line : lines) {
                if (line.isBlank()) {
                    continue;
                }
                String id = field(line, "id");
                String prompt = unescape(field(line, "prompt"));
                List<Integer> tokens = new ArrayList<>();
                if (model.shouldAddBeginOfText()) {
                    tokens.add(chat.getBeginOfText());
                }
                if (systemText != null) {
                    tokens.addAll(
                            chat.encodeMessage(
                                    new ChatFormat.Message(ChatFormat.Role.SYSTEM, systemText)));
                }
                tokens.addAll(
                        chat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, prompt)));
                tokens.addAll(
                        chat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
                tokens.addAll(chat.encodeThinkingControl(false));
                int[] promptTokens = tokens.stream().mapToInt(Integer::intValue).toArray();

                batched.resetSequenceState();
                // The batched prefill computes no logits: the prompt but its last token goes
                // through it, the last token through the decode step, whose logits start the
                // generation (the same shape the NLL screens use).
                int prefix = promptTokens.length - 1;
                for (int off = 0; off < prefix; off += batch) {
                    int size = Math.min(batch, prefix - off);
                    int[] chunk = java.util.Arrays.copyOfRange(promptTokens, off, off + size);
                    TornadoBatchPrefillPass.batchPrefill(model, state, chunk, off, size, batched);
                }
                List<Integer> generated = new ArrayList<>();
                int position = promptTokens.length;
                int next =
                        argmax(
                                TornadoBatchPrefillPass.decode(
                                        model, state, promptTokens[prefix], prefix, batched));
                String finish = "length";
                for (int i = 0; i < maxTokens; i++) {
                    if (stops.contains(next)) {
                        finish = "stop";
                        break;
                    }
                    generated.add(next);
                    Logits logits =
                            TornadoBatchPrefillPass.decode(model, state, next, position, batched);
                    position++;
                    next = argmax(logits);
                }
                String text = model.tokenizer().decode(generated);
                out.append("{\"id\": \"")
                        .append(id)
                        .append("\", \"promptTokens\": ")
                        .append(promptTokens.length)
                        .append(", \"prefillChunks\": ")
                        .append((prefix + batch - 1) / batch)
                        .append(", \"partialTail\": ")
                        .append(prefix % batch)
                        .append(", \"generated\": ")
                        .append(generated.size())
                        .append(", \"finish\": \"")
                        .append(finish)
                        .append("\", \"text\": \"")
                        .append(escape(text))
                        .append("\"}\n");
                done++;
                if (done % 10 == 0) {
                    System.out.println("[tasks] " + done + " / " + lines.size());
                }
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
        Files.writeString(Paths.get(outFile), out.toString());
    }

    private static int argmax(Logits logits) {
        int best = 0;
        float bestValue = logits.get(0);
        for (int i = 1; i < logits.size(); i++) {
            float v = logits.get(i);
            if (v > bestValue) {
                bestValue = v;
                best = i;
            }
        }
        return best;
    }

    /** The value of a string field of one JSON object line (the task files are flat). */
    private static String field(String line, String name) {
        String key = "\"" + name + "\": \"";
        int start = line.indexOf(key);
        if (start < 0) {
            throw new IllegalArgumentException("no field " + name + " in " + line);
        }
        start += key.length();
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '\\') {
                sb.append(c).append(line.charAt(++i));
            } else if (c == '"') {
                return sb.toString();
            } else {
                sb.append(c);
            }
        }
        throw new IllegalArgumentException("unterminated field " + name);
    }

    private static String unescape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(++i);
                switch (n) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
                        i += 4;
                    }
                    default -> sb.append(n);
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }
}

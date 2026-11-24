package org.beehive.jitllm.model.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntPredicate;
import org.beehive.jitllm.tokenizer.GptOssTokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;
import org.junit.Test;

/**
 * Harmony's channels in a response: what is shown, and what a later turn keeps. Over a byte-level
 * vocabulary with no merges, so every character is its own token.
 */
public class HarmonyChatFormatTest {

    private static final String[] SPECIALS = {
        "<|endoftext|>",
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>"
    };

    private static GptOssTokenizer tokenizer() {
        List<String> tokens = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            tokens.add(new String(Character.toChars(byteToUnicode(b))));
        }
        int base = tokens.size();
        tokens.addAll(List.of(SPECIALS));
        int[] types = new int[tokens.size()];
        for (int i = 0; i < types.length; i++) {
            types[i] = i < base ? 1 : 3;
        }
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("tokenizer.ggml.tokens", tokens.toArray(new String[0]));
        metadata.put("tokenizer.ggml.token_type", types);
        metadata.put("tokenizer.ggml.merges", new String[0]);
        return new GptOssTokenizer(metadata, Vocabulary.fromTokens(metadata));
    }

    /** GPT-2's byte-to-character map. */
    private static int byteToUnicode(int b) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            return b;
        }
        int n = 0;
        for (int c = 0; c < b; c++) {
            if (!((c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF))) {
                n++;
            }
        }
        return 256 + n;
    }

    /** A response as the model writes it: specials by name, text in between. */
    private static List<Integer> response(GptOssTokenizer t, String... parts) {
        List<Integer> ids = new ArrayList<>();
        for (String part : parts) {
            Integer special = t.getSpecialTokens().get(part);
            if (special != null) {
                ids.add(special);
            } else {
                ids.addAll(t.encodeOrdinaryAsList(part));
            }
        }
        return ids;
    }

    private static String shown(GptOssTokenizer t, IntPredicate filter, List<Integer> ids) {
        List<Integer> kept = new ArrayList<>();
        for (int id : ids) {
            if (filter.test(id)) {
                kept.add(id);
            }
        }
        return t.decode(kept);
    }

    @Test
    public void showsTheFinalChannelOnly() {
        GptOssTokenizer t = tokenizer();
        HarmonyChatFormat format = new HarmonyChatFormat(t, "medium");
        List<Integer> ids =
                response(
                        t,
                        "<|channel|>",
                        "analysis",
                        "<|message|>",
                        "The user asks for 2+2.",
                        "<|end|>",
                        "<|start|>",
                        "assistant",
                        "<|channel|>",
                        "final",
                        "<|message|>",
                        "2+2 = 4.",
                        "<|return|>");
        assertEquals("2+2 = 4.", shown(t, format.displayFilter(t), ids));
    }

    @Test
    public void eachResponseGetsItsOwnFilter() {
        GptOssTokenizer t = tokenizer();
        HarmonyChatFormat format = new HarmonyChatFormat(t, "medium");
        IntPredicate first = format.displayFilter(t);
        first.test(t.special("<|channel|>"));
        // A fresh filter starts at the response's header, not where the last one stopped.
        IntPredicate second = format.displayFilter(t);
        assertFalse(second.test(t.encodeOrdinaryAsList("x").get(0)));
    }

    @Test
    public void aMessageWithNoChannelIsShown() {
        GptOssTokenizer t = tokenizer();
        HarmonyChatFormat format = new HarmonyChatFormat(t, "medium");
        List<Integer> ids = response(t, "<|message|>", "Paris.", "<|return|>");
        assertEquals("Paris.", shown(t, format.displayFilter(t), ids));
    }

    @Test
    public void historyKeepsTheFinalAnswerAsTheTemplateRendersIt() {
        GptOssTokenizer t = tokenizer();
        HarmonyChatFormat format = new HarmonyChatFormat(t, "medium");
        List<Integer> ids =
                response(
                        t,
                        "<|channel|>",
                        "analysis",
                        "<|message|>",
                        "Thinking.",
                        "<|end|>",
                        "<|start|>",
                        "assistant",
                        "<|channel|>",
                        "final",
                        "<|message|>",
                        "Paris.");
        Optional<List<Integer>> kept = format.historyOfResponse(ids);
        assertTrue(kept.isPresent());
        assertEquals(response(t, "<|channel|>", "final", "<|message|>", "Paris."), kept.get());
    }

    @Test
    public void aResponseCutShortInItsReasoningIsKeptAsIs() {
        GptOssTokenizer t = tokenizer();
        HarmonyChatFormat format = new HarmonyChatFormat(t, "medium");
        List<Integer> ids = response(t, "<|channel|>", "analysis", "<|message|>", "Thinking");
        assertTrue(format.historyOfResponse(ids).isEmpty());
    }
}

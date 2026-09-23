package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jllm.tokenizer.Tokenizer;
import org.junit.Test;

/**
 * The stub is what makes multi-byte decoding testable at all: a real fixture would have to be
 * coaxed into emitting a split character, while here it is stated.
 */
public class TokenEventStreamTest {

    /** Tokens 1.n decode to letters; 90 and 91 are two halves of one character; 99 is a stop. */
    private static final class StubTokenizer implements Tokenizer {
        private static final Map<Integer, String> PIECES =
                Map.of(1, "a", 2, "b", 3, "c", 99, "<stop>");

        @Override
        public String decode(List<Integer> tokens) {
            StringBuilder out = new StringBuilder();
            for (int i = 0; i < tokens.size(); i++) {
                int token = tokens.get(i);
                if (token == 90) {
                    // Half a character: complete only when 91 immediately follows.
                    if (i + 1 < tokens.size() && tokens.get(i + 1) == 91) {
                        out.append('é');
                        i++;
                    } else {
                        out.append('�');
                    }
                } else if (token == 91) {
                    out.append('�');
                } else {
                    out.append(PIECES.getOrDefault(token, "?"));
                }
            }
            return out.toString();
        }

        @Override
        public boolean shouldDisplayToken(int token) {
            return token != 50;
        }

        @Override
        public List<Integer> encode(String text, Set<String> allowedSpecial) {
            return List.of();
        }

        @Override
        public List<Integer> encodeAsList(String text) {
            return List.of();
        }

        @Override
        public Map<String, Integer> getSpecialTokens() {
            return Map.of();
        }

        @Override
        public boolean isSpecialToken(int token) {
            return token == 99;
        }

        @Override
        public String regexPattern() {
            return "";
        }
    }

    private static final Set<Integer> STOP = Set.of(99);

    private record Capture(List<GenerationEvent> events, List<String> texts) {}

    private static Capture run(List<Integer> tokens) {
        List<GenerationEvent> events = new ArrayList<>();
        List<String> texts = new ArrayList<>();
        TokenEventStream stream =
                new TokenEventStream(new StubTokenizer(), STOP, events::add, texts::add);
        tokens.forEach(stream::accept);
        String text = stream.finish();
        assertEquals(
                "concatenated event text must equal the response", String.join("", texts), text);
        return new Capture(events, texts);
    }

    @Test
    public void theTerminalStopTokenIsNeitherEmittedNorCounted() {
        Capture capture = run(List.of(1, 2, 99));
        assertEquals(
                List.of(1, 2), capture.events().stream().map(GenerationEvent::tokenId).toList());
        assertTrue(
                "no control token reaches the public stream",
                capture.events().stream().noneMatch(e -> e.tokenId() == 99));
    }

    @Test
    public void eventsAreInGenerationOrder() {
        assertEquals(
                List.of(1, 2, 3),
                run(List.of(1, 2, 3, 99)).events().stream().map(GenerationEvent::tokenId).toList());
    }

    @Test
    public void generationThatEndsWithoutAStopTokenEmitsEveryToken() {
        // Running out of budget is not a stop token, and the last token is real output.
        assertEquals(
                List.of(1, 2, 3),
                run(List.of(1, 2, 3)).events().stream().map(GenerationEvent::tokenId).toList());
    }

    @Test
    public void aMultiByteCharacterAttachesToTheEventThatCompletesIt() {
        Capture capture = run(List.of(1, 90, 91, 99));
        assertEquals(3, capture.events().size());
        assertEquals("a", capture.events().get(0).text());
        assertEquals(
                "the first half completes nothing, and says so",
                "",
                capture.events().get(1).text());
        assertEquals("é", capture.events().get(2).text());
        assertEquals("aé", String.join("", capture.texts()));
    }

    @Test
    public void aNonDisplayableTokenIsAnEventWithNoText() {
        Capture capture = run(List.of(1, 50, 2, 99));
        assertEquals(
                "it is still an emitted token, so it is still an event",
                3,
                capture.events().size());
        assertEquals("", capture.events().get(1).text());
        assertEquals("ab", String.join("", capture.texts()));
    }

    @Test
    public void onEventRunsBeforeOnTokenForTheSameEvent() {
        List<String> order = new ArrayList<>();
        TokenEventStream stream =
                new TokenEventStream(
                        new StubTokenizer(),
                        STOP,
                        event -> order.add("event:" + event.tokenId()),
                        text -> order.add("text:" + text));
        stream.accept(1);
        stream.accept(99);
        stream.finish();
        assertEquals(List.of("event:1", "text:a"), order);
    }

    @Test
    public void aThrowingOnEventStopsTheLaterCallbackForThatEvent() {
        List<String> texts = new ArrayList<>();
        TokenEventStream stream =
                new TokenEventStream(
                        new StubTokenizer(),
                        STOP,
                        event -> {
                            throw new IllegalStateException("consumer failed");
                        },
                        texts::add);
        stream.accept(1);
        assertThrows(IllegalStateException.class, () -> stream.accept(2));
        assertTrue("onToken must not see a token whose onEvent threw", texts.isEmpty());
    }

    @Test
    public void aStopTokenInTheMiddleIsEmitted() {
        // Reachable with the ignore-end-of-sequence debug flag, which makes the loop continue past
        // a stop token. It is then genuinely part of the output, and is counted as such.
        assertEquals(
                List.of(1, 99, 2),
                run(List.of(1, 99, 2)).events().stream().map(GenerationEvent::tokenId).toList());
    }
}

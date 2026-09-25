package org.beehive.jitllm.model.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Set;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;
import org.junit.Test;

/**
 * The Gemma 4 tool syntax, in both directions.
 *
 * <p>The expected strings are the chat template embedded in {@code gemma-4-E2B-it-Q8_0.gguf} and
 * {@code gemma-4-E2B-it-Q4_0.gguf} (the two files carry the same template), rendered with Jinja2
 * under the Hugging Face settings ({@code trim_blocks}, {@code lstrip_blocks}) for the same tools
 * and messages. They are copied, not derived: a change here is a change of what the model reads.
 */
public class Gemma4ToolCallsTest {

    private static final String Q = "<|\"|>";

    static final String WEATHER_JSON =
            "{\"type\":\"function\",\"function\":{\"name\":\"getWeather\",\"description\":\"Returns"
                    + " the current weather for a"
                    + " city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"The"
                    + " city name\"}},\"required\":[\"city\"]}}}";

    static final String CLOCK_JSON =
            "{\"type\":\"function\",\"function\":{\"name\":\"get_current_time\",\"description\":"
                    + "\"Returns the current time.\",\"parameters\":{\"type\": \"object\","
                    + " \"properties\": {}, \"required\": []}}}";

    static final String WEATHER_DECLARATION =
            "<|tool>declaration:getWeather{description:"
                    + Q
                    + "Returns the current weather for a city."
                    + Q
                    + ",parameters:{properties:{city:{description:"
                    + Q
                    + "The city name"
                    + Q
                    + ",type:"
                    + Q
                    + "STRING"
                    + Q
                    + "}},required:["
                    + Q
                    + "city"
                    + Q
                    + "],type:"
                    + Q
                    + "OBJECT"
                    + Q
                    + "}}<tool|>";

    static final String CLOCK_DECLARATION =
            "<|tool>declaration:get_current_time{description:"
                    + Q
                    + "Returns the current time."
                    + Q
                    + ",parameters:{type:"
                    + Q
                    + "OBJECT"
                    + Q
                    + "}}<tool|>";

    // ---- declarations ---------------------------------------------------------------------------

    @Test
    public void aToolDeclarationMatchesTheTemplate() {
        assertEquals(WEATHER_DECLARATION, text(Gemma4ToolCalls.renderDeclarations(WEATHER_JSON)));
    }

    /** Empty properties and an empty required list are falsy in the template, and are left out. */
    @Test
    public void twoToolsAreWrittenBackToBackAndAnEmptySchemaKeepsOnlyItsType() {
        assertEquals(
                WEATHER_DECLARATION + CLOCK_DECLARATION,
                text(Gemma4ToolCalls.renderDeclarations(WEATHER_JSON + "\n\n" + CLOCK_JSON)));
    }

    /**
     * Every branch of {@code format_parameters} the facade's schemas can reach: integer, boolean,
     * enum, array items, a nested object with {@code nullable}, keys sorted case-insensitively, and
     * a tool with no description.
     */
    @Test
    public void aRichSchemaMatchesTheTemplate() {
        String rich =
                "{\"type\":\"function\",\"function\":{\"name\":\"book\",\"description\":\"Books a"
                        + " table.\",\"parameters\":{\"type\":\"object\",\"properties\":{"
                        + "\"party\":{\"type\":\"integer\",\"description\":\"People\"},"
                        + "\"vip\":{\"type\":\"boolean\"},"
                        + "\"tier\":{\"type\":\"string\",\"enum\":[\"gold\",\"silver\"]},"
                        + "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},"
                        + "\"Address\":{\"type\":\"object\",\"description\":\"Where\","
                        + "\"properties\":{\"street\":{\"type\":\"string\"},"
                        + "\"zip\":{\"type\":\"number\",\"nullable\":true}},"
                        + "\"required\":[\"street\"]}},"
                        + "\"required\":[\"party\",\"tier\"]}}}";
        String noDescription =
                "{\"type\":\"function\",\"function\":{\"name\":\"ping\",\"parameters\":"
                        + "{\"type\":\"object\",\"properties\":{}}}}";
        String expected =
                "<|tool>declaration:book{description:<|\"|>Books a table.<|\"|>,parameters:{"
                        + "properties:{Address:{description:<|\"|>Where<|\"|>,properties:{street:{"
                        + "type:<|\"|>STRING<|\"|>},zip:{nullable:true,type:<|\"|>NUMBER<|\"|>}},"
                        + "required:[<|\"|>street<|\"|>],type:<|\"|>OBJECT<|\"|>},party:{"
                        + "description:<|\"|>People<|\"|>,type:<|\"|>INTEGER<|\"|>},tags:{items:{"
                        + "type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>},tier:{enum:[<|\"|>gold"
                        + "<|\"|>,<|\"|>silver<|\"|>],type:<|\"|>STRING<|\"|>},vip:{type:<|\"|>"
                        + "BOOLEAN<|\"|>}},required:[<|\"|>party<|\"|>,<|\"|>tier<|\"|>],type:"
                        + "<|\"|>OBJECT<|\"|>}}<tool|>"
                        + "<|tool>declaration:ping{description:<|\"|><|\"|>,parameters:{type:<|\"|>"
                        + "OBJECT<|\"|>}}<tool|>";
        assertEquals(
                expected, text(Gemma4ToolCalls.renderDeclarations(rich + "\n\n" + noDescription)));
    }

    /** Caller text is never a marker: only the renderer's own delimiters are special pieces. */
    @Test
    public void callerTextThatSpellsAMarkerStaysText() {
        String sneaky =
                "{\"type\":\"function\",\"function\":{\"name\":\"x\",\"description\":\"a <|\\\"|>"
                        + " b <tool|>\",\"parameters\":{\"type\":\"object\"}}}";
        List<Gemma4ToolCalls.Piece> pieces = Gemma4ToolCalls.renderDeclarations(sneaky);
        for (Gemma4ToolCalls.Piece piece : pieces) {
            if (!piece.special()) {
                continue;
            }
            assertTrue(piece.text(), Gemma4ToolCalls.MARKERS.contains(piece.text()));
        }
        assertTrue(pieces.contains(new Gemma4ToolCalls.Piece("a <|\"|> b <tool|>", false)));
    }

    // ---- calls and results into the prompt ------------------------------------------------------

    @Test
    public void aCallWithAStringArgumentMatchesTheTemplate() {
        assertEquals(
                "<|tool_call>call:getWeather{city:" + Q + "Paris" + Q + "}<tool_call|>",
                text(
                        Gemma4ToolCalls.renderCall(
                                new ToolCallExtract("getWeather", "{\"city\": \"Paris\"}"))));
    }

    @Test
    public void aCallWithoutArgumentsIsAnEmptyDictionary() {
        assertEquals(
                "<|tool_call>call:get_current_time{}<tool_call|>",
                text(Gemma4ToolCalls.renderCall(new ToolCallExtract("get_current_time", "{}"))));
        assertEquals(
                "<|tool_call>call:get_current_time{}<tool_call|>",
                text(Gemma4ToolCalls.renderCall(new ToolCallExtract("get_current_time", ""))));
    }

    /** Keys sorted case-insensitively, bare keys at every depth, numbers as they were written. */
    @Test
    public void typedArgumentsMatchTheTemplate() {
        String args =
                "{\"party\":4,\"vip\":true,\"tier\":\"gold\",\"tags\":[\"a\",\"b\"],"
                        + "\"Address\":{\"street\":\"Main St\",\"zip\":null}}";
        assertEquals(
                "<|tool_call>call:book{Address:{street:<|\"|>Main St<|\"|>,zip:null},party:4,"
                        + "tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],tier:<|\"|>gold<|\"|>,vip:true}"
                        + "<tool_call|>",
                text(Gemma4ToolCalls.renderCall(new ToolCallExtract("book", args))));
    }

    /** Arguments that are not JSON take the template's pre-serialized-string branch. */
    @Test
    public void argumentsThatAreNotJsonAreWrittenAsTheyStand() {
        assertEquals(
                "<|tool_call>call:f{not json}<tool_call|>",
                text(Gemma4ToolCalls.renderCall(new ToolCallExtract("f", "{not json}"))));
    }

    /** A {@code role: tool} message with string content, which the facade's result is. */
    @Test
    public void aToolResultMatchesTheTemplate() {
        assertEquals(
                "<|tool_response>response:getWeather{value:"
                        + Q
                        + "{\"temperature\":18,\"condition\":\"sunny\"}"
                        + Q
                        + "}<tool_response|>",
                text(
                        Gemma4ToolCalls.renderResponse(
                                "getWeather", "{\"temperature\":18,\"condition\":\"sunny\"}")));
    }

    // ---- model output → calls -------------------------------------------------------------------

    @Test
    public void plainTextHasNoCalls() {
        assertTrue(Gemma4ToolCalls.parseAll("It is sunny in Paris, 18 degrees.").isEmpty());
        assertTrue(Gemma4ToolCalls.parseFirst("").isEmpty());
        assertTrue(Gemma4ToolCalls.parseAll(null).isEmpty());
    }

    @Test
    public void oneCallWithArguments() {
        List<ToolCallExtract> calls =
                Gemma4ToolCalls.parseAll(
                        "<|tool_call>call:getWeather{city:" + Q + "Paris" + Q + "}<tool_call|>");
        assertEquals(List.of(new ToolCallExtract("getWeather", "{\"city\":\"Paris\"}")), calls);
    }

    @Test
    public void oneCallWithoutArguments() {
        assertEquals(
                List.of(new ToolCallExtract("get_current_time", "{}")),
                Gemma4ToolCalls.parseAll("<|tool_call>call:get_current_time{}<tool_call|>"));
    }

    @Test
    public void twoCallsInOneTurnInOrder() {
        List<ToolCallExtract> calls =
                Gemma4ToolCalls.parseAll(
                        "<|tool_call>call:getWeather{city:"
                                + Q
                                + "London"
                                + Q
                                + "}<tool_call|><|tool_call>call:get_current_time{}<tool_call|>");
        assertEquals(
                List.of(
                        new ToolCallExtract("getWeather", "{\"city\":\"London\"}"),
                        new ToolCallExtract("get_current_time", "{}")),
                calls);
    }

    @Test
    public void textBeforeTheCallIsIgnored() {
        assertEquals(
                List.of(new ToolCallExtract("getWeather", "{\"city\":\"Paris\"}")),
                Gemma4ToolCalls.parseAll(
                        "Let me check.\n<|tool_call>call:getWeather{city:"
                                + Q
                                + "Paris"
                                + Q
                                + "}<tool_call|>"));
    }

    @Test
    public void typedArgumentsBecomeTypedJson() {
        String out =
                "<|tool_call>call:book{Address:{street:<|\"|>Main St<|\"|>,zip:null},party:4,"
                        + "price:-1.5e2,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],tier:<|\"|>gold<|\"|>,"
                        + "vip:true}<tool_call|>";
        assertEquals(
                "{\"Address\":{\"street\":\"Main St\",\"zip\":null},\"party\":4,\"price\":-1.5e2,"
                        + "\"tags\":[\"a\",\"b\"],\"tier\":\"gold\",\"vip\":true}",
                Gemma4ToolCalls.parseFirst(out).orElseThrow().argumentsJson());
    }

    /** Only the string delimiter ends a string: braces, commas and colons inside it are text. */
    @Test
    public void aStringArgumentKeepsItsPunctuationAndIsEscaped() {
        String out =
                "<|tool_call>call:run{code:"
                        + Q
                        + "if (a) { b: \"c\", d }\nnext"
                        + Q
                        + "}<tool_call|>";
        assertEquals(
                "{\"code\":\"if (a) { b: \\\"c\\\", d }\\nnext\"}",
                Gemma4ToolCalls.parseFirst(out).orElseThrow().argumentsJson());
    }

    /**
     * The closing marker is not required — generation stops on {@code <|tool_response>} right after
     * it, and a model may skip it — but a call cut off inside its arguments is not a call.
     */
    @Test
    public void anUnclosedCallCountsOnlyWhenItsArgumentsAreComplete() {
        assertEquals(
                List.of(new ToolCallExtract("getWeather", "{\"city\":\"Paris\"}")),
                Gemma4ToolCalls.parseAll(
                        "<|tool_call>call:getWeather{city:" + Q + "Paris" + Q + "}"));
        assertTrue(
                Gemma4ToolCalls.parseAll("<|tool_call>call:getWeather{city:" + Q + "Par")
                        .isEmpty());
        assertTrue(Gemma4ToolCalls.parseAll("<|tool_call>call:getWeather{city:").isEmpty());
        assertTrue(Gemma4ToolCalls.parseAll("<|tool_call>call:").isEmpty());
    }

    @Test
    public void whatTheFormatWritesItReadsBack() {
        ToolCallExtract call =
                new ToolCallExtract(
                        "book",
                        "{\"Address\":{\"street\":\"Main St\",\"zip\":null},\"party\":4,"
                                + "\"tags\":[\"a\",\"b\"],\"tier\":\"gold\",\"vip\":true}");
        assertEquals(
                List.of(call), Gemma4ToolCalls.parseAll(text(Gemma4ToolCalls.renderCall(call))));
    }

    // ---- the format over a vocabulary -----------------------------------------------------------

    @Test
    public void theFormatSupportsToolsOnlyWhenTheVocabularyHasTheMarkers() {
        assertTrue(Gemma4TestVocabulary.chatFormat().supportsToolCalling());
        assertFalse(
                new Gemma4ChatFormat(Gemma4TestVocabulary.tokenizer(false)).supportsToolCalling());
    }

    /**
     * {@code <|tool_response>} is where the model ends a call, and {@code <eos>} — an ordinary
     * token in the GGUF — where it ends a run of calls; both stop generation when tools are
     * attached, and neither changes a request without tools.
     */
    @Test
    public void toolAwareStopTokensAddTheToolResponseMarkerAndEos() {
        Gemma4Tokenizer tokenizer = Gemma4TestVocabulary.tokenizer();
        Gemma4ChatFormat format = new Gemma4ChatFormat(tokenizer);
        int endTurn = tokenizer.getSpecialTokens().get("<turn|>");
        int toolResponse = tokenizer.getSpecialTokens().get("<|tool_response>");
        int eos = tokenizer.tokenIndex("<eos>");
        assertFalse("<eos> is typed ordinary", tokenizer.getSpecialTokens().containsKey("<eos>"));
        assertEquals(Set.of(endTurn), format.getStopTokens());
        assertEquals(Set.of(endTurn, toolResponse, eos), format.getToolAwareStopTokens());
    }

    /** A call is read back from the response text, so the markers have to reach the text. */
    @Test
    public void theToolMarkersAreDisplayedAndTheTurnMarkersAreNot() {
        Gemma4Tokenizer tokenizer = Gemma4TestVocabulary.tokenizer();
        for (String marker :
                List.of(
                        "<|tool_call>",
                        "<tool_call|>",
                        "<|tool_response>",
                        "<tool_response|>",
                        Q)) {
            assertTrue(
                    marker, tokenizer.shouldDisplayToken(tokenizer.getSpecialTokens().get(marker)));
        }
        for (String hidden : List.of("<|turn>", "<turn|>", "<bos>", "<|channel>", "<|tool>")) {
            assertFalse(
                    hidden, tokenizer.shouldDisplayToken(tokenizer.getSpecialTokens().get(hidden)));
        }
    }

    private static String text(List<Gemma4ToolCalls.Piece> pieces) {
        return Gemma4ToolCalls.asText(pieces);
    }
}

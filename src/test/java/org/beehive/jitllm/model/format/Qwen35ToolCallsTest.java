package org.beehive.jllm.model.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Optional;
import org.junit.Test;

/**
 * The {@code qwen35} tool-call wire format, both directions.
 *
 * <p>The reason this has its own test rather than riding on Qwen 3's: the two formats are
 * incompatible and both are well-formed, so feeding one to a model trained on the other produces no
 * error anywhere — the model answers in prose and the parser finds nothing, which reads as "the
 * model chose not to call a tool".
 */
public class Qwen35ToolCallsTest {

    @Test
    public void parsesASingleCallWithOneParameter() {
        String response =
                """
                I'll look that up.
                <tool_call>
                <function=get_weather>
                <parameter=location>
                Boston
                </parameter>
                </function>
                </tool_call>""";
        Optional<ToolCallExtract> call = Qwen35ToolCalls.parseFirst(response);
        assertTrue(call.isPresent());
        assertEquals("get_weather", call.get().name());
        assertEquals("{\"location\":\"Boston\"}", call.get().argumentsJson());
    }

    /**
     * The format erases types, so they are recovered by shape: anything that is well-formed JSON
     * stays as it is, and everything else becomes a string.
     */
    @Test
    public void recoversNonStringArgumentsByShape() {
        String response =
                """
                <tool_call>
                <function=configure>
                <parameter=retries>
                3
                </parameter>
                <parameter=ratio>
                -1.5e2
                </parameter>
                <parameter=verbose>
                true
                </parameter>
                <parameter=options>
                {"deep":[1,2]}
                </parameter>
                <parameter=label>
                three
                </parameter>
                </function>
                </tool_call>""";
        ToolCallExtract call = Qwen35ToolCalls.parseFirst(response).orElseThrow();
        assertEquals(
                "{\"retries\":3,\"ratio\":-1.5e2,\"verbose\":true,"
                        + "\"options\":{\"deep\":[1,2]},\"label\":\"three\"}",
                call.argumentsJson());
    }

    /** A value's own newlines are its own; only the delimiters' newlines are removed. */
    @Test
    public void keepsMultiLineValuesIntact() {
        String response =
                """
                <tool_call>
                <function=write_file>
                <parameter=body>
                line one

                line three
                </parameter>
                </function>
                </tool_call>""";
        ToolCallExtract call = Qwen35ToolCalls.parseFirst(response).orElseThrow();
        assertEquals("{\"body\":\"line one\\n\\nline three\"}", call.argumentsJson());
    }

    @Test
    public void parsesSeveralCallsInOrder() {
        String response =
                """
                <tool_call>
                <function=first>
                <parameter=a>
                1
                </parameter>
                </function>
                </tool_call>
                <tool_call>
                <function=second>
                <parameter=b>
                2
                </parameter>
                </function>
                </tool_call>""";
        List<ToolCallExtract> calls = Qwen35ToolCalls.parseAll(response);
        assertEquals(2, calls.size());
        assertEquals("first", calls.get(0).name());
        assertEquals("second", calls.get(1).name());
    }

    /**
     * A model that ran out of budget mid-call has still said what it wants. Dropping that silently
     * would look like a model that decided not to call anything.
     */
    @Test
    public void parsesATruncatedCall() {
        String response =
                """
                <tool_call>
                <function=get_weather>
                <parameter=location>
                Boston""";
        ToolCallExtract call = Qwen35ToolCalls.parseFirst(response).orElseThrow();
        assertEquals("get_weather", call.name());
        assertEquals("{\"location\":\"Boston\"}", call.argumentsJson());
    }

    @Test
    public void findsNothingInAnOrdinaryAnswer() {
        assertEquals(List.of(), Qwen35ToolCalls.parseAll("The capital of France is Paris."));
        assertEquals(Optional.empty(), Qwen35ToolCalls.parseFirst(""));
        assertEquals(List.of(), Qwen35ToolCalls.parseAll(null));
    }

    /** Qwen 3's JSON body is not this format, and must not be mistaken for it. */
    @Test
    public void doesNotParseQwen3sJsonBody() {
        String response =
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Boston\"}}\n</tool_call>";
        assertEquals(List.of(), Qwen35ToolCalls.parseAll(response));
    }

    @Test
    public void rendersACallBackIntoTheWireFormat() {
        ToolCallExtract call =
                new ToolCallExtract("get_weather", "{\"location\":\"Boston\",\"days\":3}");
        assertEquals(
                """
                <function=get_weather>
                <parameter=location>
                Boston
                </parameter>
                <parameter=days>
                3
                </parameter>
                </function>""",
                Qwen35ToolCalls.renderFunctionBlock(call));
    }

    /** What is written must read back as what went in — for every argument shape. */
    @Test
    public void roundTripsThroughTheWireFormat() {
        for (String arguments :
                new String[] {
                    "{}",
                    "{\"a\":\"plain\"}",
                    "{\"n\":42,\"f\":-0.5,\"b\":false,\"z\":null}",
                    "{\"o\":{\"k\":[1,2,3]}}",
                    "{\"text\":\"has \\\"quotes\\\" and\\nnewlines\"}",
                    "{\"first\":\"one\",\"second\":2,\"third\":\"three\"}"
                }) {
            ToolCallExtract original = new ToolCallExtract("f", arguments);
            String wire =
                    "<tool_call>\n"
                            + Qwen35ToolCalls.renderFunctionBlock(original)
                            + "\n</tool_call>";
            ToolCallExtract parsed = Qwen35ToolCalls.parseFirst(wire).orElseThrow();
            assertEquals("name for " + arguments, "f", parsed.name());
            assertEquals("arguments for " + arguments, arguments, parsed.argumentsJson());
        }
    }
}

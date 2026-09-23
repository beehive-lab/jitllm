package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * The reasoning-mode option: model default, session override, and what each value means.
 *
 * <p>Class A — resolution is a function of two options and nothing else. What a family
 * <i>encodes</i> for a mode is exercised on a real model in {@code ThinkingModeAccelTest}.
 */
public class ThinkingModeTest {

    @Test
    public void theDefaultChangesNothing() {
        assertEquals(ThinkingMode.DEFAULT, ModelOptions.defaults().thinkingMode());
        assertFalse(
                "DEFAULT asks the model for nothing, so there is nothing to encode",
                ThinkingMode.DEFAULT.isExplicit());
        assertTrue(ThinkingMode.ENABLED.isExplicit());
        assertTrue(ThinkingMode.DISABLED.isExplicit());
    }

    @Test
    public void aSessionInheritsTheModelDefaultWhenItSaysNothing() {
        SessionOptions silent = SessionOptions.builder().build();
        assertNull(
                "null is 'I did not say', which is not the same as DEFAULT", silent.thinkingMode());
        assertEquals(ThinkingMode.DISABLED, silent.resolveThinkingMode(ThinkingMode.DISABLED));
        assertEquals(ThinkingMode.ENABLED, silent.resolveThinkingMode(ThinkingMode.ENABLED));
    }

    @Test
    public void aSessionOverrideWins() {
        SessionOptions override =
                SessionOptions.builder().thinkingMode(ThinkingMode.DISABLED).build();
        assertEquals(ThinkingMode.DISABLED, override.resolveThinkingMode(ThinkingMode.ENABLED));
    }

    /**
     * A session can return to the family's own behaviour even on a model configured otherwise.
     *
     * <p>This is why the session's field is nullable rather than defaulting to {@code DEFAULT}:
     * collapsing "I did not say" into "leave the family alone" would make this impossible to
     * express.
     */
    @Test
    public void aSessionCanExplicitlyChooseTheFamilysOwnBehaviour() {
        SessionOptions plain = SessionOptions.builder().thinkingMode(ThinkingMode.DEFAULT).build();
        assertEquals(ThinkingMode.DEFAULT, plain.resolveThinkingMode(ThinkingMode.ENABLED));
    }

    @Test
    public void aNullOnEitherBuilderMeansTheNeutralAnswerForThatLevel() {
        assertEquals(
                "a model with no stated mode leaves the family alone",
                ThinkingMode.DEFAULT,
                ModelOptions.builder().thinkingMode(null).build().thinkingMode());
        assertNull(
                "a session with no stated mode inherits",
                SessionOptions.builder().thinkingMode(null).build().thinkingMode());
    }

    /** No formatting detail is public: the mode is three values and nothing else. */
    @Test
    public void nothingAboutTheEncodingIsExposed() {
        assertEquals(
                java.util.EnumSet.of(
                        ThinkingMode.DEFAULT, ThinkingMode.ENABLED, ThinkingMode.DISABLED),
                java.util.EnumSet.allOf(ThinkingMode.class));
        for (var method : ThinkingMode.class.getDeclaredMethods()) {
            assertFalse(
                    "a token, template or format accessor would defeat the encapsulation: "
                            + method.getName(),
                    method.getName().toLowerCase().contains("token")
                            || method.getName().toLowerCase().contains("template")
                            || method.getName().toLowerCase().contains("format"));
        }
    }
}

package org.beehive.jitllm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.beehive.jitllm.api.GenerationRequest;
import org.junit.Test;

/**
 * An unset {@code --temperature} or {@code --top-p} is NaN in {@link Options}. Passed through to
 * the request, it divided every logit by NaN and the model produced nothing readable until the
 * context filled.
 */
public class JitllmAppRequestTest {

    private static GenerationRequest requestFor(String... args) {
        return JitllmApp.request(Options.parseOptions(args)).prompt("x").build();
    }

    @Test
    public void unsetSamplingSettingsFallBackToTheRequestDefaults() {
        GenerationRequest request = requestFor("-m", "model.gguf", "-p", "x");
        GenerationRequest defaults = GenerationRequest.builder().prompt("x").build();
        assertFalse(Float.isNaN(request.temperature()));
        assertFalse(Float.isNaN(request.topP()));
        assertEquals(defaults.temperature(), request.temperature(), 0f);
        assertEquals(defaults.topP(), request.topP(), 0f);
    }

    @Test
    public void explicitSamplingSettingsAreKept() {
        GenerationRequest request =
                requestFor("-m", "model.gguf", "-p", "x", "--temperature", "0.7", "--top-p", "0.5");
        assertEquals(0.7f, request.temperature(), 0f);
        assertEquals(0.5f, request.topP(), 0f);
    }
}

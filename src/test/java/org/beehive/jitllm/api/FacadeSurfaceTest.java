package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;

public class FacadeSurfaceTest {

    private static Set<String> methodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .map(Method::getName)
                .collect(Collectors.toSet());
    }

    @Test
    public void localModelIsGenerationNeutral() {
        Set<String> methods = methodNames(LocalModel.class);
        assertFalse(
                "newSession() belongs on the generation capability, not on LocalModel",
                methods.contains("newSession"));
        assertFalse("generation policy does not live on the model", methods.contains("generate"));
        assertFalse("execution lives below the model", methods.contains("forward"));
        assertEquals(Set.of("info", "configuration", "close"), methods);
    }

    @Test
    public void theGenerationCapabilityOwnsNewSession() {
        assertTrue(LocalModel.class.isAssignableFrom(TextGenerationModel.class));
        assertEquals(Set.of("newSession"), methodNames(TextGenerationModel.class));
    }

    @Test
    public void theSessionHasNoLowLevelForward() {
        Set<String> methods = methodNames(GenerationSession.class);
        assertFalse(
                "forward(token, position) is not part of the public surface",
                methods.contains("forward"));
        assertEquals(Set.of("generate", "prepare", "position", "reset", "close"), methods);
    }

    @Test
    public void theRequestTakesAPromptOrAConversationAndNeverATemplate() {
        Set<String> builderMethods = methodNames(GenerationRequest.Builder.class);
        assertTrue(builderMethods.contains("prompt"));
        assertTrue(builderMethods.contains("systemPrompt"));
        assertTrue(
                "messages(...) is the conversation surface: the whole conversation, not"
                        + " as a decision rather than by a test edit",
                builderMethods.contains("messages"));
        assertFalse(
                "chat formatting stays internal and model-driven",
                builderMethods.contains("chatTemplate") || builderMethods.contains("template"));
        assertFalse(
                "no chat format, stop-token set or template reaches the facade",
                builderMethods.contains("chatFormat") || builderMethods.contains("stopTokens"));
        assertEquals(
                "the request builder's surface is pinned exactly, so a tenth method is a"
                        + " failure rather than a surprise",
                Set.of(
                        "prompt",
                        "systemPrompt",
                        "messages",
                        "tools",
                        "maxNewTokens",
                        "temperature",
                        "topP",
                        "seed",
                        "stopSequences",
                        "onToken",
                        "onEvent",
                        "cancellation",
                        "build"),
                builderMethods);
    }

    /**
     * What the types are matters as much as that they exist: {@code backend} takes a {@code
     * BackendId} and {@code device} takes a {@code DeviceSelector} — <b>neutral selection
     * values</b>. A builder taking a resolved {@code Device}, or a backend implementation object,
     * would publish the backend through the facade, which is the thing this arrangement exists to
     * prevent.
     */
    @Test
    public void modelOptionsCarryBackendAndDeviceAsNeutralSelectionValues() {
        Set<String> builderMethods = methodNames(ModelOptions.Builder.class);
        assertEquals(
                Set.of(
                        "contextLength",
                        "executionPolicy",
                        "storageOptions",
                        "backend",
                        "device",
                        "thinkingMode",
                        "maxConcurrentSessions",
                        "build"),
                builderMethods);

        assertEquals(
                org.beehive.jitllm.runtime.backend.BackendId.class,
                parameterTypeOf(ModelOptions.Builder.class, "backend"));
        assertEquals(
                org.beehive.jitllm.runtime.backend.DeviceSelector.class,
                parameterTypeOf(ModelOptions.Builder.class, "device"));
    }

    private static Class<?> parameterTypeOf(Class<?> type, String method) {
        return java.util.Arrays.stream(type.getMethods())
                .filter(m -> m.getName().equals(method) && m.getParameterCount() == 1)
                .findFirst()
                .orElseThrow(() -> new AssertionError("no single-argument " + method))
                .getParameterTypes()[0];
    }

    @Test
    public void sessionOptionsCarryContextLengthAndThePolicyOverride() {
        assertEquals(
                Set.of("contextLength", "executionPolicy", "thinkingMode", "build"),
                methodNames(SessionOptions.Builder.class));
    }

    /**
     * Read from the class file rather than by reflection: the marker is {@code
     * RetentionPolicy.CLASS}, so it is deliberately invisible at runtime — it costs a dependent
     * nothing to run, and tools that care read the bytecode. This test reads the bytecode too.
     */
    @Test
    public void theExperimentalMarkersMatchTheAcceptedSurface() throws Exception {
        Class<?>[] stable = {
            LocalModels.class, LocalModel.class, TextGenerationModel.class,
            GenerationSession.class, GenerationRequest.class, GenerationResult.class,
            ModelOptions.class, SessionOptions.class,
        };
        Class<?>[] experimental = {
            GenerationTimings.class, FinishReason.class, ModelInfo.class, ModelConfiguration.class,
        };
        for (Class<?> type : stable) {
            assertFalse(
                    type.getSimpleName()
                            + " is frozen and must not carry a"
                            + " type-level @Experimental",
                    carriesMarker(type));
        }
        for (Class<?> type : experimental) {
            assertTrue(
                    type.getSimpleName() + " stays experimental and must say so",
                    carriesMarker(type));
        }
    }

    /** Whether a <b>type-level</b> {@code @Experimental} is present, read from source. */
    private static boolean carriesMarker(Class<?> type) throws Exception {
        java.nio.file.Path source =
                java.nio.file.Path.of("src/main/java", type.getName().replace('.', '/') + ".java");
        String body = java.nio.file.Files.readString(source);
        return java.util.regex.Pattern.compile(
                        "@Experimental\\s*\\n\\s*public (?:final |abstract |sealed )?"
                                + "(?:interface|class|enum|record) "
                                + type.getSimpleName()
                                + "\\b")
                .matcher(body)
                .find();
    }

    /**
     * The marker itself is not runtime-visible, so depending on the library costs nothing extra.
     */
    @Test
    public void theMarkerIsNotRetainedAtRuntime() {
        assertEquals(
                java.lang.annotation.RetentionPolicy.CLASS,
                Experimental.class.getAnnotation(java.lang.annotation.Retention.class).value());
    }

    /** Ownership is visible in the types: what holds device memory is closeable. */
    @Test
    public void whatOwnsResourcesIsCloseable() {
        assertTrue(AutoCloseable.class.isAssignableFrom(LocalModel.class));
        assertTrue(AutoCloseable.class.isAssignableFrom(GenerationSession.class));
    }
}

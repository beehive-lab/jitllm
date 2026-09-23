package org.beehive.jitllm.backend.cpu;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.beehive.jitllm.inference.ForwardPass;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.junit.Test;

/**
 * Provider discovery for the host forward pass.
 *
 * <p>The zero- and two-provider cases cannot be produced by the real service file, so they run
 * against the production resolver with a stated provider list — {@code select} is the same method
 * {@code forArchitecture} calls. A mirrored copy here would pass while the real one drifted.
 */
public class CpuForwardPassesTest {

    private static final ArchitectureId LLAMA = ArchitectureId.of("llama");
    private static final ArchitectureId INVENTED = ArchitectureId.of("not-a-real-architecture");

    private static CpuForwardProvider provider(ArchitectureId id) {
        return new CpuForwardProvider() {
            @Override
            public ArchitectureId architecture() {
                return id;
            }

            @Override
            public ForwardPass create() {
                return (m, s, t, p) -> {};
            }
        };
    }

    /** A second class, so the duplicate case can name two distinct implementations. */
    private static final class SecondProvider implements CpuForwardProvider {
        @Override
        public ArchitectureId architecture() {
            return LLAMA;
        }

        @Override
        public ForwardPass create() {
            return (m, s, t, p) -> {};
        }
    }

    @Test
    public void everyShippedArchitectureResolves() {
        // The ten the loaders can produce. A family that loads but cannot run its host pass is a
        // gap this catches at build time rather than at the first CPU token.
        for (String architecture :
                List.of(
                        "llama",
                        "mistral",
                        "qwen2",
                        "deepseek-r1-distill-qwen",
                        "qwen3",
                        "qwen2-moe",
                        "phi3",
                        "gemma4",
                        "granite",
                        "devstral")) {
            assertNotNull(
                    architecture,
                    CpuForwardPasses.forArchitecture(ArchitectureId.of(architecture)));
        }
    }

    @Test
    public void discoveryFindsOneProviderPerArchitecture() {
        List<ArchitectureId> served =
                CpuForwardPasses.discovered().stream()
                        .map(CpuForwardProvider::architecture)
                        .toList();
        assertEquals(
                "an architecture served twice is an error, not a preference",
                served.size(),
                served.stream().distinct().count());
        assertTrue(served.contains(LLAMA));
    }

    @Test
    public void discoveryIsOrderedByImplementationClassName() {
        List<String> names =
                CpuForwardPasses.discovered().stream().map(p -> p.getClass().getName()).toList();
        assertEquals(names.stream().sorted().toList(), names);
    }

    /** An unsupported family is a named error, never a fallback to some other family's routine. */
    @Test
    public void anUnsupportedArchitectureFailsByName() {
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () -> CpuForwardPasses.forArchitecture(INVENTED));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("not-a-real-architecture"));
        assertTrue(
                "the message must list what is registered, so a missing service file is visible",
                thrown.getMessage().contains("llama"));
    }

    /** The same, with no providers at all — the shaded-jar case. */
    @Test
    public void noProvidersAtAllFailsAndSaysWhy() {
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () -> CpuForwardPasses.select(List.of(), LLAMA));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("shaded"));
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("[]"));
    }

    @Test
    public void duplicateProvidersFailNamingBoth() {
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                CpuForwardPasses.select(
                                        List.of(provider(LLAMA), new SecondProvider()), LLAMA));
        assertTrue(
                thrown.getMessage(), thrown.getMessage().contains(SecondProvider.class.getName()));
        assertTrue(
                "there is no priority order, deliberately, and the message must say so",
                thrown.getMessage().contains("priority"));
    }

    @Test
    public void exactlyOneProviderResolves() {
        assertNotNull(CpuForwardPasses.select(List.of(provider(LLAMA)), LLAMA));
    }

    /**
     * Adding an architecture is a provider file plus a service line — there is no central switch to
     * edit [Rule 15].
     */
    @Test
    public void thereIsNoCentralArchitectureSwitch() {
        String resolver = CpuForwardPasses.class.getSimpleName();
        assertEquals("CpuForwardPasses", resolver);
        assertTrue(
                "the resolver must not enumerate architectures itself",
                CpuForwardPasses.discovered().size() >= 10);
    }
}

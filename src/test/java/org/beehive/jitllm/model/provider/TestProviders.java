package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;

/**
 * Providers that exist only to be discovered. They recognize synthetic metadata and refuse to load,
 * which is enough to test discovery, dispatch and ambiguity without a model file.
 */
public final class TestProviders {

    private TestProviders() {}

    /** Registered in {@code META-INF/services}; claims sources whose architecture is "fixture". */
    public static final class FixtureProvider implements ModelProvider {

        @Override
        public boolean supports(ModelSource source) {
            return "fixture".equals(source.metadata().get("general.architecture"));
        }

        @Override
        public ArchitectureId architecture(ModelSource source) {
            return ArchitectureId.of("fixture");
        }

        @Override
        public Model load(
                ModelSource source,
                org.beehive.jitllm.runtime.backend.BackendId backend,
                int contextLength) {
            throw new UnsupportedOperationException(
                    "the fixture provider recognizes, it does not load");
        }
    }

    /** Also registered, and claims a different architecture — discovery must find both. */
    public static final class OtherFixtureProvider implements ModelProvider {

        @Override
        public boolean supports(ModelSource source) {
            return "other-fixture".equals(source.metadata().get("general.architecture"));
        }

        @Override
        public ArchitectureId architecture(ModelSource source) {
            return ArchitectureId.of("other-fixture");
        }

        @Override
        public Model load(
                ModelSource source,
                org.beehive.jitllm.runtime.backend.BackendId backend,
                int contextLength) {
            throw new UnsupportedOperationException(
                    "the fixture provider recognizes, it does not load");
        }
    }
}

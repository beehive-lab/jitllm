package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.DevstralModelLoader;
import org.beehive.jitllm.model.loader.GraniteLoader;
import org.beehive.jitllm.model.loader.LlamaModelLoader;
import org.beehive.jitllm.model.loader.MistralModelLoader;
import org.beehive.jitllm.model.loader.Phi3ModelLoader;
import org.beehive.jitllm.model.loader.Qwen2MoEModelLoader;
import org.beehive.jitllm.model.loader.Qwen2ModelLoader;
import org.beehive.jitllm.model.loader.Qwen3ModelLoader;
import org.beehive.jitllm.runtime.backend.BackendId;

/**
 * The providers for the architectures this build knows, one per family.
 *
 * <p>These predate the SPI and are kept as one table: each is four lines saying the same three
 * things — which names it claims, what identity it gives them, which loader it calls — and
 * spreading eight of those across a package would gain nothing. Each is still a separate service
 * entry, so a family can be removed or replaced independently.
 *
 * <p>They delegate to today's loaders. Replacing the loaders is not this task; making the dispatch
 * discoverable is, so that adding a family stops meaning "edit a switch" (<a
 * href="./././././././docs/architecture/architecture.md">Rule 15</a>).
 */
public final class FamilyProviders {

    private FamilyProviders() {}

    public static final class LlamaProvider extends FamilyProvider {
        public LlamaProvider() {
            super("llama");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new LlamaModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    public static final class MistralProvider extends FamilyProvider {
        public MistralProvider() {
            super("mistral");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new MistralModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    public static final class DevstralProvider extends FamilyProvider {
        public DevstralProvider() {
            super("devstral");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new DevstralModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    public static final class Qwen2Provider extends FamilyProvider {
        public Qwen2Provider() {
            super("qwen2");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new Qwen2ModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    /**
     * Qwen2's mixture-of-experts sibling. It declares {@code qwen2moe}, not {@code qwen2}, and its
     * weight set is different enough (routed and shared experts, no dense FFN) that the Qwen2
     * loader cannot read it.
     */
    public static final class Qwen2MoEProvider extends FamilyProvider {
        public Qwen2MoEProvider() {
            super("qwen2moe");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new Qwen2MoEModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    public static final class Qwen3Provider extends FamilyProvider {
        public Qwen3Provider() {
            super("qwen3");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new Qwen3ModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    /**
     * A Qwen2-architecture distill with its own identity: the RoPE base differs from Qwen2.5's, so
     * calling it "qwen2" would be losing the distinction that made its GPU output wrong once
     * already.
     */
    public static final class DeepSeekR1DistillQwenProvider extends FamilyProvider {
        public DeepSeekR1DistillQwenProvider() {
            super("deepseek-r1-distill-qwen");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new Qwen2ModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    public static final class Phi3Provider extends FamilyProvider {
        public Phi3Provider() {
            super("phi3");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new Phi3ModelLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }

    /**
     * Granite is also recognizable without a name, by the presence of {@code granite.block_count};
     * {@link GgufRecognition} keeps that fallback, so a file whose {@code general.name} was
     * stripped still loads.
     */
    public static final class GraniteProvider extends FamilyProvider {
        public GraniteProvider() {
            super("granite");
        }

        @Override
        public Model load(ModelSource source, BackendId backend, int contextLength) {
            return new GraniteLoader(
                            source.gguf().getFileChannel(),
                            source.gguf(),
                            contextLength,
                            !BackendId.CPU.equals(backend))
                    .loadModel();
        }
    }
}

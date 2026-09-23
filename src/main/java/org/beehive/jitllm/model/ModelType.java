package org.beehive.jitllm.model;

import java.nio.channels.FileChannel;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.model.loader.DevstralModelLoader;
import org.beehive.jitllm.model.loader.Gemma4ModelLoader;
import org.beehive.jitllm.model.loader.GraniteLoader;
import org.beehive.jitllm.model.loader.LlamaModelLoader;
import org.beehive.jitllm.model.loader.MistralModelLoader;
import org.beehive.jitllm.model.loader.Phi3ModelLoader;
import org.beehive.jitllm.model.loader.Qwen2MoEModelLoader;
import org.beehive.jitllm.model.loader.Qwen2ModelLoader;
import org.beehive.jitllm.model.loader.Qwen35ModelLoader;
import org.beehive.jitllm.model.loader.Qwen3ModelLoader;

/**
 * Enumerates the different types of models supported by jitllm. This enum helps in categorizing and
 * handling model-specific logic based on the type of model being used.
 *
 * <p><b>Usage:</b> Use {@code ModelType} to specify or retrieve the type of large language model
 * (LLM), such as Llama or Qwen3. This ensures clean and structured handling of model behaviors and
 * configurations by dispatching calls to the appropriate model loader for each model type.
 *
 * <p>Each enum value represents a distinct model type, which might be used for conditional logic,
 * initialization, or resource allocation within jitllm.
 */
public enum ModelType {
    LLAMA_3 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new LlamaModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    MISTRAL {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new MistralModelLoader(fileChannel, gguf, contextLength, useTornadovm)
                    .loadModel();
        }
    },

    DEVSTRAL_2 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new DevstralModelLoader(fileChannel, gguf, contextLength, useTornadovm)
                    .loadModel();
        }
    },

    QWEN_2 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    QWEN_3 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen3ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    QWEN_2_MOE {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen2MoEModelLoader(fileChannel, gguf, contextLength, useTornadovm)
                    .loadModel();
        }
    },

    DEEPSEEK_R1_DISTILL_QWEN {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    PHI_3 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Phi3ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    GRANITE {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new GraniteLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    QWEN_3_5 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen35ModelLoader(fileChannel, gguf, contextLength, useTornadovm)
                    .loadModel();
        }
    },

    GEMMA_4 {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Gemma4ModelLoader(fileChannel, gguf, contextLength, useTornadovm)
                    .loadModel();
        }
    },

    UNKNOWN {
        @Override
        public Model loadModel(
                FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            throw new UnsupportedOperationException("Cannot load unknown model type");
        }
    };

    // Abstract method that each enum constant must implement
    public abstract Model loadModel(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm);

    public boolean isDeepSeekR1() {
        return this == DEEPSEEK_R1_DISTILL_QWEN;
    }
}

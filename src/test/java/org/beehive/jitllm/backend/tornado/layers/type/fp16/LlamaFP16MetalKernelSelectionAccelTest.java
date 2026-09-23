package org.beehive.jitllm.backend.tornado.layers.type.fp16;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * Metal parity, task 5→6 follow-up — regression coverage for capability-gated QKV/residual/FFN
 * kernel selection ({@code DeviceCapability.SUBGROUP_SHUFFLE_32}).
 *
 * <p>Constructing {@link LlamaFP16FFNLayers} snapshots every layer's {@code TaskGraph} inside its
 * constructor ({@code AbstractTransformerLayerTaskGraphs#setupFFNLayers}), which is what triggers
 * TornadoVM's sketch phase for {@code qkv_projection}, {@code attn_output_proj}, {@code
 * ffn_down_proj} and {@code rms_ffn_gate_up}. On a device without {@code SUBGROUP_SHUFFLE_32}, this
 * would select the generic kernels unchanged from before this task — the same ones production
 * already relies on. <b>On Metal specifically</b>, it selects {@code fusedQKVMatmulXSimd32}, {@code
 * matrixVectorGenericWithResidualSimd32} and {@code fusedRmsNormFFNGateUpWarp} instead: if a future
 * change silently reverted the selection to the generic kernels for Metal, this constructor call
 * would throw the exact {@code TornadoBailoutRuntimeException} sketch failure this task fixed, and
 * this test would catch it immediately rather than at a full golden/parity run.
 *
 * <p>Does not itself prove {@code vocab_proj} sketches — {@code LogitsFP16Layer}'s {@code
 * matrixVectorGeneric} is a separate, still-open blocker (no existing SIMD32/Warp sibling was found
 * for it) and is out of this test's scope.
 */
public class LlamaFP16MetalKernelSelectionAccelTest {

    @Test
    public void onMetalTheFixedLayerTaskGraphsSketchWithTheSimd32Kernels() throws Exception {
        assumeTrue("not a Metal run", SchedulerDetectionService.isSubgroupShuffle32Supported());

        Path model = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        assumeTrue("fixture absent", model != null);

        Model loaded = ModelLoader.loadModel(model, 512, true, true);
        State state = loaded.createNewState();
        LlamaTornadoWeights weights = (LlamaTornadoWeights) loaded.weights();
        LlamaConfiguration config = (LlamaConfiguration) loaded.configuration();
        var schedulerType = SchedulerDetectionService.determineSchedulerType(loaded);

        // Constructing this snapshots every layer's TaskGraph, sketching qkv_projection,
        // attn_output_proj, ffn_down_proj and rms_ffn_gate_up. A TornadoBailoutRuntimeException
        // here means the fix regressed - the assertion below is reached only if it did not.
        new LlamaFP16FFNLayers("llamaFFNRegressionProbe", state, weights, config, schedulerType);

        assertTrue(
                "this test only means something on a device with the verified capability",
                SchedulerDetectionService.isSubgroupShuffle32Supported());
    }
}

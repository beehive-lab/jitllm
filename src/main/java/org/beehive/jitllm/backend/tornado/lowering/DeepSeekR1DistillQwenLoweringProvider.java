package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * DeepSeek-R1-Distill-Qwen's lowering: Qwen2's implementation under its own identity.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families. Its service line is the only other thing an addition touches.
 */
public final class DeepSeekR1DistillQwenLoweringProvider implements TornadoLoweringProvider {

    private static final ArchitectureId ID = ArchitectureId.of("deepseek-r1-distill-qwen");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.STANDARD_ONLY;
    }

    @Override
    public FamilyLowering create(CompileOptions options, DeviceCapabilities capabilities) {
        return new Qwen2Lowering(options, capabilities);
    }
}

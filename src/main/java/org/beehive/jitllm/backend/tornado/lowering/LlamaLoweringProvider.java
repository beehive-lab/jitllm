package org.beehive.jllm.backend.tornado.lowering;

import java.util.Set;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * Llama's lowering — all three plan shapes, both representations.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families. Its service line is the only other thing an addition touches.
 */
public final class LlamaLoweringProvider implements TornadoLoweringProvider {

    private static final ArchitectureId ID = ArchitectureId.of("llama");

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
        return new LlamaLowering(options, capabilities, ID);
    }
}

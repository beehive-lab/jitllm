package org.beehive.jllm.backend.tornado.lowering;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.metrics.MetricsSink;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * This class and its service registration are the entire addition. No production file is edited to
 * make it discoverable — not a switch, not an enum, not a list — and {@code
 * TornadoBackendSupportTest} asserts it is found. If someone reintroduces a central table, the
 * assertion that this provider resolves keeps passing while the *reason* it passes changes; so the
 * companion assertion is that {@code TornadoBackendSupport} contains no family name, which is
 * checked by reading the source.
 *
 * <p>It lowers nothing: {@code create} returns a lowering that refuses. The test is about
 * <b>registration and validation</b>, and a synthetic thing that pretended to compile task graphs
 * would be testing the mock.
 */
public final class SyntheticLoweringProvider implements TornadoLoweringProvider {

    /** Deliberately unlike any real identity, so it cannot collide with a shipped provider. */
    public static final ArchitectureId ID = ArchitectureId.of("synthetic-test-architecture");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.F16);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return EnumSet.of(ExecutionMode.STANDARD);
    }

    @Override
    public FamilyLowering create(CompileOptions options, DeviceCapabilities capabilities) {
        return new FamilyLowering() {
            @Override
            public ArchitectureId architecture() {
                return ID;
            }

            @Override
            public void validate(InferenceProgram program) {
                throw new UnsupportedProgramException(
                        "the synthetic test architecture", "programs", "nothing", "a program");
            }

            @Override
            public TornadoVMMasterPlan lower(
                    InferenceProgram program, State state, Model model, MetricsSink sink) {
                throw new UnsupportedOperationException("the synthetic provider lowers nothing");
            }
        };
    }
}

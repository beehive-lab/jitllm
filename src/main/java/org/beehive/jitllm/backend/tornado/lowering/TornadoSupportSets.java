package org.beehive.jitllm.backend.tornado.lowering;

import java.util.EnumSet;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The capability sets providers reuse — <b>family-neutral</b>, so it names no architecture.
 *
 * <p>Shared constants are allowed to be central; a list of families is not. The difference is what
 * has to change when an architecture is added: nothing here.
 */
public final class TornadoSupportSets {

    /** The two representations that reach the device: a K-quant file arrives as {@code Q8_0}. */
    public static final Set<DataType> BOTH_REPRESENTATIONS = Set.of(DataType.F16, DataType.Q8_0);

    /** All three plan shapes. */
    public static final Set<ExecutionMode> EVERY_MODE = EnumSet.allOf(ExecutionMode.class);

    /** Single-token only, which is most families today. */
    public static final Set<ExecutionMode> STANDARD_ONLY = EnumSet.of(ExecutionMode.STANDARD);

    private TornadoSupportSets() {}
}

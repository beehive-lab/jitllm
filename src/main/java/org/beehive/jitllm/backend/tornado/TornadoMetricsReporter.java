package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.TornadoProfilerResult;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

/**
 * Turns TornadoVM's profiler output into {@link MetricsSink} records — the Tornado end of the Rule
 * 17 seam.
 *
 * <p>Everything reported here was already being produced by the runtime and discarded: {@code
 * TornadoExecutionResult.getProfilerResult()} carries device kernel time, host↔device transfer time
 * and bytes, and device memory in use. Nothing new is measured on the device.
 *
 * <h2>Cost</h2>
 *
 * <p>The enabled state is read once, at construction. A sink that switches itself on mid-run would
 * otherwise be asked for profiler results the plan was never configured to collect.
 */
public final class TornadoMetricsReporter {

    private final MetricsSink sink;
    private final boolean enabled;

    public TornadoMetricsReporter(MetricsSink sink) {
        this.sink = sink == null ? MetricsSink.disabled() : sink;
        this.enabled = this.sink.isEnabled();
    }

    /** Whether this reporter will do anything; useful to skip building values for it. */
    public boolean isEnabled() {
        return enabled;
    }

    /**
     * Switches the plan's profiler on when someone is listening. {@code SILENT} because the sink is
     * the output; the console modes would print a JSON block per execution.
     */
    public void enableOn(TornadoExecutionPlan plan) {
        if (enabled) {
            plan.withProfiler(ProfilerMode.SILENT);
        }
    }

    /**
     * Records one execution's device-side measurements and returns the result unchanged, so call
     * sites stay single expressions: {@code metrics.report(plan.withGraph(i).execute())}.
     */
    public TornadoExecutionResult report(TornadoExecutionResult result) {
        if (!enabled || result == null) {
            return result;
        }
        TornadoProfilerResult profile = result.getProfilerResult();
        sink.record(MetricKey.DEVICE_KERNEL_TIME, profile.getDeviceKernelTime());
        sink.record(MetricKey.DEVICE_WRITE_TIME, profile.getDeviceWriteTime());
        sink.record(MetricKey.DEVICE_READ_TIME, profile.getDeviceReadTime());
        sink.record(MetricKey.BYTES_COPIED_TO_DEVICE, profile.getTotalBytesCopyIn());
        sink.record(MetricKey.BYTES_COPIED_FROM_DEVICE, profile.getTotalBytesCopyOut());
        sink.record(MetricKey.DEVICE_MEMORY_USED, profile.getTotalDeviceMemoryUsage());
        return result;
    }

    /**
     * Records the one-off costs of standing the plan up. These are wall-clock measurements the plan
     * constructors already take for {@code RunMetrics}; they reach the sink as well so that an
     * embedder sees them without the CLI's printout.
     */
    public void reportSetUp(long planCreationNs, long jitNs, long weightUploadNs) {
        if (!enabled) {
            return;
        }
        sink.record(MetricKey.PLAN_CREATION_TIME, planCreationNs);
        sink.record(MetricKey.JIT_COMPILE_TIME, jitNs);
        sink.record(MetricKey.WEIGHT_UPLOAD_TIME, weightUploadNs);
    }
}

package org.beehive.jitllm.backend.tornado;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.backend.tornado.tensor.PackedWeights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.StorageOptions;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * Where a family's plan cannot run at all, whatever the key/value cache precision: on a device its
 * kernels are not written for, split across devices in a mode a split has no plan for or that the
 * family's stages cannot build, or under a policy its loaded weight representation cannot serve.
 *
 * <p>Asked before the cache and batched-prefill checks, which assume the plan can run, and before
 * any device buffer exists. Each family's plan provider names its own devices and splits ({@code
 * TornadoPlanProvider.deviceUnsupported}, {@code splitUnsupported}); this class names no family.
 */
public final class PlanSupport {

    private PlanSupport() {}

    /**
     * Why {@code architecture}'s plan for {@code weights} cannot run under {@code policy} on {@code
     * devices} — one device, or each stage's of a split — or empty when it can.
     */
    public static Optional<String> unsupported(
            String architecture,
            DataType weights,
            ExecutionPolicy policy,
            List<BackendId> devices,
            boolean split) {
        if (split) {
            Optional<String> mode = splitUnsupported(policy);
            if (mode.isPresent()) {
                return mode;
            }
        }
        for (BackendId backend : devices) {
            Optional<String> device = deviceUnsupported(architecture, backend);
            if (device.isPresent()) {
                return device;
            }
        }
        if (split) {
            return TornadoPlanRegistry.provider(architecture)
                    .flatMap(p -> p.splitUnsupported(weights, policy, devices.size()));
        }
        return Optional.empty();
    }

    /** Why a model split across devices cannot run {@code policy}, if it cannot. */
    static Optional<String> splitUnsupported(ExecutionPolicy policy) {
        return ExecutionMode.of(policy) == ExecutionMode.PREFILL_DECODE
                ? Optional.of(
                        "a model split across devices (--devices) runs single-token or with"
                                + " batched prefill (--batch-prefill-size N, N > 1), not"
                                + " sequential prefill/decode; choose one of them or drop"
                                + " --devices")
                : Optional.empty();
    }

    /** Why {@code architecture}'s plans cannot run on a {@code backend} device, if they cannot. */
    static Optional<String> deviceUnsupported(String architecture, BackendId backend) {
        if (BackendId.CPU.equals(backend)) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(architecture)
                .flatMap(provider -> provider.deviceUnsupported(backend))
                .map(reason -> "on " + backend + ", " + reason);
    }

    /**
     * Refuses a plan {@code model} cannot run under {@code policy}: on the device it would build
     * for, on each stage's device of a split, or in a mode a split cannot run.
     *
     * @throws UnsupportedOperationException naming the model and the reason
     */
    // @formatter:off
    /**
     * Refuses a configuration that any plan check refuses, before any device buffer exists. The one
     * list of checks every entry point runs, in this order:
     *
     * <ol>
     *   <li>the family's device and split support, and packed weights;
     *   <li>the FP16 key/value cache, which checks batched prefill first itself;
     *   <li>native libraries;
     *   <li>batched prefill;
     *   <li>the provider's weight type and execution mode.
     * </ol>
     *
     * @throws UnsupportedOperationException or, from the FP16 cache check, {@link
     *     IllegalArgumentException}, naming the reason
     */
    // @formatter:on
    public static void require(
            Model model, ExecutionPolicy policy, StorageOptions storage, boolean gpu) {
        requireDevices(model, policy, gpu);
        Fp16KeyValueSupport.require(model, policy, storage, gpu);
        NativeLibrarySupport.require(model, policy, gpu);
        BatchPrefillSupport.require(model, policy, gpu);
        if (gpu) {
            TornadoPlanRegistry.unsupported(
                            model, model.weights().dataType(), ExecutionMode.of(policy))
                    .ifPresent(
                            reason -> {
                                throw new UnsupportedOperationException(reason);
                            });
        }
    }

    private static void requireDevices(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!gpu) {
            return;
        }
        Optional<DeviceSplit> split = DeviceSplit.fromSystemProperties();
        List<BackendId> devices = new ArrayList<>();
        if (split.isPresent()) {
            for (TornadoDevice device : TornadoVMMasterPlanPipeline.stageDevices(split.get())) {
                devices.add(TornadoDevices.of(device).backend());
            }
        } else {
            devices.add(TornadoDevices.target().backend());
        }
        String architecture = model.architectureId().toString();
        unsupported(architecture, model.weights().dataType(), policy, devices, split.isPresent())
                .or(() -> packedUnsupported(architecture, packed(model), policy))
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(
                                    refusal(architecture, model, reason));
                        });
    }

    /**
     * Why projections a load packed cannot run under {@code policy}, or empty when they can or
     * nothing is packed. The representation is chosen once, at load, for the policy the model was
     * loaded with; a session that asks for a policy it cannot serve is refused before any device
     * buffer, rather than failing in its first generation.
     */
    static Optional<String> packedUnsupported(
            String architecture, boolean packed, ExecutionPolicy policy) {
        if (!packed) {
            return Optional.empty();
        }
        return PackedWeights.unsupported(policy)
                .or(
                        () ->
                                TornadoPlanRegistry.provider(architecture)
                                        .flatMap(p -> p.packedWeightsUnsupported(policy)))
                .map(
                        reason ->
                                "the model was loaded with packed projections and "
                                        + reason
                                        + "; load it with this execution policy"
                                        + " (ModelOptions.executionPolicy), or with"
                                        + " -Djitllm.q4.packed=false -Djitllm.q8.packed=false");
    }

    private static boolean packed(Model model) {
        return model.weights() instanceof TornadoWeights weights
                && weights.holdsPackedProjections();
    }

    static String refusal(String architecture, Model model, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                architecture + " / " + model.weights().dataType() + " cannot run: " + reason);
    }
}

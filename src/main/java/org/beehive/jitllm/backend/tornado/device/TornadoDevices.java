package org.beehive.jitllm.backend.tornado.device;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.Device;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.backend.DeviceId;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * The one place this process asks TornadoVM what it is running on.
 *
 * <p>Before this, four call sites asked independently — {@code LoweredPlanSelection}'s device label
 * for the cache key, {@code SchedulerDetectionService} for the scheduler type and two backend
 * predicates, {@code TensorCoreSupport} for MMA, and {@code JitllmBench} inline for a report
 * heading — and all four pinned {@code getBackend(0).getDefaultDevice()}. Four answers that must
 * agree, derived four times, is a disagreement waiting for a machine with two backends installed.
 *
 * <p><b>Resolved once, lazily, and cached.</b> That is not an optimization: a cache-key component
 * must not change underneath the cache. {@code LoweredPlanSelection} learned this the expensive way
 * — it used to read {@code System.getProperty("tornado.device")}, which TornadoVM sets while
 * initializing, so it read {@code "default"} before the first plan and {@code "0"} after, and the
 * same program keyed differently in the first session than in every later one. Asking the runtime
 * rather than the property, and asking once, is the fix.
 *
 * <p><b>No accelerator is a normal answer.</b> Resolution never throws: a machine without a device
 * gets a stable placeholder identity and an empty capability set, so a caller that only wants a
 * label for a log line does not have to handle an exception, and a program compiled without a
 * device is one nothing will execute anyway.
 */
public final class TornadoDevices {

    /**
     * What the label was before a device could be resolved. Kept verbatim: it reaches cache keys.
     */
    private static final String UNAVAILABLE = "unavailable";

    private TornadoDevices() {}

    /** The device this process executes on. Resolved on first use and stable thereafter. */
    public static Device current() {
        return Holder.DEVICE;
    }

    private static final class Holder {
        private static final Device DEVICE = resolve();

        private static Device resolve() {
            try {
                var backend = TornadoRuntimeProvider.getTornadoRuntime().getBackend(0);
                TornadoVMBackendType type = backend.getBackendType();
                var device = backend.getDefaultDevice();
                String platformName = device.getPlatformName();
                String deviceInfo = "";
                long maxWorkGroup = 0L;
                try {
                    deviceInfo = device.getPhysicalDevice().getDeviceInfo();
                    maxWorkGroup = maxWorkGroupOf(device.getPhysicalDevice());
                } catch (RuntimeException e) {
                    // A device that cannot describe itself gets no architecture-gated grants
                    // and no known workgroup limit.
                }
                return new ResolvedDevice(
                        backendId(type),
                        platformName,
                        capabilitiesOf(type, platformName, deviceInfo),
                        TornadoNativeArray.ARRAY_HEADER,
                        maxWorkGroup);
            } catch (RuntimeException | LinkageError e) {
                // No accelerator present. The identity still has to be stable and comparable.
                // No accelerator: no native-array header either, which is what a caller mapping
                // for it should reserve.
                return new ResolvedDevice(BackendId.CPU, UNAVAILABLE, DeviceCapabilities.NONE, 0L);
            }
        }
    }

    /**
     * TornadoVM's backend type as a {@link BackendId}. An unrecognised type keeps its own name
     * rather than collapsing to a default — a backend this project has not heard of is still a
     * distinct backend, and merging it into another would make two of them share cache entries.
     *
     * <p>NVIDIA devices arrive as {@code CUDA}. TornadoVM removed its separate PTX backend in
     * favour of CUDA, so there is no PTX case to write and naming the constant would not compile.
     */
    private static BackendId backendId(TornadoVMBackendType type) {
        return switch (type) {
            case OPENCL -> BackendId.OPENCL;
            case METAL -> BackendId.METAL;
            default -> BackendId.of(type.name());
        };
    }

    /**
     * The four device facts the tree branches on, each preserving exactly the predicate it
     * replaces.
     *
     * <ul>
     *   <li><b>warp shuffle</b> — <b>granted to nothing today.</b> The predicate was "PTX only",
     *       written when NVIDIA devices arrived under a separate PTX backend; TornadoVM has since
     *       folded PTX into CUDA, so the test matched no device and the warp GEMVs were never
     *       selected. Granting it to CUDA is correct — the 11 CPU-parity cases pass either way —
     *       but measurably slower on the one device it was tried on (RTX 5090 Laptop, Qwen3-1.7B
     *       decode: 141 to 134 tok/s in FP16, 149 to 125 in Q8_0), so it stays off deliberately
     *       rather than by accident. The OpenCL backend is a separate matter: it compiles {@code
     *       KernelContext.simdShuffleDown} and computes the wrong answer, so it must never have
     *       this capability regardless of speed.
     *   <li><b>tensor-core MMA</b> — CUDA only; TornadoVM lowers the MMA intrinsics nowhere else.
     *   <li><b>packed integer dot</b> — CUDA, where the {@code dp4a} path was measured <i>and</i>
     *       where the warp shuffle the packed kernels reduce with is correct. The instruction is
     *       lowered on the other backends as well and its Java body is correct everywhere, so that
     *       half withholds a preference rather than a result; the reduction half does not, and is
     *       why this must never be granted on OpenCL.
     *   <li><b>split-KV attention</b> — everywhere except Metal, which fails to JIT {@code
     *       processHeadsFlashAttentionSplitKV}.
     *   <li><b>single-pass RMS</b> — the device half of the scheduler type: an NVIDIA platform.
     *       Elsewhere lowering emits an extra {@code *_rms_finalize} task per block. The model half
     *       of that decision is not a device fact and stays in {@code SchedulerDetectionService}.
     *   <li><b>shuffle-reduced FP16 GEMV</b> — CUDA, where the 32-lane butterfly is lowered
     *       correctly and checked by the per-family CPU-parity gates. A correctness grant; the
     *       preference that reads it is {@code Fp16GemvReductionPolicy}.
     *   <li><b>32-wide subgroup shuffle</b> — Metal only, and deliberately not the same grant as
     *       warp shuffle above: verified for exactly the fused Q/K/V projection kernel's five-step
     *       butterfly reduction (Metal parity task, {@code DeviceCapability.SUBGROUP_SHUFFLE_32}),
     *       not for warp shuffle in general. CUDA and OpenCL are unaffected by this grant.
     * </ul>
     */
    static DeviceCapabilities capabilitiesOf(
            TornadoVMBackendType type, String platformName, String deviceInfo) {
        Set<DeviceCapability> capabilities = new HashSet<>();
        String name = platformName.toLowerCase(Locale.ROOT);
        if (type == TornadoVMBackendType.CUDA) {
            // Granted by what the device can execute, not by the backend alone: the tensor-core
            // families need mma.sync m16n8k16 (FP16) / m16n8k32 (int8) with ldmatrix and
            // cp.async, which is compute capability 8.0 and up; the packed-integer path needs
            // dp4a, 6.1 and up. An unreadable capability grants nothing, and the scalar kernels
            // run everywhere.
            int sm = cudaComputeCapability(deviceInfo);
            if (sm >= 80) {
                capabilities.add(DeviceCapability.TENSOR_CORE_MMA);
                capabilities.add(DeviceCapability.INT8_TENSOR_CORE_MMA);
            }
            // Support only: the shuffle-reducing FP16 matrix-vector kernels compute correct
            // results on CUDA, which the CPU-parity gates check for every FP16 family. Whether to
            // PREFER them is a workload question and is not decided here -- see
            // Fp16GemvReductionPolicy. Deliberately not WARP_SHUFFLE, which bundles the same
            // support question with a preference for a wider set of kernels.
            capabilities.add(DeviceCapability.SHUFFLE_REDUCED_FP16_GEMV);
            // dp4a is registered for OpenCL and Metal too, and its Java body is a correct scalar
            // fallback everywhere, so the instruction half of this grant is about where the packed
            // path has been measured rather than about where it computes the right answer. The
            // reduction half is not: the packed kernels reduce with simdShuffleDown, which OpenCL
            // miscompiles, so on that backend this grant would be wrong rather than merely
            // unmeasured. CUDA is the one backend where both halves hold.
            if (sm >= 61) {
                capabilities.add(DeviceCapability.PACKED_INTEGER_DOT);
            }
        }
        if (type != TornadoVMBackendType.METAL) {
            capabilities.add(DeviceCapability.SPLIT_KV_ATTENTION);
        }
        if (name.contains("nvidia") || name.contains("cuda")) {
            capabilities.add(DeviceCapability.SINGLE_PASS_RMS);
        }
        if (type == TornadoVMBackendType.METAL) {
            capabilities.add(DeviceCapability.SUBGROUP_SHUFFLE_32);
        }
        // Withheld on OpenCL only: the packed FP16 multiply rounds every product to FP16 before
        // it is accumulated, and on OpenCL that costs enough accuracy for the Llama-shaped FP16
        // QKV projection to fail CPU parity. The identical kernel holds parity on CUDA, so this
        // is a device property, not a kernel defect, and every other backend keeps the fast path.
        if (type != TornadoVMBackendType.OPENCL) {
            capabilities.add(DeviceCapability.PACKED_HALF2_MATH);
        }
        return DeviceCapabilities.of(capabilities);
    }

    /**
     * The CUDA compute capability as major*10+minor, read from the device's own description —
     * TornadoVM's CUDA device reports it as "Device version : CUDA X.Y" — or -1 when it cannot be
     * read. Package-private for the unit test.
     */
    static int cudaComputeCapability(String deviceInfo) {
        if (deviceInfo == null) {
            return -1;
        }
        java.util.regex.Matcher m =
                java.util.regex.Pattern.compile("Device version\\s*:\\s*CUDA\\s+(\\d+)\\.(\\d+)")
                        .matcher(deviceInfo);
        if (!m.find()) {
            return -1;
        }
        return Integer.parseInt(m.group(1)) * 10 + Integer.parseInt(m.group(2));
    }

    /**
     * The device's threads-per-block limit: the smallest of what it reports as its maximum
     * workgroup size (one dimension) and its maximum threads per block, or 0 when it reports
     * neither.
     */
    static long maxWorkGroupOf(uk.ac.manchester.tornado.api.TornadoTargetDevice physical) {
        long limit = 0L;
        long[] sizes = physical.getDeviceMaxWorkGroupSize();
        if (sizes != null && sizes.length > 0 && sizes[0] > 0) {
            limit = sizes[0];
        }
        int threads = physical.getMaxThreadsPerBlock();
        if (threads > 0) {
            limit = limit == 0 ? threads : Math.min(limit, threads);
        }
        return limit;
    }

    private record ResolvedDevice(
            DeviceId id,
            String displayName,
            DeviceCapabilities capabilities,
            long nativeArrayHeaderBytes,
            long maxWorkGroupSize)
            implements Device {

        ResolvedDevice(
                BackendId backend,
                String platformName,
                DeviceCapabilities capabilities,
                long nativeArrayHeaderBytes) {
            this(
                    DeviceId.of(backend, platformName),
                    platformName,
                    capabilities,
                    nativeArrayHeaderBytes,
                    0L);
        }

        ResolvedDevice(
                BackendId backend,
                String platformName,
                DeviceCapabilities capabilities,
                long nativeArrayHeaderBytes,
                long maxWorkGroupSize) {
            this(
                    DeviceId.of(backend, platformName),
                    platformName,
                    capabilities,
                    nativeArrayHeaderBytes,
                    maxWorkGroupSize);
        }
    }
}

package org.beehive.jllm.backend.tornado.memory;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.NativePrefillSupport;
import org.beehive.jllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jllm.backend.tornado.plan.layout.TornadoGraphTopology;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.backend.Device;
import org.beehive.jllm.runtime.memory.BufferClass;
import org.beehive.jllm.runtime.memory.MemoryComponent;
import org.beehive.jllm.runtime.memory.MemoryPlan;
import org.beehive.jllm.runtime.memory.WeightFootprint;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/**
 * The Tornado backend's memory-plan model — <b>the only place that knows what a task graph
 * costs</b>.
 *
 * <p>The neutral layer describes logical buffers and who owns them. Everything Tornado-specific
 * lives here: the native-array header, the per-graph allocation rule, and which execution modes
 * build which graph families.
 *
 * <h2>Multiplicity is derived, not assumed</h2>
 *
 * <p>{@code LocalObjectState} is held on {@code TornadoExecutionContext}, which is per task graph,
 * so the same Java array bound into two graphs is allocated twice. <b>Graph count alone is not the
 * rule</b> — sequential prefill lays out N+2 graphs with <b>one</b> layer family, the same shape
 * single-token uses, and costs the same as single-token. Batched prefill lays out 2N+3 with
 * <b>two</b>. The families are what differ, and the layouts already record it.
 *
 * <p>What matters is the number of distinct <b>layer graph families</b>: batched prefill is the
 * only mode that builds a second one (`…LayersBatchPrefill…` alongside the decode layers), and both
 * bind the per-layer weights with {@code FIRST_EXECUTION}.
 *
 * <h2>Measured, not asserted</h2>
 *
 * <p>Minimum successful {@code -Dtornado.device.memory}, bisected in a fresh JVM per probe, {@code
 * Llama-3.2-1B-Instruct}, ctx 512, CUDA:
 *
 * <pre>
 *   F16  single-token         2362 MiB      Q8_0 single-token   1256 MiB
 *   F16  sequential prefill   2365 MiB
 *   F16  batched prefill (8)  4234 MiB      Q8_0 batched (8)    2262 MiB
 * </pre>
 *
 * <p>The batched/single difference is <b>1872 MiB</b> for F16 against <b>1856 MiB</b> of per-layer
 * weights — and the full weight set is 2357 MiB, which would not have matched. That is the evidence
 * the duplication is per-layer and not global. Q8_0 repeats it: 1006 MiB measured against ~986 MiB
 * of per-layer weights, with the same ~16–20 MiB of batch staging on top.
 */
public final class TornadoMemoryModel {

    /** Layer graph families for this policy — <b>read from the layout, not declared here</b>. */
    private static int layerGraphFamilies(ExecutionPolicy policy, int layers) {
        return TornadoGraphTopology.layerGraphFamilies(executionMode(policy), layers);
    }

    /** The backend's execution mode for a policy — the same mapping plan selection makes. */
    private static ExecutionMode executionMode(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return ExecutionMode.STANDARD;
        }
        return policy.prefillBatchSize() > 1
                ? ExecutionMode.BATCH_PREFILL_DECODE
                : ExecutionMode.PREFILL_DECODE;
    }

    private TornadoMemoryModel() {}

    // @formatter:off
    /**
     * Whether this session will resolve to a native batch-prefill family.
     *
     * <p>The same answer the planner asks for, so the prediction and the plan cannot disagree about
     * which buffers exist. It reaches the capability probe, which dlopens cuBLAS — on a host that
     * is about to build exactly that plan, and behind the tensor-core gate that makes it a no-op
     * everywhere else.
     *
     * <p><b>The fallback family is deliberately not consulted.</b> It is a third layer graph family
     * and it allocates three carriers and copies nothing: every weight, workspace and cache it
     * touches is bound from the primary family's buffers. Letting a family count raise any
     * multiplicity here would charge the session a second copy of the weights for graphs that share
     * one.
     *
     * <p>Only the projections are consulted, not the fused attention. The attention probe needs a
     * device and a batch width resolved, and the staging it would add is at most 8 MiB at the
     * widest supported chunk — not worth threading a second capability question through the
     * preflight to shave.
     */
    // @formatter:on
    private static boolean nativePrefillSelected(ExecutionPolicy policy) {
        return executionMode(policy) == ExecutionMode.BATCH_PREFILL_DECODE
                && NativePrefillSupport.nativeProjections();
    }

    /**
     * Predicts the device budget a configuration will consume, before anything is allocated.
     *
     * @param weights the model's weight footprint, split per-layer and global
     * @param config the resolved model configuration
     * @param policy the session's execution policy, which selects the graph topology
     * @param device the resolved device, for its native-array header width
     * @param configuredBudgetBytes the backend's configured budget, or 0 when unknown
     */
    public static MemoryPlan predict(
            WeightFootprint weights,
            Configuration config,
            ExecutionPolicy policy,
            Device device,
            long configuredBudgetBytes) {
        long header = device.nativeArrayHeaderBytes();
        int layoutFamilies = layerGraphFamilies(policy, config.numberOfLayers());
        // How many of those families upload the weights, which is what costs memory. A family
        // that consumes another's copy costs graphs and not gigabytes.
        int families = config.weightBindingFamilies(layoutFamilies);
        List<MemoryComponent> components = new ArrayList<>();

        // ── weights, classified by whether a layer graph binds them ──────────
        long perLayer = weights.perLayerBytes();
        long global = weights.globalBytes();
        components.add(
                new MemoryComponent(
                        "weights (per-layer)",
                        BufferClass.WEIGHTS_PER_LAYER,
                        perLayer,
                        families,
                        (long) weights.perLayerTensors() * header * families));
        // Global weights include the tied embedding/output pair. Llama-3.2-1B ties them and the two
        // wrappers share ONE segment — verified by segment address — so this counts the storage
        // once, which is what the descriptor set already does.
        components.add(
                new MemoryComponent(
                        "weights (embeddings, output, RoPE)",
                        BufferClass.WEIGHTS_GLOBAL,
                        global,
                        1,
                        (long) weights.globalTensors() * header));

        // ── key/value cache ──────────────────────────────────────────────────
        // The layers that actually hold key/value entries, which is every layer for every family
        // but qwen35 — where it is one in four, and the layer count would predict four times the
        // store that is allocated.
        long kvElements =
                (long) config.contextLength() * config.keyValueLayerCount() * config.kvDim();
        // FP16 KV is a storage choice, so it must be read rather than assumed FP32 — assuming
        // FP32 would over-predict a configured FP16 cache by exactly its own size.
        int kvElementBytes = kvBytesPerElement();
        components.add(
                new MemoryComponent(
                        "key/value cache",
                        BufferClass.KV_CACHE,
                        kvElements * 2L * kvElementBytes,
                        1,
                        2 * header));

        // ── fixed activation and attention workspace ─────────────────────────
        components.add(
                new MemoryComponent(
                        "activation workspace",
                        BufferClass.ACTIVATION_WORKSPACE,
                        activationWorkspaceBytes(config),
                        1,
                        24 * header));

        // ── recurrent state, for a family that keeps one ─────────────────────
        // Zero for every attention-only stack, and 151 MiB for Qwen3.8-27B. It is neither cache
        // nor scratch: fixed-size per layer, updated in place, and independent of the context
        // length, so neither of the two components above accounts for it.
        if (config.recurrentStateBytes() > 0) {
            components.add(
                    new MemoryComponent(
                            "recurrent state",
                            BufferClass.ACTIVATION_WORKSPACE,
                            config.recurrentStateBytes(),
                            1,
                            2 * header));
        }

        // ── batch staging, only when a batched capacity is configured ────────
        if (policy.phaseStrategy() == ExecutionPolicy.PhaseStrategy.PREFILL_DECODE
                && policy.prefillBatchSize() > 1) {
            components.add(
                    new MemoryComponent(
                            "batch staging",
                            BufferClass.BATCH_STAGING,
                            batchStagingBytes(config, policy.prefillBatchSize())
                                    + config.additionalBatchWorkspaceBytes(
                                            policy.prefillBatchSize()),
                            1,
                            11 * header));
        }

        // ── what a native batch-prefill family adds ──────────────────────────
        // Measured rather than reasoned, from -Dtornado.print.bytecodes on Qwen3-0.6B FP16 at
        // width 128, native on against native off:
        //
        //   batch-prefill graphs   914.5 -> 916.5 MiB   (+2.0, the staging quartet)
        //   decode layer graphs      0.1 -> 560.1 MiB   (+560.0, five weights x 28 layers)
        //   fallback family         absent -> 0.0 MiB   (three carriers, nothing uploaded)
        //
        // The stacked copies do not show up as growth in the prefill family because they displace
        // exactly the five originals it stopped reading. Those five reappear in the decode family,
        // which now uploads them itself -- so the cost of the native path is one extra copy of the
        // projection weights, plus the staging, and that is what these two components describe.
        boolean nativePrefill = nativePrefillSelected(policy);
        long stackedPerLayer = config.nativeStackedProjectionBytesPerLayer();
        if (nativePrefill && stackedPerLayer > 0) {
            components.add(
                    new MemoryComponent(
                            "stacked projection weights (native prefill)",
                            BufferClass.WEIGHTS_PER_LAYER,
                            stackedPerLayer * config.numberOfLayers(),
                            1,
                            // Two stacked arrays per layer, each its own device buffer.
                            2L * config.numberOfLayers() * header));
            long staging = config.nativeAttentionStagingBytes(policy.prefillBatchSize());
            if (staging > 0) {
                components.add(
                        new MemoryComponent(
                                "fused attention staging (native prefill)",
                                BufferClass.BATCH_STAGING,
                                staging,
                                1,
                                4 * header));
            }
        }

        // ── control and result carriers ──────────────────────────────────────
        components.add(
                new MemoryComponent(
                        "control carriers",
                        BufferClass.CONTROL,
                        4L * Integer.BYTES * 8,
                        1,
                        4 * header));
        components.add(
                new MemoryComponent(
                        "logits and sampling",
                        BufferClass.RESULT,
                        (long) config.vocabularySize() * Float.BYTES + Integer.BYTES,
                        1,
                        2 * header));

        // EXACT only when the topology agrees with its own layout arithmetic. A layout that grew
        // a layer family without saying so downgrades the plan instead of keeping a stale EXACT,
        // and admission does not enforce on anything but EXACT.
        //
        // The multiplicity model and the per-array header counts above were bisected against
        // measurement on CUDA only. Nothing has measured whether Metal's unified memory charges
        // the same quantity, so Metal is capped at CONSERVATIVE rather than claiming EXACT from
        // CUDA-derived assumptions; every other backend keeps the topology-only rule.
        boolean measuredOnThisBackend = device.backend() != BackendId.METAL;
        // A native batch-prefill family is NOT covered by the bisected multiplicity model above.
        // Its two components are sized from direct allocation evidence, but the driver reservation
        // that evidence sits inside has not been bisected for this path, and the per-layer rule it
        // is added to over-counts a family whose decode graphs consume rather than upload. Both
        // errors are in the safe direction, and an over-prediction that kept EXACT would be
        // enforced: only EXACT refuses a load. So this path reports what it is -- conservative.
        MemoryPlan.Confidence confidence =
                (measuredOnThisBackend
                                && !nativePrefill
                                && TornadoGraphTopology.verify(
                                        executionMode(policy), config.numberOfLayers()))
                        ? MemoryPlan.Confidence.EXACT
                        : MemoryPlan.Confidence.CONSERVATIVE;
        return new MemoryPlan(
                components,
                configuredBudgetBytes,
                confidence,
                "tornado backend; "
                        + families
                        + " layer graph "
                        + (families == 1 ? "family" : "families")
                        + "; context "
                        + config.contextLength()
                        + "; kv "
                        + (kvBytesPerElement() == 2 ? "FP16" : "FP32")
                        + (nativePrefill && stackedPerLayer > 0 ? "; native prefill" : "")
                        + "; native-array header "
                        + header
                        + " B");
    }

    /**
     * Bytes per key/value element, from the selected storage representation.
     *
     * <p>Read from the same switch the state reads, so the prediction and the allocation cannot
     * disagree about which representation was chosen.
     */
    private static int kvBytesPerElement() {
        return org.beehive.jllm.inference.state.State.USE_FP16_KV ? 2 : 4;
    }

    /**
     * The fixed per-session activation and attention scratch.
     *
     * <p>Derived from the transformer's own dimensions rather than from a table of field names, so
     * a family that adds a buffer is under-counted rather than mis-counted — and the components are
     * gated individually so that under-count is visible instead of being absorbed by the weights.
     */
    private static long activationWorkspaceBytes(Configuration config) {
        long dim = config.dim();
        long hidden = config.hiddenDim();
        long kvDim = config.kvDim();
        // Two attention buffers where a family decodes with the split-KV decomposition: the
        // scalar scratch of heads x context, and the split scratch holding per head nSplits
        // partial numerators of headSize plus nSplits maxima and sums. The term below counted the
        // second as another copy of the first; at a wide head and a short context the split
        // scratch is the larger, so it is now sized from what is actually allocated.
        long scalarAttention = (long) config.numberOfHeads() * config.contextLength();
        long splitKvAttention =
                (long) config.numberOfHeads()
                        * org.beehive.jllm.inference.state.State.SPLIT_KV
                        * (config.headSize() + 2L);
        long floats =
                dim * 6 // x, xb, xb2, q, and two spare dim-sized activations
                        + hidden * 2 // hb, hb2
                        + kvDim * 2 // k, v
                        + scalarAttention // att
                        + splitKvAttention // the split-KV partials
                        + dim * 2; // FP16 staging mirrors, counted as floats for headroom
        // The packed-integer projections' activation, in Q8 blocks: four quants per int plus a
        // scale and a sum of quants per block of 32, sized for the widest activation any of them
        // reads. About 22 KiB at Qwen3.8-27B's shape, which is why it changes no prediction here
        // -- it is counted so the term stays a description of what is allocated.
        long widest = Math.max(dim, hidden);
        long quantizationBytes =
                widest / 4 * Integer.BYTES
                        + widest / 32 * Float.BYTES
                        + widest / 32 * Integer.BYTES;
        return floats * Float.BYTES + quantizationBytes;
    }

    /** Staging for a batched prefill chunk: embeddings and per-row activations. */
    private static long batchStagingBytes(Configuration config, int batchSize) {
        long rows = batchSize;
        long perRow = (long) config.dim() * 3 + config.hiddenDim() * 2 + config.kvDim() * 2;
        return rows * perRow * Float.BYTES;
    }
}

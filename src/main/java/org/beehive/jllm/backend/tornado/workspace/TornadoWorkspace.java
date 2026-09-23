package org.beehive.jllm.backend.tornado.workspace;

import org.beehive.jllm.inference.Logits;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * The device arrays one session executes against — activations, attention scratch, the key/value
 * views, the control and result carriers, and the batched-prefill workspace.
 *
 * <p><b>Not neutral, and not an interface.</b> Nothing outside {@code backend.tornado} names this
 * type. A neutral interface carrying {@code FloatArray} accessors would be the same violation
 * wearing a different name, which is why there is no property bag, no {@code Object}-typed
 * workspace, and no widened accessor.
 *
 * <p><b>Identity is fixed for its life</b> [C1]: a captured CUDA graph replays against the
 * addresses it captured, so nothing here is reallocated while a plan exists. A shared workspace
 * belongs to its binding domain and is borrowed under that domain's invocation lock.
 *
 * <p>Fields, not accessors, exactly as before: the per-token path reads them directly and must not
 * gain a call.
 */
public final class TornadoWorkspace {

    public FloatArray wrapAttSplit;
    public FloatArray
            wrapLogits; // FloatArray wrapper for the logits tensor, compatible with TornadoVM for
    // GPU execution.
    public FloatArray
            wrapXb; // FloatArray wrapper for xb (residual branch activation), optimized for
    // TornadoVM usage.
    public FloatArray
            wrapXb2; // FloatArray wrapper for xb2, another residual buffer to aid in computations
    // with TornadoVM.
    public FloatArray
            wrapHb; // FloatArray wrapper for hb (hidden dimension buffer for FFN), optimized for
    // TornadoVM.
    public FloatArray wrapHb2; // FloatArray wrapper for hb2, additional hidden buffer for FFN, for
    // compatibility with TornadoVM.
    public FloatArray
            wrapX; // FloatArray wrapper for the current activation tensor, optimized for TornadoVM.
    public FloatArray wrapQ; // FloatArray wrapper for the query tensor, optimized for TornadoVM.
    public FloatArray wrapK; // FloatArray wrapper for the key tensor, optimized for TornadoVM.
    public FloatArray wrapV; // FloatArray wrapper for the value tensor, optimized for TornadoVM.
    public FloatArray
            wrapAtt; // FloatArray wrapper for the attention scores, optimized for TornadoVM.
    public FloatArray
            wrapKeyCache; // FloatArray wrapper for the key cache, optimized for TornadoVM.
    public FloatArray
            wrapValueCache; // FloatArray wrapper for the value cache, optimized for TornadoVM.
    public HalfFloatArray
            wrapKeyCacheFP16; // Optional half-precision key cache (see StorageOptions); null unless
    // enabled.
    public HalfFloatArray
            wrapValueCacheFP16; // Optional half-precision value cache (see StorageOptions); null

    // unless enabled.
    // @formatter:off
    /**
     * The normalized activation, quantized per block of 32 for the packed-integer projections: four
     * signed bytes per int, one scale per block, and the sum of each block's quants.
     *
     * <p>Written once per branch and read by every Q4_0 projection that takes the normed activation
     * as its input, which is what pays for the quantization.
     */
    // @formatter:on
    public IntArray wrapXbQuants;

    public FloatArray wrapXbScales;

    public IntArray wrapXbSums;

    public IntArray positionHolder;
    public IntArray wrapBlockTable;
    public TornadoNativeArray embeddingX;
    public HalfFloatArray
            wrapXbFP16; // FloatArray wrapper for xb (residual branch activation), optimized for
    // TornadoVM usage.
    public FloatArray
            temp; // Temporary buffer for intermediate calculations, size adjusted for local
    // workgroup size.
    public FloatArray
            tempFFN; // Temporary buffer for feed-forward network calculations, size adjusted for
    // local workgroup size.
    public FloatArray
            tempLogits; // Temporary buffer for logits calculations, size adjusted for local
    // workgroup size.
    public HalfFloatArray wrapXFP16;
    public HalfFloatArray embeddingXBatch; // B × dim  (FP16 input)
    public FloatArray wrapXBatch; // B × dim  (live activations / Q8_0 dequant)
    public HalfFloatArray wrapXbFP16Batch; // B × dim  (RMSNorm output, FP16)
    public FloatArray wrapQBatch; // B × qDim (Q projection)
    public FloatArray wrapKBatch; // B × kvDim
    public FloatArray wrapVBatch; // B × kvDim
    public FloatArray wrapXbBatch; // B × qDim  (attention output)
    public FloatArray wrapHbBatch; // B × hiddenDim
    public FloatArray attnScaleBatch; // B        (per-token RMS scale, attn)
    public FloatArray ffnScaleBatch; // B        (per-token RMS scale, FFN)
    public IntArray batchStartPosHolder; // 1      (start position of chunk)
    public HalfFloatArray normedXFFNFP16;
    public FloatArray ffnGateResult;
    public FloatArray ffnUpResult;
    public HalfFloatArray xbFP16Batch;
    public HalfFloatArray attnOutFP16;
    public FloatArray woOut;
    public HalfFloatArray wrapHbFP16Batch;
    public FloatArray w2Out;
    public FloatArray qkvResultBatch; // B × (dim + 2*kvDim), packed [q|k|v] rows
    public FloatArray gateUpResultBatch; // B × 2*hiddenDim, packed [gate|up] rows

    // ── Family-specific device arrays ──────────────────────────────────────────────
    //
    // Qwen3, Phi3, Gemma4 and Qwen2-MoE each declared these on their own State subtype, which is
    // why those four were Rule 1 entries after the base class stopped being one. They are device
    // arrays like every other field here; that one family uses a buffer and another does not is not
    // a reason for it to live somewhere else.
    public FloatArray wrapPerLayerInputs;
    public FloatArray wrapPerLayerProjScratch;
    public FloatArray wrapPerLayerGate;
    public FloatArray wrapPerLayerOut;
    public FloatArray wrapPerLayerTokenEmbedRow;
    public FloatArray tempPostAttn;
    public FloatArray tempPostFfn;
    public FloatArray tempPostPle;
    public FloatArray wrapQkv; // TornadoVM wrapper for QKV buffer
    public FloatArray wrapHbG; // TornadoVM wrapper for gate states
    public FloatArray wrapHbU; // TornadoVM wrapper for up states
    public FloatArray wrapRouterLogits;
    public IntArray wrapSelectedExperts;

    // qwen35. Its recurrent layers hold state that is neither a key/value cache nor scratch: the
    // convolution's rolling window and the delta-net matrices persist across tokens and are
    // updated in place. Both live in one array per kind, addressed by a per-layer offset, because
    // a device buffer per layer would be 48 of them to transfer and keep resident.
    public FloatArray wrapConvState;
    public FloatArray wrapDeltaState;
    public FloatArray wrapSsmQkv;
    public FloatArray wrapSsmConvOut;
    public FloatArray wrapSsmZ;
    public FloatArray wrapSsmAlpha;
    public FloatArray wrapSsmBeta;
    public FloatArray wrapSsmQ;
    public FloatArray wrapSsmK;
    public FloatArray wrapSsmV;
    public FloatArray wrapSsmOut;

    // gemma4 batched prefill. The chunk-wide twins of the per-layer-embedding buffers above, plus
    // the two this family's batched layer needs and no single-token graph does: one FP16 carrier
    // per GEMM input, and a score scratch the attention kernel writes because a window of up to the
    // whole context does not fit in a workgroup's shared memory.
    public FloatArray wrapPerLayerInputsBatch;
    public FloatArray wrapPerLayerProjScratchBatch;
    public FloatArray wrapPerLayerGateBatch;
    public HalfFloatArray wrapPerLayerGateFP16Batch;
    public FloatArray wrapPerLayerOutBatch;
    public FloatArray wrapPerLayerTokenEmbedRowBatch;
    public HalfFloatArray wrapXFP16Batch;
    public FloatArray branchScaleBatch;
    public FloatArray attnScoresBatch;
    public FloatArray splitKPartialBatch;
    public HalfFloatArray weightsF16Scratch;

    /** The query half of an attention layer's fused query/gate projection, de-interleaved. */
    public FloatArray wrapAttnQ;

    /** Its gate half, applied through a logistic to the attention result. */
    public FloatArray wrapAttnGate;

    // qwen35 batched prefill. The same scratch a chunk wide, row-major: a row is one prompt
    // token, and every kernel that walks a chunk strides by the buffer's own width. The
    // single-token buffers are not reused with a stride parameter because a projection's input
    // width and its output width differ, and a shared buffer would make one of the two a lie.
    public FloatArray wrapNormedBatch;

    /** The normed chunk as FP16, for the tensor-core projections. */
    public HalfFloatArray wrapNormedFP16Batch;

    /** Gate and up, before SwiGLU combines them, when the tensor-core path computes them. */
    public FloatArray wrapGateBatch;

    public FloatArray wrapUpBatch;

    /** The SwiGLU output as FP16, and the tensor-core ffn_down result before it is added back. */
    public HalfFloatArray wrapHbFP16BatchMMA;

    public FloatArray wrapFFNDownBatch;

    /** The delta-net readout as FP16, for the tensor-core ssm_out projection. */
    public HalfFloatArray wrapSsmOutFP16Batch;

    public FloatArray wrapQGateBatch;
    public FloatArray wrapAttnQBatch;
    public FloatArray wrapAttnGateBatch;

    /**
     * Batched prefill attention's query-key dot products, one span per (chunk row, query head) of
     * the context capacity, written and read within a single attention launch. Null unless the
     * batched FP16 key/value path was built, which is the only reader.
     */
    public FloatArray wrapAttnScoresBatch;

    /**
     * FP16 staging for the tensor-core batched attention: per (16-query tile, head) workgroup, its
     * queries and its probability tile, converted through global memory because the kernel language
     * has no in-register float-to-half conversion. Allocated alongside the score scratch when the
     * chunk is whole 16-row tiles.
     */
    public HalfFloatArray wrapAttnStageFP16;

    /**
     * One FP16 matrix of scratch for the batched prefill's dequantize-then-GEMM projections — the
     * wide Q4_0 ones, the Q4_1 ffn_down and the Q5_K ssm_out — sized to the largest such matrix and
     * reused by every projection of every layer graph in turn: each projection's dequantization
     * writes it and its GEMM reads it before the next projection's dequantization runs, whatever
     * the two matrices' sizes. Null unless the width takes that path.
     */
    public HalfFloatArray wrapDequantScratchFP16;

    /**
     * The int8 pair's scratch: the chunk's activations quantized to int8 with a scale per 32 (one
     * buffer, requantized before each group of consumers in graph order), and one Q4_0 matrix
     * decoded to int8 in the B-operand word layout with its FP32 block scales.
     */
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray wrapQ8ActBatch;

    public FloatArray wrapQ8ActScales;
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray wrapInt8WeightScratch;
    public FloatArray wrapInt8WeightScales;

    public FloatArray wrapSsmQkvBatch;
    public FloatArray wrapSsmConvOutBatch;
    public FloatArray wrapSsmZBatch;
    public FloatArray wrapSsmAlphaBatch;
    public FloatArray wrapSsmBetaBatch;
    public FloatArray wrapSsmQBatch;
    public FloatArray wrapSsmKBatch;
    public FloatArray wrapSsmVBatch;
    public FloatArray wrapSsmOutBatch;
    public FloatArray wrapRoutingWeights;
    public FloatArray wrapExpertGate;
    public FloatArray wrapSharedGate;
    public FloatArray wrapSharedOutput;
    public FloatArray wrapRouterLogitsBatch;
    public IntArray activeBatchSizeHolder;
    public IntArray wrapSelectedExpertsBatch;
    public FloatArray wrapRoutingWeightsBatch;
    public IntArray wrapGroupedAssignmentIds;
    public IntArray wrapGroupedPositionByAssignment;
    public FloatArray wrapGroupedExpertHidden;
    public FloatArray wrapGroupedExpertDown;
    public FloatArray wrapSharedHiddenBatch;
    public FloatArray wrapSharedWeightBatch;
    public FloatArray tempQcur;
    public FloatArray tempKcur;

    /** One slot: the token an on-device sampler wrote, read by the loop. */
    public IntArray sampledToken = new IntArray(1);

    /**
     * The token an on-device sampler wrote, as an {@code int}.
     *
     * <p>The generation loop needs the value, not the carrier. Reading {@code
     * workspace.sampledToken.get(0)} from the loop is what made {@code TokenGenerationLoop} name
     * {@code IntArray}; this returns the same slot without naming it.
     */
    public int deviceSampledToken() {
        return sampledToken.get(0);
    }

    private FloatArray logitsViewTarget;
    private Logits logitsView;

    /**
     * {@code array} as the neutral {@link Logits} view.
     *
     * <p>A one-element identity cache, and that is the whole of it. The view is always over the
     * array the plan actually returned — a legacy plan hands back this workspace's {@code
     * wrapLogits}, a lowered one hands back the session's own copy — so the identity is checked
     * rather than assumed. Steady state allocates nothing: a decode loop asks for the same array
     * every token, which is why this is not built fresh per call.
     */
    public Logits logitsView(FloatArray array) {
        if (array != logitsViewTarget) {
            logitsViewTarget = array;
            logitsView = TornadoLogits.of(array);
        }
        return logitsView;
    }
}

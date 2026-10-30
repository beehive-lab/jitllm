package org.beehive.jitllm.backend.tornado.workspace;

import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacker;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.inference.Logits;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
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

    /**
     * The prepacking of this session's projections on the device, collected from the layer graphs
     * as they are built and appended to the plan as graphs of its own.
     */
    public final WeightPrepacker prepacker = new WeightPrepacker();

    /** Which weights this session's plan prepacks, set when the plan is built. */
    public WeightPrepacking prepacking = WeightPrepacking.NONE;

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

    /**
     * B rows of a Q8_0 embedding as the file stores them, for a batch activation that decodes them
     * on the device. Null unless such an activation allocated it: the host then bulk-copies raw
     * rows here instead of decoding them element by element in Java.
     */
    public ByteArray embeddingQ8Batch;

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
    // Gemma 4 tensor-core prefill attention: the unnormalized FP32 output before it is narrowed,
    // and each workgroup's FP16 probability tile.
    public FloatArray attnOutF32Batch;
    public HalfFloatArray attnProbStageBatch;

    /** The query half of an attention layer's fused query/gate projection, de-interleaved. */
    public FloatArray wrapAttnQ;

    /** A {@code deepseek2} token's compressed query, before its norm and {@code attn_q_b}. */
    public FloatArray wrapQueryLatent;

    /** {@code attn_kv_a_mqa}'s output: the latent, then the rotated key. */
    public FloatArray wrapCompressedKv;

    /** Every head's query absorbed into the latent space, then its rotated part. */
    public FloatArray wrapAbsorbedQuery;

    /** Every head's attended latent, before {@code attn_v_b}. */
    public FloatArray wrapLatentOut;

    // A deepseek2 chunk's buffers (see DeepSeek2BatchKernels): the projections' outputs, the
    // latent attention's (token, head) rows and their FP16 copies, and the dense feed-forward's.
    public FloatArray queryLatentBatch;
    public FloatArray compressedKvBatch;
    public FloatArray queryBatch;
    public HalfFloatArray queryF16Batch;
    public FloatArray absorbedBatch;
    public HalfFloatArray absorbedF16Batch;
    public HalfFloatArray keysF16;
    public HalfFloatArray latentTF16;
    public FloatArray scoresBatch;
    public HalfFloatArray probsF16Batch;
    public FloatArray latentBatch;
    public HalfFloatArray latentF16Batch;
    public FloatArray attnOutBatch;
    public FloatArray gateBatch;
    public FloatArray hiddenBatch;

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
     * One projection's weights decoded to FP32, for a batched projection run by cuBLAS on a
     * representation it cannot read (Q8_0). Allocated by the batch prefill that needs it.
     */
    public FloatArray wrapDequantScratchF32;

    /**
     * The partial sums of a batched Q8_0 projection whose int8 GEMM is split along K, one output
     * matrix per split, before they are added together. Allocated by the batch prefill that needs
     * it.
     */
    public FloatArray wrapQ8SplitPartial;

    /**
     * The int8 pair's scratch: the chunk's activations quantized to int8 with a scale per 32 (one
     * buffer, requantized before each group of consumers in graph order), and one Q4_0 matrix
     * decoded to int8 in the B-operand word layout with its FP32 block scales.
     */
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray wrapQ8ActBatch;

    public FloatArray wrapQ8ActScales;

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

    /**
     * A {@code qwen35moe} token's expert activations: every selected expert's hidden rows, then the
     * shared expert's.
     */
    public FloatArray wrapMoeHidden;

    /** {@link #wrapMoeHidden} quantized per 32-block, for the packed-integer down projection. */
    public IntArray wrapMoeHiddenQuants;

    public FloatArray wrapMoeHiddenQScales;
    public IntArray wrapMoeHiddenQSums;

    // A qwen35moe chunk's routing and expert activations (see Qwen35MoeBatchKernels).
    public FloatArray wrapMoeLogitsBatch;
    public IntArray wrapMoeIdsBatch;
    public FloatArray wrapMoeWeightsBatch;
    public FloatArray wrapMoeSharedGateBatch;
    public IntArray wrapMoeSortedToken;
    public IntArray wrapMoePosition;
    public IntArray wrapMoeTiles;
    public FloatArray wrapMoeHiddenBatch;
    public ByteArray wrapMoeHiddenQ8;
    public FloatArray wrapMoeHiddenScales;
    public FloatArray wrapMoeOutBatch;
    public FloatArray wrapMoeSharedGateUp;
    public FloatArray wrapMoeSharedHidden;
    public ByteArray wrapMoeSharedQ8;
    public FloatArray wrapMoeSharedScales;
    public FloatArray wrapMoeSharedOut;

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

package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels for the Gated Delta Net mixer — the recurrent three quarters of a {@code qwen35}
 * stack.
 *
 * <h2>Every kernel here is one lane, written as a function</h2>
 *
 * <p>Each kernel body is a static method taking an explicit lane index, and the kernel itself is a
 * two-line wrapper that passes {@code context.globalIdx}. That is not decoration: a method taking
 * {@link KernelContext} cannot be called on the host, so a kernel written directly against it can
 * only be checked by running it, on a device, inside a model. Lifting the arithmetic out means
 * {@code Qwen35DeltaNetKernelParityTest} can run every lane on the host and compare it to {@code
 * CpuOperations} element by element — which is the check that catches an indexing mistake, and the
 * one that is otherwise impossible to write.
 *
 * <p>It costs nothing at run time. TornadoVM inlines the helper.
 *
 * <h2>Why no reductions and no barriers</h2>
 *
 * <p>The delta rule looks like it needs one — it is a matrix-vector product per head — and it does
 * not. Give a lane one <b>value column</b> {@code j} of a head's state and every quantity it needs
 * is its own: the decayed column, the prediction {@code Sᵀk} for that column, the correction, the
 * rank-one update to that column, and the readout {@code Sᵀq} for that column. Nothing is shared
 * between lanes, so there is no barrier, no local memory, and no cross-lane reduction.
 *
 * <p>The state layout makes that coalesced rather than merely correct. State is {@code (h * S + i)
 * * S + j} — key row {@code i}, value column {@code j} — so at each step of the inner loop the
 * {@code S} lanes of a head read {@code S} consecutive floats. Per-lane the access is strided; what
 * a GPU is paid for is the access being contiguous <i>across</i> lanes, and it is.
 *
 * <p>This is the same layout the host path uses, deliberately. Two layouts would mean the parity
 * test compares a transpose against a transpose and proves nothing about either.
 */
public final class Qwen35DeltaNetKernels {

    private Qwen35DeltaNetKernels() {}

    // ---- causal convolution --------------------------------------------------

    /**
     * One channel of a depthwise causal convolution, advancing that channel's window.
     *
     * <p>Depthwise means the channels never mix, so a channel is a lane and there is nothing to
     * reduce. The window is state and this owns advancing it; each lane touches only its own slice
     * of it, so the in-place shift races with nothing.
     *
     * <p>Both the kernel and the window are channel-major — a channel's taps are contiguous, oldest
     * first — matching how GGUF stores {@code ssm_conv1d} and how the host path reads it.
     *
     * <p>{@code windowOffset} is where this layer's window starts. Every recurrent layer's window
     * lives in one array — a device buffer per layer would be 48 of them to transfer and persist —
     * so the layer is an offset rather than a separate allocation. It is a parameter and not
     * derived from anything, because deriving it would mean the kernel knowing how many layers
     * there are.
     *
     * @param channel the lane: which of {@code channels} this call computes
     */
    static void causalConv1dLane(
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            int kernel,
            int windowOffset,
            int channel) {
        int history = kernel - 1;
        int wBase = channel * kernel;
        int hBase = windowOffset + channel * history;
        float x = input.get(channel);

        float sum = 0.0f;
        for (int t = 0; t < history; t++) {
            sum += weight.get(wBase + t) * window.get(hBase + t);
        }
        sum += weight.get(wBase + history) * x;
        out.set(channel, sum);

        for (int t = 0; t + 1 < history; t++) {
            window.set(hBase + t, window.get(hBase + t + 1));
        }
        if (history > 0) {
            window.set(hBase + history - 1, x);
        }
    }

    /** One lane per channel. */
    public static void causalConv1d(
            KernelContext context,
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            int channels,
            int kernel,
            int windowOffset) {
        int channel = context.globalIdx;
        if (channel >= channels) {
            return;
        }
        causalConv1dLane(input, weight, window, out, kernel, windowOffset, channel);
    }

    /**
     * SiLU over the convolved result, in place. One lane per channel.
     *
     * <p>A separate kernel rather than folded into the convolution: the host path applies it to the
     * whole vector after the convolution, and keeping the boundary in the same place keeps the two
     * comparable term by term.
     */
    public static void siluInPlace(KernelContext context, FloatArray values, int count) {
        int i = context.globalIdx;
        if (i >= count) {
            return;
        }
        float v = values.get(i);
        values.set(i, v / (1.0f + TornadoMath.exp(-v)));
    }

    // ---- L2 normalization ----------------------------------------------------

    /**
     * One head scaled to unit length.
     *
     * <p>A lane per head rather than a workgroup per head with a reduction: a head is 128 wide here
     * and there are 16 of them, so the reduction would cost more in barriers than the serial loop
     * costs in arithmetic. Epsilon floors the divisor rather than being added under the root, which
     * is what {@code ggml_l2_norm} does and what the host path was written against.
     *
     * @param head the lane
     */
    static void l2NormLane(FloatArray values, int headDim, float eps, int head) {
        int base = head * headDim;
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.max(TornadoMath.sqrt(ss), eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, values.get(base + i) * inv);
        }
    }

    // @formatter:off
    /**
     * The L2 norm with a workgroup per head and a lane per element.
     *
     * <p>{@link #l2NormPerHead} gives one <b>thread</b> a whole head, so at this model's geometry —
     * 16 key heads of 128 — the whole kernel is sixteen threads, half a warp, and each walks its
     * head's 128 elements twice in series. This owns a head per workgroup instead: a lane loads its
     * element once and keeps it, the sum of squares is a shared tree, and lane zero turns it into
     * the reciprocal once for the others to read back.
     *
     * <p><b>The equation is the lane version's, including where the epsilon sits.</b> It is {@code
     * 1 / max(sqrt(ss), eps)} — the epsilon clamps the norm itself and is not added under the root,
     * and there is no division by the head width. That is not the gated norm's convention and the
     * two must not be made to look alike. A head of exact zeros therefore takes {@code inv = 1/eps}
     * and stays zero, as it does today.
     *
     * <p>What changes is the <b>order of the sum of squares</b>, a tree instead of a left fold, so
     * this is not bit-identical to {@link #l2NormPerHead} and is not claimed to be.
     *
     * <p>The reduction is shared memory and a barrier — no subgroup shuffle — which is the
     * primitive {@code rowDotShared} already uses on every backend. This kernel is therefore
     * dispatched exactly where the lane version was, and no backend is added. It does require
     * {@code headDim} to be a power of two and to equal the local work size; the caller keeps
     * {@link #l2NormPerHead} for any geometry that is not.
     */
    // @formatter:on
    public static void l2NormPerHeadWide(
            KernelContext context, FloatArray values, int headDim, float eps) {
        int head = context.groupIdx;
        int lane = context.localIdx;
        int index = head * headDim + lane;

        float[] shared = context.allocateFloatLocalArray(headDim);

        // Read once. The lane that writes this element is the lane that read it, so the in-place
        // store below cannot race a read of the same address.
        float v = values.get(index);
        shared[lane] = v * v;
        context.localBarrier();
        for (int stride = headDim >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = shared[lane] + shared[lane + stride];
            }
            context.localBarrier();
        }
        if (lane == 0) {
            shared[0] = 1.0f / TornadoMath.max(TornadoMath.sqrt(shared[0]), eps);
        }
        context.localBarrier();

        values.set(index, v * shared[0]);
    }

    /** One lane per head. */
    public static void l2NormPerHead(
            KernelContext context, FloatArray values, int heads, int headDim, float eps) {
        int head = context.globalIdx;
        if (head >= heads) {
            return;
        }
        l2NormLane(values, headDim, eps, head);
    }

    // ---- the decay and write strengths ---------------------------------------

    /**
     * One value head's decay and write strength, from their raw projections.
     *
     * <p>{@code beta = sigmoid(betaRaw)} and {@code decay = exp(a * softplus(alphaRaw + dtBias))},
     * both in place. Folded into one lane because they are consumed together and each is a handful
     * of operations on one element — two kernels over 48 elements would be two launches to save
     * nothing.
     *
     * <p>{@code a} is {@code -exp(A_log)} as the file stores it, so the product is a log decay and
     * its exponential lands in {@code (0, 1)}: the state is forgotten, never amplified.
     */
    static void decayAndBetaLane(
            FloatArray alpha, FloatArray beta, FloatArray dtBias, FloatArray a, int head) {
        float raw = beta.get(head);
        beta.set(head, 1.0f / (1.0f + TornadoMath.exp(-raw)));

        float biased = alpha.get(head) + dtBias.get(head);
        // softplus, guarded the way the host path guards it: for a large argument log1p(exp(x))
        // is x to within float precision, and exp(x) alone would overflow.
        float softplus = biased > 20.0f ? biased : TornadoMath.log(1.0f + TornadoMath.exp(biased));
        alpha.set(head, TornadoMath.exp(a.get(head) * softplus));
    }

    /** One lane per value head. */
    public static void decayAndBeta(
            KernelContext context,
            FloatArray alpha,
            FloatArray beta,
            FloatArray dtBias,
            FloatArray a,
            int valueHeads) {
        int head = context.globalIdx;
        if (head >= valueHeads) {
            return;
        }
        decayAndBetaLane(alpha, beta, dtBias, a, head);
    }

    // ---- the delta rule ------------------------------------------------------

    /**
     * One value column of one head: decay, correct, accumulate, read back.
     *
     * <pre>
     *   s[i]  = S[i][j] * decay          // forget
     *   sk    = Σ s[i]·k[i]              // what the state already predicts for this column
     *   d     = (v[j] - sk) * beta       // the part it does not
     *   S[i][j] = s[i] + k[i]·d          // write the correction
     *   out[j]  = Σ S[i][j]·q[i]         // read it back
     * </pre>
     *
     * <p>Two passes over the column rather than one, because the readout must see the updated state
     * and the update needs the whole prediction first. Both passes are over the same 128 values, so
     * the second finds them in cache.
     *
     * <p><b>A value head reads key head {@code h % keyHeads}.</b> Modulo, not division: the
     * reference repeats the key heads by tiling. Dividing pairs every value head with the wrong key
     * and produces fluent, slowly degrading output — the defect this port already made once on the
     * host, which is why it is stated here rather than left to the caller.
     *
     * <p>{@code stateOffset} is where this layer's state starts, for the reason the convolution's
     * window offset exists: every recurrent layer's state lives in one array, and 48 separate
     * device buffers would be 48 transfers to arrange and keep resident.
     *
     * @param lane {@code head * stateDim + column}
     */
    static void deltaRuleLane(
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int keyHeads,
            int stateDim,
            int stateOffset,
            int lane) {
        int head = lane / stateDim;
        int column = lane - head * stateDim;

        int stateBase = stateOffset + head * stateDim * stateDim;
        int kvBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;

        float g = decay.get(head);
        float b = beta.get(head);

        float prediction = 0.0f;
        for (int i = 0; i < stateDim; i++) {
            int index = stateBase + i * stateDim + column;
            float decayed = state.get(index) * g;
            state.set(index, decayed);
            prediction += decayed * k.get(kvBase + i);
        }

        float correction = (v.get(valueBase + column) - prediction) * b;

        float readout = 0.0f;
        for (int i = 0; i < stateDim; i++) {
            int index = stateBase + i * stateDim + column;
            float updated = state.get(index) + k.get(kvBase + i) * correction;
            state.set(index, updated);
            readout += updated * q.get(kvBase + i);
        }
        out.set(valueBase + column, readout);
    }

    // @formatter:off
    /**
     * The delta rule with each column's reduction split across two lanes.
     *
     * <p>{@link #deltaRule} gives one lane a whole column, walking the column's {@code stateDim}
     * rows twice in two dependent accumulations. This widens the workgroup to {@code 2 * stateDim}
     * and gives a column two lanes, each taking half the rows, halving each chain and doubling the
     * warps per workgroup. The grid is still one workgroup per value head, so it does not reach the
     * multiprocessors that get no workgroup at all; that would mean a reduction across workgroups,
     * which is not done here.
     *
     * <p><b>Ownership.</b> Lane {@code (half, column)} owns rows {@code [half*rows, (half+1)*rows)}
     * of that column, in both sweeps, so every state element has exactly one writer and is written
     * once per sweep. {@code out} is written by the {@code half == 0} lane alone. Each lane writes
     * only its own slot of the shared array.
     *
     * <p><b>Synchronisation.</b> Three barriers. The first publishes both partial predictions
     * before any lane forms the correction. The second separates every lane's read of those
     * partials from the reuse of the shared array, so a fast lane cannot overwrite a value a slow
     * lane has not read. The third publishes both partial readouts before the output is written. No
     * barrier crosses a workgroup and no state is added.
     *
     * <p><b>Arithmetic.</b> Per element it is the accepted kernel's, expression for expression:
     * {@code state * g} is stored before it is used, and {@code state + k * correction} is formed
     * the same way, so the intermediate rounding the accepted kernel produces is preserved and the
     * fused multiply-add the compiler may form is the same one. What changes is the <b>association
     * of the two reductions</b>: a sum of two half-length folds rather than one full-length fold.
     * Both lanes of a column combine the two partials in the same order, so they form bit-identical
     * corrections and cannot diverge from each other.
     */
    // @formatter:on
    public static void deltaRuleSplit(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int keyHeads,
            int stateDim,
            int stateOffset) {
        int head = context.groupIdx;
        int tid = context.localIdx;
        int column = tid % stateDim;
        int half = tid / stateDim;

        float[] shared = context.allocateFloatLocalArray(2 * stateDim);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int kvBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;

        float g = decay.get(head);
        float b = beta.get(head);

        int rows = stateDim / 2;
        int rowStart = half * rows;
        int rowEnd = rowStart + rows;

        float partial = 0.0f;
        for (int i = rowStart; i < rowEnd; i++) {
            int index = stateBase + i * stateDim + column;
            float decayed = state.get(index) * g;
            state.set(index, decayed);
            partial += decayed * k.get(kvBase + i);
        }
        shared[tid] = partial;
        context.localBarrier();

        // Both lanes of a column read the same two partials in the same order.
        float prediction = shared[column] + shared[stateDim + column];
        float correction = (v.get(valueBase + column) - prediction) * b;
        // Every lane has read the partials; the shared array may now be reused.
        context.localBarrier();

        float partialReadout = 0.0f;
        for (int i = rowStart; i < rowEnd; i++) {
            int index = stateBase + i * stateDim + column;
            float updated = state.get(index) + k.get(kvBase + i) * correction;
            state.set(index, updated);
            partialReadout += updated * q.get(kvBase + i);
        }
        shared[tid] = partialReadout;
        context.localBarrier();

        if (half == 0) {
            out.set(valueBase + column, shared[column] + shared[stateDim + column]);
        }
    }

    /** Row parts a column's reductions are split across in {@link #deltaRuleSplit8}. */
    public static final int DELTA_RULE_PARTS = 8;

    // @formatter:off
    /**
     * {@link #deltaRuleSplit} with eight lanes a column instead of two: lane {@code (part, column)}
     * owns rows {@code [part * 16, part * 16 + 16)} of its column, so each dependent chain is a
     * quarter of the two-lane kernel's and the workgroup holds four times the warps. Lanes are laid
     * out column-fastest ({@code tid = part * stateDim + column}), so a warp's loads of one row are
     * 32 consecutive floats.
     *
     * <p>Arithmetic per element is the two-lane kernel's, expression for expression; the two
     * reductions are sums of eight sixteen-row folds, combined in part order {@code ((p0 + p1) +
     * p2) + ...} by every lane of a column alike, so the lanes of a column form bit-identical
     * corrections. Not bit-identical to the two-lane kernel (a different association).
     *
     * <p>Worker: one workgroup of {@code 8 * stateDim} lanes per value head.
     */
    // @formatter:on
    public static void deltaRuleSplit8(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int keyHeads,
            int stateDim,
            int stateOffset) {
        int head = context.groupIdx;
        int tid = context.localIdx;
        int column = tid % stateDim;
        int part = tid / stateDim;

        float[] shared = context.allocateFloatLocalArray(DELTA_RULE_PARTS * stateDim);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int kvBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;

        float g = decay.get(head);
        float b = beta.get(head);

        int rows = stateDim / DELTA_RULE_PARTS;
        int rowStart = part * rows;
        int rowEnd = rowStart + rows;

        float partial = 0.0f;
        for (int i = rowStart; i < rowEnd; i++) {
            int index = stateBase + i * stateDim + column;
            float decayed = state.get(index) * g;
            state.set(index, decayed);
            partial += decayed * k.get(kvBase + i);
        }
        shared[tid] = partial;
        context.localBarrier();

        float prediction = 0.0f;
        for (int p = 0; p < DELTA_RULE_PARTS; p++) {
            prediction += shared[p * stateDim + column];
        }
        float correction = (v.get(valueBase + column) - prediction) * b;
        context.localBarrier();

        float partialReadout = 0.0f;
        for (int i = rowStart; i < rowEnd; i++) {
            int index = stateBase + i * stateDim + column;
            float updated = state.get(index) + k.get(kvBase + i) * correction;
            state.set(index, updated);
            partialReadout += updated * q.get(kvBase + i);
        }
        shared[tid] = partialReadout;
        context.localBarrier();

        if (part == 0) {
            float readout = 0.0f;
            for (int p = 0; p < DELTA_RULE_PARTS; p++) {
                readout += shared[p * stateDim + column];
            }
            out.set(valueBase + column, readout);
        }
    }

    /** One lane per (value head, value column) — {@code valueHeads * stateDim} of them. */
    public static void deltaRule(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * stateDim) {
            return;
        }
        deltaRuleLane(q, k, v, decay, beta, state, out, keyHeads, stateDim, stateOffset, lane);
    }

    // ---- the gated norm ------------------------------------------------------

    /**
     * One head of {@code rms_norm(values, weight) * silu(gate)}, in place on {@code values}.
     *
     * <p>A lane per head, for the reason {@link #l2NormLane} is: the head is narrow and there are
     * few of them. The learned scale is one head wide and shared by every head.
     */
    static void gatedNormLane(
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int headDim,
            float eps,
            int head) {
        int base = head * headDim;

        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);

        for (int i = 0; i < headDim; i++) {
            float z = gate.get(base + i);
            float silu = z / (1.0f + TornadoMath.exp(-z));
            values.set(base + i, weight.get(i) * (inv * values.get(base + i)) * silu);
        }
    }

    // @formatter:off
    /**
     * The gated norm with a workgroup per head and a lane per element.
     *
     * <p>{@link #gatedNormPerHead} gives one <b>thread</b> a whole head, so at this model's
     * geometry — 48 value heads of 128 — the entire kernel is 48 threads in one workgroup, and it
     * walks each head's 128 elements twice in series. Two things the generated code showed, rather
     * than the source:
     *
     * <ul>
     *   <li>the apply loop <b>re-loads</b> {@code values}, which the summing loop has already read,
     *       so the activation is read twice;
     *   <li>{@code inv} is written above the loop in the source and the backend <b>sinks it back
     *       in</b>, so the divide, the add and the {@code rsqrt} run once per element rather than
     *       once per head — 6144 reciprocal square roots a call where 48 are needed.
     * </ul>
     *
     * <p>Both disappear with the mapping rather than with a rewrite of the arithmetic. A lane owns
     * one element: it loads its value <b>once</b> and keeps it, the sum of squares is a shared tree
     * over the workgroup, lane zero turns that into {@code inv} — once per head — and every lane
     * reads it back through the same shared cell.
     *
     * <p>The equation, the gate's position in it, the epsilon and the in-place layout are
     * unchanged: {@code weight[i] * (inv * v) * silu(z)}, associated exactly as before. What moves
     * is the <b>order of the sum of squares</b>, a tree instead of a left fold, so this is not
     * bit-identical to {@link #gatedNormPerHead} and is not claimed to be.
     *
     * <p>The tree is shared memory and a barrier, not a subgroup shuffle, so nothing here depends
     * on the backend. It does require {@code headDim} to be a power of two and to equal the local
     * work size; the caller keeps {@link #gatedNormPerHead} for any geometry that is not.
     */
    // @formatter:on
    public static void gatedNormPerHeadWide(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int headDim,
            float eps) {
        int head = context.groupIdx;
        int lane = context.localIdx;
        int index = head * headDim + lane;

        float[] shared = context.allocateFloatLocalArray(headDim);

        // Read once. The lane that writes this element is the lane that read it, so the in-place
        // store below cannot race a read of the same address.
        float v = values.get(index);
        shared[lane] = v * v;
        context.localBarrier();
        for (int stride = headDim >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = shared[lane] + shared[lane + stride];
            }
            context.localBarrier();
        }
        if (lane == 0) {
            shared[0] = 1.0f / TornadoMath.sqrt(shared[0] / headDim + eps);
        }
        context.localBarrier();
        float inv = shared[0];

        float z = gate.get(index);
        float silu = z / (1.0f + TornadoMath.exp(-z));
        values.set(index, weight.get(lane) * (inv * v) * silu);
    }

    /** One lane per head. */
    public static void gatedNormPerHead(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps) {
        int head = context.globalIdx;
        if (head >= heads) {
            return;
        }
        gatedNormLane(values, gate, weight, headDim, eps, head);
    }
}

package org.beehive.jllm.inference.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.jllm.auxiliary.Parallel;
import org.beehive.jllm.backend.cpu.InferenceCore;
import org.beehive.jllm.inference.state.Gemma4State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.jllm.model.gemma4.Gemma4Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.runtime.tensor.LongIndexedTensor;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * The fixture is built to exercise everything this family forced into the vocabulary, because a
 * simpler one would let most of it pass untested:
 *
 * <ul>
 *   <li><b>both sliding-window and full-attention layers</b> — the pattern alternates;
 *   <li><b>both own-KV and KV-reuse layers</b> — {@code sharedKvLayers = 2} of 4, so the last two
 *       borrow;
 *   <li><b>per-layer head geometry</b> — sliding-window layers use a different head width from full
 *       ones, and neither equals {@code dim / heads};
 *   <li><b>weighted and unweighted RMS normalization</b> — keys use the first, values the second;
 *   <li><b>post-norm residual ordering</b> — the branch output is normalized before it is added;
 *   <li><b>the per-layer embedding contribution</b>, read from a long-indexed table;
 *   <li><b>the logit soft-cap</b>, non-zero here.
 * </ul>
 */
public class Gemma4CpuOperationEquivalenceTest {

    private static final int DIM = 32;
    private static final int LAYERS = 4;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM_SWA = 4; // != DIM / HEADS, which is 8
    private static final int HEAD_DIM_FULL = 6; // and different from the sliding-window one
    private static final int FF = 24;
    private static final int VOCAB = 20;
    private static final int CONTEXT = 16;
    private static final int WINDOW = 3;
    private static final int SHARED_KV_LAYERS = 2;
    private static final int PER_LAYER_DIM = 8;
    private static final float EPS = 1e-5f;
    private static final float SOFTCAP = 5.0f;

    /** Alternating, so both attention kinds and both head widths run. */
    private static final boolean[] SWA_PATTERN = {true, false, true, false};

    @Test
    public void theRefactoredForwardPassIsBitIdentical() {
        Gemma4Configuration config = config();
        Gemma4StandardWeights weights = syntheticWeights(new Random(20260831L));

        Gemma4State refactored = new Gemma4State(config, -1);
        Gemma4State reference = new Gemma4State(config, -1);

        int[] tokens = {7, 3, 11, 0, 5, 9};
        for (int position = 0; position < tokens.length; position++) {
            FloatTensor a =
                    InferenceCore.forwardJavaGemma4(
                            config, weights, refactored, tokens[position], position);
            FloatTensor b =
                    referenceForwardJavaGemma4(
                            config, weights, reference, tokens[position], position);
            assertBitIdentical("logits at position " + position, b, a, VOCAB);
        }
    }

    @Test
    public void theKeyValueCachesAgree() {
        Gemma4Configuration config = config();
        Gemma4StandardWeights weights = syntheticWeights(new Random(404L));

        Gemma4State refactored = new Gemma4State(config, -1);
        Gemma4State reference = new Gemma4State(config, -1);

        int[] tokens = {2, 13, 6, 8, 1};
        for (int position = 0; position < tokens.length; position++) {
            InferenceCore.forwardJavaGemma4(
                    config, weights, refactored, tokens[position], position);
            referenceForwardJavaGemma4(config, weights, reference, tokens[position], position);
        }

        for (int l = 0; l < LAYERS; l++) {
            int kvDim = KV_HEADS * config.headDim(l);
            assertBitIdentical(
                    "key cache, layer " + l,
                    reference.keyCache[l],
                    refactored.keyCache[l],
                    tokens.length * kvDim);
            assertBitIdentical(
                    "value cache, layer " + l,
                    reference.valueCache[l],
                    refactored.valueCache[l],
                    tokens.length * kvDim);
        }
    }

    /**
     * The fixture really does contain both layer kinds, both attention kinds and distinct head
     * widths. Stated as a test so it cannot drift into a configuration that proves less.
     */
    @Test
    public void theFixtureExercisesWhatItClaimsTo() {
        Gemma4Configuration config = config();

        boolean sawOwnKv = false;
        boolean sawReuse = false;
        boolean sawSwa = false;
        boolean sawFull = false;
        for (int l = 0; l < LAYERS; l++) {
            sawOwnKv |= config.hasOwnKv(l);
            sawReuse |= !config.hasOwnKv(l);
            sawSwa |= config.isSwa(l);
            sawFull |= !config.isSwa(l);
        }
        assertTrue("the fixture must contain a layer that owns its KV", sawOwnKv);
        assertTrue("the fixture must contain a layer that reuses another's KV", sawReuse);
        assertTrue("the fixture must contain a sliding-window layer", sawSwa);
        assertTrue("the fixture must contain a full-attention layer", sawFull);

        assertTrue(
                "sliding-window and full layers must use different head widths",
                HEAD_DIM_SWA != HEAD_DIM_FULL);
        assertTrue(
                "neither head width may equal dim / heads, or the geometry would be derivable",
                HEAD_DIM_SWA != DIM / HEADS && HEAD_DIM_FULL != DIM / HEADS);
        assertTrue(
                "the window must be shorter than the positions tested, or it never bites",
                WINDOW < 6);
        assertTrue("the soft-cap must be non-zero, or the operation never runs", SOFTCAP != 0f);
    }

    @Test
    public void everyKvSourcePrecedesItsConsumerAndMatchesItsWindowKind() {
        Gemma4Configuration config = config();
        for (int l = 0; l < LAYERS; l++) {
            if (config.hasOwnKv(l)) {
                assertEquals("an owning layer has no source", -1, config.kvReuseLayer(l));
                continue;
            }
            int source = config.kvReuseLayer(l);
            assertTrue(
                    "the source must precede its consumer: " + source + " before " + l, source < l);
            assertTrue(
                    "the source must own its key/value data — chained reuse is not designed",
                    config.hasOwnKv(source));
            assertEquals(
                    "source and consumer must agree on the window kind, or the head width and"
                            + " rotary table would silently differ",
                    config.isSwa(source),
                    config.isSwa(l));
            assertEquals(
                    "which is what makes the head widths equal",
                    config.headDim(source),
                    config.headDim(l));
        }
    }

    /** A window that bites must change the result; otherwise the window parameter is untested. */
    @Test
    public void theSlidingWindowChangesTheResult() {
        Gemma4Configuration windowed = config();
        Gemma4Configuration wide =
                new Gemma4Configuration(
                        "FP16",
                        DIM,
                        LAYERS,
                        HEADS,
                        KV_HEADS,
                        HEAD_DIM_SWA,
                        HEAD_DIM_FULL,
                        feedForward(),
                        SWA_PATTERN,
                        CONTEXT,
                        SHARED_KV_LAYERS,
                        PER_LAYER_DIM,
                        VOCAB,
                        CONTEXT,
                        CONTEXT,
                        EPS,
                        10000f,
                        10000f,
                        SOFTCAP);

        FloatTensor narrow = runTo(windowed, 5);
        FloatTensor broad = runTo(wide, 5);

        boolean differs = false;
        for (int i = 0; i < VOCAB; i++) {
            differs |=
                    Float.floatToRawIntBits(narrow.getFloat(i))
                            != Float.floatToRawIntBits(broad.getFloat(i));
        }
        assertTrue(
                "widening the sliding window to the whole context must change the logits", differs);
    }

    private static FloatTensor runTo(Gemma4Configuration config, int lastPosition) {
        Gemma4StandardWeights weights = syntheticWeights(new Random(7L));
        Gemma4State state = new Gemma4State(config, -1);
        FloatTensor logits = null;
        for (int position = 0; position <= lastPosition; position++) {
            logits =
                    InferenceCore.forwardJavaGemma4(config, weights, state, position + 1, position);
        }
        float[] copy = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            copy[i] = logits.getFloat(i);
        }
        return new ArrayFloatTensor(copy);
    }

    // the model

    private static int[] feedForward() {
        int[] ff = new int[LAYERS];
        java.util.Arrays.fill(ff, FF);
        return ff;
    }

    private static Gemma4Configuration config() {
        return new Gemma4Configuration(
                "FP16",
                DIM,
                LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_DIM_SWA,
                HEAD_DIM_FULL,
                feedForward(),
                SWA_PATTERN,
                WINDOW,
                SHARED_KV_LAYERS,
                PER_LAYER_DIM,
                VOCAB,
                CONTEXT,
                CONTEXT,
                EPS,
                10000f,
                10000f,
                SOFTCAP);
    }

    private static Gemma4StandardWeights syntheticWeights(Random random) {
        int maxHeadDim = Math.max(HEAD_DIM_SWA, HEAD_DIM_FULL);
        int perLayerTotal = LAYERS * PER_LAYER_DIM;
        return new Gemma4StandardWeights(
                tensor(random, VOCAB * DIM),
                tensor(random, VOCAB * DIM),
                tensor(random, DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, HEADS * maxHeadDim * DIM),
                tensors(random, LAYERS, KV_HEADS * maxHeadDim * DIM),
                tensors(random, LAYERS, KV_HEADS * maxHeadDim * DIM),
                tensors(random, LAYERS, DIM * HEADS * maxHeadDim),
                tensors(random, LAYERS, maxHeadDim),
                tensors(random, LAYERS, maxHeadDim),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, FF * DIM),
                tensors(random, LAYERS, FF * DIM),
                tensors(random, LAYERS, DIM * FF),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, PER_LAYER_DIM * DIM),
                tensors(random, LAYERS, DIM * PER_LAYER_DIM),
                tensors(random, LAYERS, DIM),
                tensors(random, LAYERS, 1),
                longIndexed(random, (long) VOCAB * perLayerTotal),
                tensor(random, perLayerTotal * DIM),
                tensor(random, perLayerTotal),
                tensor(random, CONTEXT * (maxHeadDim / 2)),
                tensor(random, CONTEXT * (maxHeadDim / 2)),
                tensor(random, CONTEXT * (maxHeadDim / 2)),
                tensor(random, CONTEXT * (maxHeadDim / 2)),
                DataType.F32);
    }

    /**
     * The per-layer embedding table, as a long-indexed one. Small here, but reached through the
     * same long-element-count path the real 2.35-billion-element table needs — the table is never
     * copied.
     */
    private static LongIndexedTensor longIndexed(Random random, long elements) {
        MemorySegment segment = Arena.ofAuto().allocate(elements * Float.BYTES);
        for (long i = 0; i < elements; i++) {
            segment.set(
                    ValueLayout.JAVA_FLOAT, i * Float.BYTES, (random.nextFloat() - 0.5f) * 0.5f);
        }
        return new LongIndexedTensor(segment, DataType.F32);
    }

    private static FloatTensor tensor(Random random, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = (random.nextFloat() - 0.5f) * 0.5f;
        }
        return new ArrayFloatTensor(values);
    }

    private static FloatTensor[] tensors(Random random, int count, int size) {
        FloatTensor[] out = new FloatTensor[count];
        for (int i = 0; i < count; i++) {
            out[i] = tensor(random, size);
        }
        return out;
    }

    private static void assertBitIdentical(
            String what, FloatTensor expected, FloatTensor actual, int size) {
        for (int i = 0; i < size; i++) {
            float e = expected.getFloat(i);
            float a = actual.getFloat(i);
            if (Float.floatToRawIntBits(e) != Float.floatToRawIntBits(a)) {
                assertEquals(what + ", element " + i + " differs", e, a, 0f);
            }
        }
    }

    private static void referenceRmsnorm(
            FloatTensor out,
            FloatTensor x,
            FloatTensor weight,
            int offset,
            int size,
            float rmsNormEps) {
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(
                offset,
                size,
                (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    private static void referenceRmsnormNoWeight(
            FloatTensor out, FloatTensor x, int offset, int size, float rmsNormEps) {
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(offset, size, (value, index) -> finalss * x.getFloat(index));
    }

    private static float referenceGelu(float x) {
        return 0.5f
                * x
                * (1.0f + (float) Math.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
    }

    private static void referenceRopeRotateNeox(
            FloatTensor vec,
            int nHeads,
            int headDim,
            int position,
            FloatTensor freqCisReal,
            FloatTensor freqCisImag) {
        int nComplHead = headDim / 2;
        for (int h = 0; h < nHeads; h++) {
            int base = h * headDim;
            for (int ic = 0; ic < nComplHead; ic++) {
                float fcr = freqCisReal.getFloat(position * nComplHead + ic);
                float fci = freqCisImag.getFloat(position * nComplHead + ic);
                float v0 = vec.getFloat(base + ic);
                float v1 = vec.getFloat(base + ic + nComplHead);
                vec.setFloat(base + ic, v0 * fcr - v1 * fci);
                vec.setFloat(base + ic + nComplHead, v0 * fci + v1 * fcr);
            }
        }
    }

    private static void referenceCopyEmbeddingRow(
            LongIndexedTensor table, long rowIndex, int rowSize, FloatTensor dest, int destOffset) {
        long rowStart = rowIndex * rowSize;
        for (int i = 0; i < rowSize; i++) {
            dest.setFloat(destOffset + i, table.valueAt(rowStart + i));
        }
    }

    static FloatTensor referenceForwardJavaGemma4(
            Gemma4Configuration config,
            Gemma4StandardWeights weights,
            Gemma4State gs,
            int token,
            int position) {
        final State state = gs;

        final int dim = config.dim();
        final int nHead = config.numberOfHeads();
        final int nHeadKv = config.numberOfKeyValueHeads();
        final int kvMul = config.kvMul();
        final int nLayers = config.numberOfLayers();
        final int nEmbdPerLayer = config.embeddingLengthPerLayer();
        final int perLayerTotal = nLayers * nEmbdPerLayer;
        final float attentionScale =
                1.0f; // Gemma4 attention uses scaling = 1.0 (no 1/sqrt(headDim))

        // 1. token embedding, scaled by sqrt(dim)
        weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim);
        final float embedScale = (float) Math.sqrt(dim);
        state.x.mapInPlace(v -> v * embedScale);

        // 2. per-layer embeddings (PLE): inp_per_layer[l] = (referenceRmsnorm(proj(x) / sqrt(dim))
        // + tokEmbd[l]*sqrt(nEmbdPerLayer)) / sqrt(2)
        // per_layer_token_embd is ~2.35B elements (too large for the int-indexed FloatTensor API),
        // so it
        // is addressed one embedding row at a time directly from its raw tensor entry.
        referenceCopyEmbeddingRow(
                weights.perLayerTokenEmbd, token, perLayerTotal, gs.perLayerInputs, 0);
        final float perLayerTokEmbedScale = (float) Math.sqrt(nEmbdPerLayer);
        gs.perLayerInputs.mapInPlace(v -> v * perLayerTokEmbedScale);

        weights.perLayerModelProj.matmul(state.x, gs.perLayerProjScratch, perLayerTotal, dim);
        final float perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        gs.perLayerProjScratch.mapInPlace(v -> v * perLayerProjScale);
        for (int l = 0; l < nLayers; l++) {
            referenceRmsnorm(
                    gs.perLayerProjScratch,
                    gs.perLayerProjScratch,
                    weights.perLayerProjNorm,
                    l * nEmbdPerLayer,
                    nEmbdPerLayer,
                    config.rmsNormEps());
        }
        final float perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));
        for (int i = 0; i < perLayerTotal; i++) {
            float v =
                    (gs.perLayerProjScratch.getFloat(i) + gs.perLayerInputs.getFloat(i))
                            * perLayerInputScale;
            gs.perLayerInputs.setFloat(i, v);
        }

        // 3. transformer layers
        for (int l = 0; l < nLayers; l++) {
            final int curLayer = l;
            final int headDim = config.headDim(l);
            final boolean isSwa = config.isSwa(l);
            final int qDim = nHead * headDim;
            final int kvDim = nHeadKv * headDim;

            FloatTensor freqCisReal = isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull;
            FloatTensor freqCisImag = isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull;

            // attn_norm
            referenceRmsnorm(state.xb, state.x, weights.attnNorm[l], 0, dim, config.rmsNormEps());

            // Q projection, per-head Q-norm, RoPE
            weights.wq[l].matmul(state.xb, state.q, qDim, dim);
            for (int h = 0; h < nHead; h++) {
                referenceRmsnorm(
                        state.q,
                        state.q,
                        weights.attnQNorm[l],
                        h * headDim,
                        headDim,
                        config.rmsNormEps());
            }
            referenceRopeRotateNeox(state.q, nHead, headDim, position, freqCisReal, freqCisImag);

            // K/V: either compute and cache them here, or reuse an earlier layer's cache ("shared
            // KV layers")
            final int kvSrcLayer;
            if (config.hasOwnKv(l)) {
                weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
                weights.wv[l].matmul(state.xb, state.v, kvDim, dim);
                for (int h = 0; h < nHeadKv; h++) {
                    referenceRmsnorm(
                            state.k,
                            state.k,
                            weights.attnKNorm[l],
                            h * headDim,
                            headDim,
                            config.rmsNormEps());
                    referenceRmsnormNoWeight(
                            state.v, state.v, h * headDim, headDim, config.rmsNormEps());
                }
                referenceRopeRotateNeox(
                        state.k, nHeadKv, headDim, position, freqCisReal, freqCisImag);

                state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
                state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);
                kvSrcLayer = l;
            } else {
                kvSrcLayer = config.kvReuseLayer(l);
            }

            // self-attention (causal; sliding-window layers additionally restrict to a local
            // window)
            final int windowStart =
                    isSwa ? Math.max(0, position - config.slidingWindowSize() + 1) : 0;
            Parallel.parallelFor(
                    0,
                    nHead,
                    h -> {
                        int qOffset = h * headDim;
                        int attOffset = h * config.contextLength();
                        int kvHeadOffset = (h / kvMul) * headDim;

                        for (int t = windowStart; t <= position; t++) {
                            int kvOffset = t * kvDim + kvHeadOffset;
                            float score =
                                    state.q.dot(
                                            qOffset, state.keyCache[kvSrcLayer], kvOffset, headDim);
                            score *= attentionScale;
                            state.att.setFloat(attOffset + t, score);
                        }

                        state.att.softmaxInPlace(
                                attOffset + windowStart, position - windowStart + 1);

                        int xbOffset = h * headDim;
                        state.xb.fillInPlace(xbOffset, headDim, 0f);
                        for (int t = windowStart; t <= position; t++) {
                            int kvOffset = t * kvDim + kvHeadOffset;
                            float a = state.att.getFloat(attOffset + t);
                            state.xb.saxpyInPlace(
                                    xbOffset, state.valueCache[kvSrcLayer], kvOffset, headDim, a);
                        }
                    });

            // wo projection, post-attention norm, residual -> attn_out (kept in state.x)
            weights.wo[curLayer].matmul(state.xb, state.xb2, dim, qDim);
            referenceRmsnorm(
                    state.xb2,
                    state.xb2,
                    weights.attnPostNorm[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());
            state.x.addInPlace(
                    state.xb2); // state.x now holds attn_out = inpL + post_attn_norm(attn(...))

            // FFN (GeGLU: down(gelu(gate(x)) * up(x))), post-FFN norm, residual -> cur (kept in
            // state.x)
            referenceRmsnorm(
                    state.xb, state.x, weights.ffnNorm[curLayer], 0, dim, config.rmsNormEps());
            weights.ffnGate[curLayer].matmul(
                    state.xb, state.hb, config.feedForwardLength(curLayer), dim);
            weights.ffnUp[curLayer].matmul(
                    state.xb, state.hb2, config.feedForwardLength(curLayer), dim);
            state.hb.mapInPlace(Gemma4CpuOperationEquivalenceTest::referenceGelu);
            state.hb.multiplyInPlace(state.hb2);
            weights.ffnDown[curLayer].matmul(
                    state.hb, state.xb2, dim, config.feedForwardLength(curLayer));
            referenceRmsnorm(
                    state.xb2,
                    state.xb2,
                    weights.ffnPostNorm[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());
            state.x.addInPlace(
                    state.xb2); // state.x now holds cur = attn_out + post_ffn_norm(ffn(...))

            // per-layer embedding (PLE): cur += per_layer_post_norm(proj(gelu(inp_gate(cur)) *
            // inp_per_layer[l]))
            weights.perLayerInpGate[curLayer].matmul(state.x, gs.perLayerGate, nEmbdPerLayer, dim);
            gs.perLayerGate.mapInPlace(Gemma4CpuOperationEquivalenceTest::referenceGelu);
            int peOffset = curLayer * nEmbdPerLayer;
            for (int j = 0; j < nEmbdPerLayer; j++) {
                gs.perLayerGate.setFloat(
                        j, gs.perLayerGate.getFloat(j) * gs.perLayerInputs.getFloat(peOffset + j));
            }
            weights.perLayerProj[curLayer].matmul(
                    gs.perLayerGate, gs.perLayerOut, dim, nEmbdPerLayer);
            referenceRmsnorm(
                    gs.perLayerOut,
                    gs.perLayerOut,
                    weights.perLayerPostNorm[curLayer],
                    0,
                    dim,
                    config.rmsNormEps());
            state.x.addInPlace(gs.perLayerOut);

            // optional learned per-layer output scale
            FloatTensor outScale = weights.layerOutputScale[curLayer];
            if (outScale != null) {
                final float scale = outScale.getFloat(0);
                state.x.mapInPlace(v -> v * scale);
            }
        }

        // final norm, classifier, and logit soft-capping: logits = softcap * tanh(logits / softcap)
        referenceRmsnorm(state.x, state.x, weights.outputNorm, 0, dim, config.rmsNormEps());
        weights.outputWeight.matmul(state.x, state.logits, config.vocabularySize(), dim);

        final float softcap = config.finalLogitSoftcapping();
        if (softcap != 0.0f) {
            final float invSoftcap = 1.0f / softcap;
            state.logits.mapInPlace(v -> (float) Math.tanh(v * invSoftcap) * softcap);
        }

        return state.logits;
    }
}

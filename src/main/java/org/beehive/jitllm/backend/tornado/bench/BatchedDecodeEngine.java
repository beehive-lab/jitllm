package org.beehive.jitllm.backend.tornado.bench;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LlamaFP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.Qwen3FP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * End-to-end static batched-decode engine for LLaMA FP16: B independent sequences generate
 * together, one token per step, each attending its OWN KV region.
 *
 * <p>Reuses the batched-prefill MMA pipeline (projections as tensor-core GEMMs, weight read once
 * and applied to all B tokens → compute-bound) but swaps the two KV-addressing kernels for the
 * per-slot decode variants ({@link
 * TransformerBatchPrefillKernels#batchedDecodeRopeWithKVCachePacked}, {@link
 * TransformerBatchPrefillKernels#batchedDecodeAttentionFP16Out}) over a B-sized KV cache.
 *
 * <p>ONE step routine serves both phases: prompt tokens are fed with logits discarded (fills the B
 * KV regions with identical RoPE to decode → no cache mismatch); then generated tokens are fed
 * back. Greedy sampling with B copies of the same prompt makes all B streams identical AND equal to
 * the single-stream greedy reference — a bit-exact end-to-end correctness check — while the
 * aggregate B×tok/s is the batching win.
 *
 * <p>Configured with a {@link BatchDecodeOptions}. {@code -Djitllm.prefillBatchSize} must still
 * equal the batch size, because the prefill plan is sized for it:
 *
 * <pre>
 *   var options = BatchDecodeOptions.of(32);          // batch 32, 512 context, 64 tokens
 *   BatchedDecodeEngine.run(model, prompt, options);  // with -Djitllm.prefillBatchSize=32
 * </pre>
 *
 * <p>The {@code -Dbatch.decode.*} properties this once took are <b>gone</b>; see {@link #run(Model,
 * String)} for why they were removed rather than deprecated.
 *
 * <h2>Reproduce (step by step)</h2>
 *
 * <ol>
 *   <li>Build a TornadoVM with the CUDA backend + MMA {@code KernelContext} API ({@code make
 *       BACKEND=cuda}); tested on TornadoVM 5.0.1-jdk21-dev, JDK 21.
 *   <li>Build this project: {@code mvn -Pjdk21 -Dtornadovm.base.version=5.0.1
 *       -Djdk.version.suffix=-jdk21-dev clean package -DskipTests}
 *   <li>Take {@code jitllm --show-command.}, swap the main class to this harness, and pass the
 *       configuration below as a {@link BatchDecodeOptions}. Keep {@code -Djitllm.prefillBatchSize}
 *       equal to its batch size.
 * </ol>
 *
 * <p>Qwen3, static batch (all B streams bit-exact vs single-stream greedy), with {@code
 * -Djitllm.prefillBatchSize=32}:
 *
 * <pre>
 *   BatchDecodeOptions.of(32)
 *   # verify: "all 32 streams identical (== single-stream greedy ref): true"
 * </pre>
 *
 * <p>Qwen3, continuous multi-request serving (paged KV + prefix cache), with {@code
 * -Djitllm.prefillBatchSize=16}:
 *
 * <pre>
 *   new BatchDecodeOptions(16, 512, 64, true,   // batch, context, tokens, CUDA graphs
 *                          true, 16, 0,         // paged, block size, blocks (0 = derive)
 *                          true, 128, 32,       // continuous, requests, minimum tokens
 *                          true, 0.0f, true);   // prefix cache, temperature, device sampling
 *   # verify: "128 requests completed, all mutually prefix-consistent (greedy): true"
 * </pre>
 *
 * <p>Each run prints {@code [verify].: true} (correctness) and {@code [perf].} (throughput).
 */
public class BatchedDecodeEngine {

    static final int RMS_LOCAL = 256;

    /**
     * What a run measured. Returned rather than printed: the backend does not own an output stream.
     */
    public sealed interface Report permits StaticReport, ContinuousReport {
        int batch();

        long steps();

        long generatedTokens();

        double aggregateTokensPerSecond();
    }

    /**
     * A static wave: B slots start together, generate together, and finish together.
     *
     * @param allStreamsIdentical greedy mode only — every slot must reproduce the single-stream
     *     reference, which is what proves the per-slot KV regions are independent
     * @param distinctStreams sampling mode only — how many of the B streams differ, which proves
     *     the same thing from the other direction
     * @param slotOutputs decoded text for the first few slots, for eyeballing
     */
    public record StaticReport(
            int batch,
            int promptLength,
            long steps,
            long generatedTokens,
            double prefillMs,
            double decodeMs,
            double perStepMs,
            double aggregateTokensPerSecond,
            double perStreamTokensPerSecond,
            boolean sampled,
            float temperature,
            boolean allStreamsIdentical,
            int distinctStreams,
            List<String> slotOutputs)
            implements Report {}

    /**
     * Continuous batching: slots are evicted on stop and refilled from a queue, so the wave never
     * drains.
     *
     * @param slotUtilization fraction of slot-steps that did real work, the number continuous
     *     batching exists to raise
     * @param peakBlocks paged mode only — high-water mark of the block pool, against {@code
     *     contiguousBlocksRequired} for what a full reservation would cost
     * @param prefillTokens prefix-cache mode only — against {@code prefillTokensWithoutCache}
     */
    public record ContinuousReport(
            int batch,
            int requests,
            long steps,
            long logitsSkipped,
            long generatedTokens,
            double wallSeconds,
            double aggregateTokensPerSecond,
            double requestsPerSecond,
            double slotUtilization,
            double perStepMs,
            boolean prefixConsistent,
            String longestOutput,
            boolean paged,
            int peakBlocks,
            int poolBlocks,
            int contiguousBlocksRequired,
            int sharedPrefixLength,
            int prefixBlocks,
            long prefillTokens,
            long prefillTokensWithoutCache)
            implements Report {}

    /**
     * Runs the batched-decode engine over {@code model} for {@code userPrompt}, replicated across
     * all B slots.
     */
    /** Runs with the default configuration. */
    public static Report run(Model model, String userPrompt) throws Exception {
        return run(model, userPrompt, BatchDecodeOptions.of(32));
    }

    /** Runs with an explicit configuration, which is what a caller should use. */
    public static Report run(Model model, String userPrompt, BatchDecodeOptions options)
            throws Exception {
        int B = options.batchSize();
        int decodeCtx = options.decodeContext();
        int nDecode = options.decodeTokens();
        boolean cudaGraphs = options.cudaGraphs();
        if (Integer.getInteger("jitllm.prefillBatchSize", 1) != B) {
            throw new IllegalStateException(
                    "Set -Djitllm.prefillBatchSize="
                            + B
                            + " to match the"
                            + " batch size ("
                            + B
                            + "): the prefill plan is sized for it");
        }

        Configuration config = model.configuration();
        TornadoWeights weights = (TornadoWeights) model.weights();
        State state = model.createNewState();
        boolean isQwen3 = config instanceof Qwen3Configuration;

        int dim = config.dim();
        int vocab = config.vocabularySize();
        int numLayers = config.numberOfLayers();
        int kvDim =
                isQwen3
                        ? ((Qwen3Configuration) config).numberOfHeadsValue()
                                * config.numberOfKeyValueHeads()
                        : (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int paddedB = (B + 127) & ~127;

        // ── Prompt tokens ──────────────────────────────────────────────────────
        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userPrompt)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        int P = prompt.size();
        var stopTokens = cf.getStopTokens();
        if (P + nDecode >= decodeCtx) {
            throw new IllegalStateException(
                    "prompt(" + P + ")+n(" + nDecode + ") >= ctx(" + decodeCtx + ")");
        }

        // ── Engine-owned buffers ────────────────────────────────────────────────
        // Paged: KV is a shared pool of fixed-size blocks addressed via a per-slot
        // block table, so the pool can be smaller than the B×ctx contiguous cache.
        boolean paged = options.paged();
        int blockSize = options.blockSize();
        int maxBlocksPerSlot = (decodeCtx + blockSize - 1) / blockSize;
        int numBlocks = options.resolvedBlocks(maxBlocksPerSlot);
        IntArray blockTable = null;
        long kvElems;
        if (paged) {
            // +1 scratch block: inactive slots still execute the KV kernels every step,
            // so they must dump into a reserved block, never a live one.
            kvElems = (long) (numBlocks + 1) * numLayers * blockSize * kvDim;
            blockTable = new IntArray(B * maxBlocksPerSlot);
            blockTable.init(numBlocks); // point everything at the scratch block initially
        } else {
            kvElems = (long) B * numLayers * decodeCtx * kvDim;
        }
        FloatArray keyCacheBatch = new FloatArray((int) kvElems);
        FloatArray valueCacheBatch = new FloatArray((int) kvElems);
        keyCacheBatch.init(0.0f);
        valueCacheBatch.init(0.0f);
        IntArray seqPositions = new IntArray(B);
        FloatArray finalScaleBatch = new FloatArray(B);
        HalfFloatArray normedFinalFP16 = new HalfFloatArray(paddedB * dim);
        normedFinalFP16.init(new uk.ac.manchester.tornado.api.types.HalfFloat(0.0f));
        FloatArray logitsBatch = new FloatArray(paddedB * vocab);
        // On-device greedy: argmax on the GPU, transfer only B token ids (not the
        // 65–78 MB logits tensor). Falls back to host path for temperature sampling.
        boolean deviceSample = options.deviceSample();
        IntArray sampledTokens = new IntArray(paddedB);

        // ── Graphs: [0] activation, [1.N] decode layers, [N+1] batched logits ──
        BatchPrefillActivation activation = new BatchPrefillActivation(state, config, B, false);

        List<ImmutableTaskGraph> layerITGs;
        String lastLayerId;
        java.util.function.Consumer<GridScheduler> updateLayerSched;
        if (isQwen3) {
            var qState = (org.beehive.jitllm.inference.state.Qwen3State) state;
            var qWeights =
                    (org.beehive.jitllm.inference.weights.tornado.Qwen3TornadoWeights) weights;
            Qwen3FP16LayersBatchDecodeMMA q =
                    paged
                            ? new Qwen3FP16LayersBatchDecodeMMA(
                                    qState,
                                    qWeights,
                                    (Qwen3Configuration) config,
                                    B,
                                    decodeCtx,
                                    keyCacheBatch,
                                    valueCacheBatch,
                                    seqPositions,
                                    blockTable,
                                    blockSize,
                                    maxBlocksPerSlot)
                            : new Qwen3FP16LayersBatchDecodeMMA(
                                    qState,
                                    qWeights,
                                    (Qwen3Configuration) config,
                                    B,
                                    decodeCtx,
                                    keyCacheBatch,
                                    valueCacheBatch,
                                    seqPositions);
            layerITGs = q.getLayerImmutableTaskGraphs();
            lastLayerId = q.getLastLayerTaskGraphID();
            updateLayerSched = q::updateGridScheduler;
        } else {
            LlamaFP16LayersBatchDecodeMMA l =
                    paged
                            ? new LlamaFP16LayersBatchDecodeMMA(
                                    (LlamaState) state,
                                    (LlamaTornadoWeights) weights,
                                    (LlamaConfiguration) config,
                                    B,
                                    decodeCtx,
                                    keyCacheBatch,
                                    valueCacheBatch,
                                    seqPositions,
                                    blockTable,
                                    blockSize,
                                    maxBlocksPerSlot)
                            : new LlamaFP16LayersBatchDecodeMMA(
                                    (LlamaState) state,
                                    (LlamaTornadoWeights) weights,
                                    (LlamaConfiguration) config,
                                    B,
                                    decodeCtx,
                                    keyCacheBatch,
                                    valueCacheBatch,
                                    seqPositions);
            layerITGs = l.getLayerImmutableTaskGraphs();
            lastLayerId = l.getLastLayerTaskGraphID();
            updateLayerSched = l::updateGridScheduler;
        }

        KernelContext logitsCtx = new KernelContext();
        HalfFloatArray wclsHalf = weights.wclsByteArray.asHalfFloatArray();
        FloatArray rmsFinal = weights.rms_final_weight_as_floatArray.asFloatArray();
        TaskGraph logitsTg =
                new TaskGraph("batchLogits")
                        .consumeFromDevice(lastLayerId, state.workspace.wrapXBatch)
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                logitsCtx,
                                normedFinalFP16,
                                finalScaleBatch,
                                wclsHalf,
                                rmsFinal)
                        .task(
                                "rms_reduce",
                                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                                logitsCtx,
                                state.workspace.wrapXBatch,
                                finalScaleBatch,
                                dim,
                                config.rmsNormEps(),
                                RMS_LOCAL)
                        .task(
                                "rms_apply",
                                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                                logitsCtx,
                                normedFinalFP16,
                                state.workspace.wrapXBatch,
                                rmsFinal,
                                finalScaleBatch,
                                dim)
                        .task(
                                "vocab",
                                TransformerBatchPrefillKernels::gemmMMA,
                                logitsCtx,
                                normedFinalFP16,
                                wclsHalf,
                                logitsBatch,
                                paddedB,
                                vocab,
                                dim);
        if (deviceSample) { // argmax on GPU → transfer only B ids
            logitsTg.transferToDevice(DataTransferMode.FIRST_EXECUTION, sampledTokens)
                    .task(
                            "argmax",
                            TransformerBatchPrefillKernels::batchedArgmaxLogits,
                            logitsCtx,
                            logitsBatch,
                            sampledTokens,
                            vocab)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, sampledTokens);
        } else {
            logitsTg.transferToHost(DataTransferMode.EVERY_EXECUTION, logitsBatch);
        }
        ImmutableTaskGraph logitsItg = logitsTg.snapshot();

        // ── Grid scheduler ──────────────────────────────────────────────────────
        GridScheduler gs = new GridScheduler();
        activation.updateGridScheduler(gs);
        updateLayerSched.accept(gs);
        gs.addWorkerGrid("batchLogits.rms_reduce", genericWorker(B * RMS_LOCAL, RMS_LOCAL));
        gs.addWorkerGrid("batchLogits.rms_apply", genericWorker(B * dim, 256));
        gs.addWorkerGrid("batchLogits.vocab", mmaGrid(paddedB, vocab));
        if (deviceSample) {
            gs.addWorkerGrid("batchLogits.argmax", genericWorker(B * 256, 256));
        }

        List<ImmutableTaskGraph> all = new ArrayList<>();
        all.add(activation.getImmutableTaskGraph());
        all.addAll(layerITGs);
        all.add(logitsItg);
        int nLayers = layerITGs.size();
        int logitsIdx = 1 + nLayers;

        MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
        long dimBytes = (long) dim * Short.BYTES;

        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]))) {

            if (options.continuous()) {
                return runContinuous(
                        plan,
                        gs,
                        nLayers,
                        logitsIdx,
                        cudaGraphs,
                        model,
                        B,
                        dim,
                        vocab,
                        nDecode,
                        prompt,
                        stopTokens,
                        embTable,
                        dimBytes,
                        state.workspace.embeddingXBatch.getSegment(),
                        seqPositions,
                        logitsBatch,
                        paged,
                        blockTable,
                        blockSize,
                        maxBlocksPerSlot,
                        numBlocks,
                        deviceSample,
                        sampledTokens,
                        options);
            }

            // greedy (bit-exact verification) or per-slot temperature sampling
            // (divergent independent streams — the real concurrent-request case).
            float temperature = options.temperature();
            boolean sample = temperature > 0.0f;
            java.util.Random[] rng = new java.util.Random[B];
            for (int b = 0; b < B; b++) {
                rng[b] = new java.util.Random(1000 + b);
            }

            // ── Prefill: all slots run the identical prompt, logits only at last pos.
            long prefillStart = System.nanoTime();
            for (int t = 0; t < P; t++) {
                int tok = prompt.get(t);
                for (int b = 0; b < B; b++) {
                    MemorySegment.copy(
                            embTable,
                            (long) tok * dimBytes,
                            state.workspace.embeddingXBatch.getSegment(),
                            (long) b * dimBytes,
                            dimBytes);
                    seqPositions.set(b, t);
                }
                boolean needLogits = (t == P - 1);
                runStep(plan, gs, nLayers, logitsIdx, needLogits, cudaGraphs);
            }
            long prefillNs = System.nanoTime() - prefillStart;

            // First generated token per slot (identical → same greedy argmax).
            int[][] streams = new int[B][];
            for (int b = 0; b < B; b++) {
                streams[b] = new int[nDecode];
            }
            int[] cur = new int[B];
            for (int b = 0; b < B; b++) {
                int first =
                        sample
                                ? sampleRow(logitsBatch, b, vocab, temperature, rng[b])
                                : (deviceSample
                                        ? sampledTokens.get(b)
                                        : argmaxRow(logitsBatch, b, vocab));
                cur[b] = first;
                streams[b][0] = first;
            }

            // ── Decode: static batch, greedy. Position advances in lock-step.
            // First few steps capture the CUDA graphs; exclude them from timing so
            // the reported latency is steady-state replay, not capture.
            int warmup = cudaGraphs ? Math.min(4, nDecode - 1) : 0;
            long decodeNs = 0;
            int steps = 0;
            for (int s = 1; s < nDecode; s++) {
                int pos = P + s - 1;
                for (int b = 0; b < B; b++) {
                    MemorySegment.copy(
                            embTable,
                            (long) cur[b] * dimBytes,
                            state.workspace.embeddingXBatch.getSegment(),
                            (long) b * dimBytes,
                            dimBytes);
                    seqPositions.set(b, pos);
                }
                long t0 = System.nanoTime();
                runStep(plan, gs, nLayers, logitsIdx, true, cudaGraphs);
                if (s > warmup) {
                    decodeNs += System.nanoTime() - t0;
                    steps++;
                }
                for (int b = 0; b < B; b++) {
                    int nt =
                            sample
                                    ? sampleRow(logitsBatch, b, vocab, temperature, rng[b])
                                    : (deviceSample
                                            ? sampledTokens.get(b)
                                            : argmaxRow(logitsBatch, b, vocab));
                    cur[b] = nt;
                    streams[b][s] = nt;
                }
            }

            // ── Verify + report ─────────────────────────────────────────────────
            boolean allIdentical = true;
            for (int b = 1; b < B; b++) {
                for (int s = 0; s < nDecode; s++) {
                    if (streams[b][s] != streams[0][s]) {
                        allIdentical = false;
                    }
                }
            }
            java.util.Set<String> distinct = new java.util.HashSet<>();
            for (int b = 0; b < B; b++) {
                distinct.add(java.util.Arrays.toString(streams[b]));
            }
            int showSlots = sample ? Math.min(3, B) : 1;
            List<String> slotOutputs = new ArrayList<>(showSlots);
            for (int b = 0; b < showSlots; b++) {
                slotOutputs.add(decodeStream(model, streams[b], nDecode, stopTokens));
            }

            double decodeSec = decodeNs / 1e9;
            int genTokens = steps * B;
            return new StaticReport(
                    B,
                    P,
                    steps,
                    genTokens,
                    prefillNs / 1e6,
                    decodeNs / 1e6,
                    decodeNs / 1e6 / steps,
                    genTokens / decodeSec,
                    steps / decodeSec,
                    sample,
                    temperature,
                    allIdentical,
                    distinct.size(),
                    List.copyOf(slotOutputs));
        }
    }

    /**
     * Continuous (iteration-level) batching: B slots, each independently either PREFILLING its
     * prompt (token-by-token, logits ignored) or DECODING. A slot that hits a stop token / its
     * max-gen is evicted and immediately refilled from the pending queue — no waiting for the
     * slowest slot in a wave. Because the per-step forward already feeds one token per slot at its
     * own {@code seqPositions[b]} with its own KV region, prefill and decode are the same op; new
     * requests join a partly-decoded batch mid-flight (Orca-style scheduling).
     *
     * <p>Workload: R requests of the same prompt (greedy → deterministic) with random max-gen
     * lengths, so slots finish at different steps and exercise evict/admit. Correctness = all
     * completed outputs are mutually prefix-consistent (the shorter is a prefix of the longer),
     * proving each slot decoded correctly regardless of when it joined.
     */
    private static ContinuousReport runContinuous(
            TornadoExecutionPlan plan,
            GridScheduler gs,
            int nLayers,
            int logitsIdx,
            boolean cudaGraphs,
            Model model,
            int B,
            int dim,
            int vocab,
            int nDecode,
            List<Integer> prompt,
            java.util.Set<Integer> stopTokens,
            MemorySegment embTable,
            long dimBytes,
            MemorySegment embBatchSeg,
            IntArray seqPositions,
            FloatArray logitsBatch,
            boolean paged,
            IntArray blockTable,
            int blockSize,
            int maxBlocksPerSlot,
            int numBlocks,
            boolean deviceSample,
            IntArray sampledTokens,
            BatchDecodeOptions options) {
        int R = options.requests();
        int minN = options.minDecodeTokens();
        boolean refill = true;
        // Prefix caching: the shared prompt prefix (block-aligned) is prefilled ONCE into
        // pinned blocks; every request points its block-table prefix rows at them and
        // starts at pos = sharedPrefixLen — skipping the prefix's prefill entirely.
        boolean prefixCache = options.prefixCache();
        int promptLen = prompt.size();
        int sharedPrefixLen = prefixCache ? ((promptLen - 1) / blockSize) * blockSize : 0;
        int prefixBlocks = sharedPrefixLen / blockSize;

        java.util.ArrayDeque<Integer> freeBlocks = new java.util.ArrayDeque<>();
        List<List<Integer>> slotBlocks =
                new ArrayList<>(); // PRIVATE blocks per slot (freed on evict)
        int[] allocedUpTo = new int[B]; // highest logical block mapped for a slot
        int[] sharedBlocks = new int[Math.max(1, prefixBlocks)];
        int peakBlocks = 0;
        if (paged) {
            for (int i = 0; i < numBlocks; i++) {
                freeBlocks.add(i);
            }
            for (int b = 0; b < B; b++) {
                slotBlocks.add(new ArrayList<>());
            }
        }
        int[] promptPos = new int[B], pos = new int[B], curTok = new int[B], maxGen = new int[B];
        boolean[] decoding = new boolean[B], active = new boolean[B];
        List<List<Integer>> gen = new ArrayList<>();
        for (int b = 0; b < B; b++) {
            gen.add(new ArrayList<>());
        }
        List<int[]> completed = new ArrayList<>();

        java.util.Random wl = new java.util.Random(7);
        java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
        for (int i = 0; i < R; i++) {
            queue.add(minN + wl.nextInt(Math.max(1, nDecode - minN + 1)));
        }

        // ── One-time shared-prefix prefill into pinned blocks ────────────────────
        if (prefixCache && sharedPrefixLen > 0) {
            for (int lb = 0; lb < prefixBlocks; lb++) {
                sharedBlocks[lb] = freeBlocks.poll(); // pinned: never returned
                blockTable.set(0 * maxBlocksPerSlot + lb, sharedBlocks[lb]);
            }
            for (int b = 1; b < B; b++) {
                blockTable.set(b * maxBlocksPerSlot, numBlocks); // scratch
            }
            for (int t = 0; t < sharedPrefixLen; t++) {
                int tok = prompt.get(t);
                MemorySegment.copy(embTable, (long) tok * dimBytes, embBatchSeg, 0L, dimBytes);
                for (int b = 1; b < B; b++) {
                    MemorySegment.copy(
                            embTable,
                            (long) prompt.get(0) * dimBytes,
                            embBatchSeg,
                            (long) b * dimBytes,
                            dimBytes);
                }
                seqPositions.set(0, t);
                for (int b = 1; b < B; b++) {
                    seqPositions.set(b, 0);
                }
                runStep(plan, gs, nLayers, logitsIdx, false, cudaGraphs);
            }
        }

        int warm = cudaGraphs ? 4 : 0;
        long steps = 0,
                genTokens = 0,
                timedNs = 0,
                timedGen = 0,
                timedBusy = 0,
                timedSteps = 0,
                prefillTokens = 0,
                logitsSkipped = 0;
        long tTimed = 0;
        while (true) {
            // ── Admission (single site) ──────────────────────────────────────────
            boolean anyActive = false;
            for (int b = 0; b < B; b++) {
                if (active[b]) {
                    anyActive = true;
                    break;
                }
            }
            boolean admitNow =
                    refill || !anyActive; // continuous: fill any gap; static-wave: only whole new
            // wave
            if (admitNow) {
                for (int b = 0; b < B; b++) {
                    if (!active[b] && !queue.isEmpty()) {
                        maxGen[b] = queue.poll();
                        gen.get(b).clear();
                        active[b] = true;
                        decoding[b] = false;
                        if (paged) {
                            slotBlocks.get(b).clear();
                        }
                        if (prefixCache && sharedPrefixLen > 0) { // reuse the shared prefix
                            for (int lb = 0; lb < prefixBlocks; lb++) {
                                blockTable.set(b * maxBlocksPerSlot + lb, sharedBlocks[lb]);
                            }
                            allocedUpTo[b] = prefixBlocks - 1;
                            promptPos[b] = sharedPrefixLen;
                            pos[b] = sharedPrefixLen;
                        } else {
                            allocedUpTo[b] = -1;
                            promptPos[b] = 0;
                            pos[b] = 0;
                        }
                    }
                }
            }
            anyActive = false;
            for (int b = 0; b < B; b++) {
                if (active[b]) {
                    anyActive = true;
                    break;
                }
            }
            if (!anyActive) {
                break;
            }

            // ── Fill embeddings + positions + block allocation ───────────────────
            for (int b = 0; b < B; b++) {
                int tok =
                        active[b]
                                ? (decoding[b] ? curTok[b] : prompt.get(promptPos[b]))
                                : prompt.get(0);
                MemorySegment.copy(
                        embTable,
                        (long) tok * dimBytes,
                        embBatchSeg,
                        (long) b * dimBytes,
                        dimBytes);
                seqPositions.set(b, active[b] ? pos[b] : 0);
                if (paged) {
                    if (active[b]) {
                        int lb = pos[b] / blockSize;
                        while (allocedUpTo[b] < lb) {
                            if (freeBlocks.isEmpty()) {
                                throw new IllegalStateException(
                                        "KV block pool exhausted ("
                                                + numBlocks
                                                + "); raise BatchDecodeOptions.blocks");
                            }
                            int phys = freeBlocks.poll();
                            allocedUpTo[b]++;
                            blockTable.set(b * maxBlocksPerSlot + allocedUpTo[b], phys);
                            slotBlocks.get(b).add(phys);
                            peakBlocks = Math.max(peakBlocks, numBlocks - freeBlocks.size());
                        }
                    } else {
                        blockTable.set(b * maxBlocksPerSlot, numBlocks); // inactive → scratch
                    }
                }
            }

            // Logits are only needed by slots that SAMPLE this step: any decoding slot,
            // or a prefill slot feeding its last prompt token. If a whole step is pure
            // prefill (common with long prompts), skip the full-vocab logits GEMM.
            boolean needLogits = false;
            for (int b = 0; b < B; b++) {
                if (active[b] && (decoding[b] || promptPos[b] == promptLen - 1)) {
                    needLogits = true;
                    break;
                }
            }
            if (!needLogits) {
                logitsSkipped++;
            }

            long t0 = System.nanoTime();
            runStep(plan, gs, nLayers, logitsIdx, needLogits, cudaGraphs);
            long dt = System.nanoTime() - t0;
            steps++;

            long stepGen = 0, stepBusy = 0;
            for (int b = 0; b < B; b++) {
                if (!active[b]) {
                    continue;
                }
                stepBusy++;
                pos[b]++;
                boolean produced = false;
                int nt = 0;
                if (!decoding[b]) {
                    prefillTokens++;
                    if (promptPos[b] == promptLen - 1) {
                        nt = deviceSample ? sampledTokens.get(b) : argmaxRow(logitsBatch, b, vocab);
                        decoding[b] = true;
                        produced = true;
                    } else {
                        promptPos[b]++;
                    }
                } else {
                    nt = deviceSample ? sampledTokens.get(b) : argmaxRow(logitsBatch, b, vocab);
                    produced = true;
                }
                if (produced) {
                    gen.get(b).add(nt);
                    curTok[b] = nt;
                    genTokens++;
                    stepGen++;
                    boolean stop = stopTokens.contains(nt) || gen.get(b).size() >= maxGen[b];
                    if (stop) {
                        completed.add(gen.get(b).stream().mapToInt(Integer::intValue).toArray());
                        if (paged) { // free PRIVATE blocks only (prefix is pinned)
                            freeBlocks.addAll(slotBlocks.get(b));
                            slotBlocks.get(b).clear();
                        }
                        active[b] = false;
                    }
                }
            }
            if (steps > warm) {
                if (timedSteps == 0) {
                    tTimed = System.nanoTime();
                }
                timedNs += dt;
                timedGen += stepGen;
                timedBusy += stepBusy;
                timedSteps++;
            }
        }
        long wallNs = System.nanoTime() - tTimed;

        // ── Verify: mutually prefix-consistent (greedy, same prompt) ─────────────
        completed.sort(java.util.Comparator.comparingInt(a -> a.length));
        boolean consistent = true;
        int[] longest = completed.isEmpty() ? new int[0] : completed.get(completed.size() - 1);
        for (int[] r : completed) {
            for (int i = 0; i < r.length; i++) {
                if (r[i] != longest[i]) {
                    consistent = false;
                }
            }
        }
        StringBuilder text = new StringBuilder();
        for (int tk : longest) {
            if (stopTokens.contains(tk)) {
                break;
            }
            text.append(model.tokenizer().decode(List.of(tk)));
        }
        double sec = wallNs / 1e9;
        double util = timedSteps > 0 ? (double) timedBusy / (timedSteps * B) : 0.0;
        long prefillNoCache = (prefixCache && sharedPrefixLen > 0) ? (long) R * promptLen : 0L;
        return new ContinuousReport(
                B,
                completed.size(),
                steps,
                logitsSkipped,
                genTokens,
                sec,
                timedGen / sec,
                completed.size() / sec,
                util,
                timedSteps > 0 ? timedNs / 1e6 / timedSteps : 0.0,
                consistent,
                text.toString(),
                paged,
                peakBlocks,
                numBlocks,
                B * maxBlocksPerSlot,
                sharedPrefixLen,
                prefixBlocks,
                prefillTokens,
                prefillNoCache);
    }

    private static void runStep(
            TornadoExecutionPlan plan,
            GridScheduler gs,
            int nLayers,
            int logitsIdx,
            boolean needLogits,
            boolean cudaGraphs) {
        execGraph(plan, gs, 0, cudaGraphs);
        for (int l = 0; l < nLayers; l++) {
            execGraph(plan, gs, 1 + l, cudaGraphs);
        }
        if (needLogits) {
            execGraph(plan, gs, logitsIdx, cudaGraphs);
        }
    }

    private static void execGraph(
            TornadoExecutionPlan plan, GridScheduler gs, int idx, boolean cudaGraphs) {
        var g = plan.withGraph(idx).withGridScheduler(gs);
        if (cudaGraphs) {
            g.withCUDAGraph();
        }
        g.execute();
    }

    private static String decodeStream(
            Model model, int[] stream, int nDecode, java.util.Set<Integer> stopTokens) {
        StringBuilder text = new StringBuilder();
        for (int s = 0; s < nDecode; s++) {
            int tk = stream[s];
            if (stopTokens.contains(tk)) {
                break;
            }
            text.append(model.tokenizer().decode(List.of(tk)));
        }
        return text.toString();
    }

    /** Temperature softmax sampling over one logits row. */
    private static int sampleRow(
            FloatArray logits, int row, int vocab, float temp, java.util.Random rng) {
        long base = (long) row * vocab;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = logits.get((int) (base + i));
            if (v > max) {
                max = v;
            }
        }
        double sum = 0.0;
        double[] probs = new double[vocab];
        for (int i = 0; i < vocab; i++) {
            double e = Math.exp((logits.get((int) (base + i)) - max) / temp);
            probs[i] = e;
            sum += e;
        }
        double r = rng.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < vocab; i++) {
            acc += probs[i];
            if (acc >= r) {
                return i;
            }
        }
        return vocab - 1;
    }

    private static int argmaxRow(FloatArray logits, int row, int vocab) {
        long base = (long) row * vocab;
        int best = 0;
        float bestV = logits.get((int) base);
        for (int i = 1; i < vocab; i++) {
            float v = logits.get((int) (base + i));
            if (v > bestV) {
                bestV = v;
                best = i;
            }
        }
        return best;
    }

    static WorkerGrid genericWorker(int global, int local) {
        WorkerGrid1D g = new WorkerGrid1D(global);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;
        int nBlocks = N / 128;
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }
}

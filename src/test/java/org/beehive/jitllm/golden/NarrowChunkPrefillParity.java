package org.beehive.jitllm.golden;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.BatchPrefillDecodePlan;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

// @formatter:off
/**
 * The narrow prefill family on one model per subclass, a plan 512 wide whose narrow family is 128
 * wide, with the native libraries (MLX on Metal, cuBLAS and cuDNN on CUDA) and with the JIT kernels
 * only.
 *
 * <ul>
 *   <li>First chunks of 40 and 128 tokens: bit for bit the rows of a plan 128 wide, which runs the
 *       same layers on as many rows, or of the plan 512 wide, whose split-K choices the narrow
 *       family's JIT GEMMs take; the same plan with either key/value cache. So the narrow graphs
 *       bind the primary's weights, caches and scratch correctly and register their own grids.
 *   <li>A full chunk followed by a later one of 48 and of 128 tokens: the FP32-cache rows held to
 *       {@link #bounds()} against the FP32 host, the contract every batched prefill is held to, and
 *       the FP16-cache rows to the FP32-cache ones within the FP16-cache contract ({@code
 *       GpuFp16KvPrecision}: cosine 0.9999, relative L2 0.01, top-1 0.95). The full-width plan's
 *       rows ({@code -Djitllm.prefill.narrowChunks=false}) are measured against the host beside
 *       them.
 *   <li>A first chunk of 129 tokens and a later one of 200, past the narrow width: they run the
 *       full-width graphs either way, so the rows are bit for bit the same.
 *   <li>One plan through the 560-token prompt, the 40-token one and a reset: the 560-token rows
 *       come back bit for bit, so neither family leaves state behind for the other.
 *   <li>One plan, counted through a proxy, runs each of those prefills' chunks on the family its
 *       size selects.
 * </ul>
 *
 * <p>The FP16-cache first chunk of Qwen3 F16 on cuBLAS is held to the FP16-cache contract rather
 * than bit for bit: the 128-wide plan's first chunk takes cuDNN attention, which the narrow family
 * never does. Two later-chunk cells are held to the ceilings the owner accepted ({@link
 * #ACCEPTED}). Every case runs and reports before the test fails, so one run gives the whole table.
 * The rows are logits at every generated position from the real generation loop.
 */
// @formatter:on
abstract class NarrowChunkPrefillParity extends CpuGpuParity {

    static final int WIDTH = 512;
    static final int NARROW = 128;
    private static final int CONTEXT = 1024;
    private static final int STEPS = 16;

    /**
     * A later-chunk cell outside one bound, accepted by the owner on 2026-11-11: {@code model}
     * under {@code libraries}, a prefill {@code tokens} long, on the narrow family. Its relative L2
     * on {@code check} ({@code "host"}: FP32 cache against the host; {@code "cache"}: FP16 against
     * FP32 cache) is held to {@code ceiling}, just above the {@code measured} value, in place of
     * the bound; every other gate of the cell keeps its bound.
     *
     * <p>Both are MLX's: it picks its GEMM route by rows (for these 2,048-wide models' projections,
     * split-K at 128 rows, fused at 512), so a 512-row chunk then a 128-row one sums in an order
     * neither width alone does. The Llama value was measured on the {@code q4_0-local} copy of the
     * file.
     */
    record Accepted(
            String model,
            String libraries,
            int tokens,
            String check,
            double bound,
            double measured,
            double ceiling) {}

    static final List<Accepted> ACCEPTED =
            List.of(
                    new Accepted(
                            "granite-3.2-2b-instruct-f16.gguf",
                            "MLX",
                            WIDTH + NARROW,
                            "cache",
                            0.01,
                            0.0107784,
                            0.0108),
                    new Accepted(
                            "Llama-3.2-1B-Instruct-Q4_0.gguf",
                            "MLX",
                            WIDTH + NARROW,
                            "host",
                            0.002,
                            0.00204265,
                            0.00205));

    private static Model gpuModel;
    private static Model cpuModel;
    private static final Map<Integer, KvPrecisionHarness.Trace> HOST = new HashMap<>();

    /** The cases that failed so far in this test, reported together at its end. */
    private final List<String> failures = new ArrayList<>();

    private final Map<String, String> previous = new HashMap<>();

    /** The model file, or null when it is not on this host. */
    protected abstract Path model();

    /** How close the narrow rows must stay to the full-width ones. */
    protected Bounds bounds() {
        return FP16;
    }

    @Before
    public void properties() {
        for (String key :
                new String[] {
                    "use.tornadovm",
                    "jitllm.withPrefillDecode",
                    "jitllm.prefillBatchSize",
                    BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY,
                    ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY
                }) {
            previous.put(key, System.getProperty(key));
        }
        System.setProperty("use.tornadovm", "true");
        System.setProperty("jitllm.withPrefillDecode", "true");
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(WIDTH));
        System.clearProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
    }

    @After
    public void restore() {
        previous.forEach(
                (k, v) -> {
                    if (v == null) {
                        System.clearProperty(k);
                    } else {
                        System.setProperty(k, v);
                    }
                });
    }

    private Model gpu() throws Exception {
        if (gpuModel == null) {
            Path file = model();
            assumeTrue("model absent: " + file, file != null && Files.isRegularFile(file));
            assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
            gpuModel = KvPrecisionHarness.load(file, CONTEXT, true);
        }
        return gpuModel;
    }

    /**
     * A prompt whose batched prefill is exactly {@code tokens} long. The generation loop prefills
     * every prompt token but the last after the state's seed, or every token but the last when the
     * prompt starts with the seed (and always so for Qwen2-MoE).
     */
    private static List<Integer> prefilling(Model model, int tokens) {
        List<Integer> text = KvPrecisionHarness.longPrompt(model, tokens * 2);
        State fresh = model.createNewState();
        boolean seedIsHead =
                model.getModelType() == ModelType.QWEN_2_MOE
                        || PromptIngestion.of(fresh, text, 0).firstIndex() == 1;
        int size = seedIsHead ? tokens + 1 : tokens;
        assertTrue("a long enough prompt: " + text.size() + " tokens", text.size() >= size);
        return new ArrayList<>(text.subList(0, size));
    }

    /** The rows of the 512-wide plan, with or without its narrow family. */
    private static KvPrecisionHarness.Trace run(
            Model model, boolean narrow, DataType keyValue, List<Integer> prompt, int[] forced) {
        if (!narrow) {
            System.setProperty(
                    BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY, "false");
        }
        try {
            return run(model, WIDTH, keyValue, prompt, forced);
        } finally {
            System.clearProperty(BatchPrefillTransformerLayerTaskGraphs.NARROW_CHUNKS_PROPERTY);
        }
    }

    /** The rows of a plan {@code width} wide. */
    private static KvPrecisionHarness.Trace run(
            Model model, int width, DataType keyValue, List<Integer> prompt, int[] forced) {
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(width));
        try {
            KvPrecisionHarness.Trace trace =
                    KvPrecisionHarness.run(model, true, keyValue, width, prompt, STEPS, forced);
            assertEquals("batch-prefill-decode", trace.executionMode());
            return trace;
        } finally {
            System.setProperty("jitllm.prefillBatchSize", String.valueOf(WIDTH));
        }
    }

    /** The FP32 host's rows for a prefill {@code tokens} long, computed once per class. */
    private KvPrecisionHarness.Trace host(int tokens, List<Integer> prompt) throws Exception {
        KvPrecisionHarness.Trace trace = HOST.get(tokens);
        if (trace == null) {
            if (cpuModel == null) {
                cpuModel = KvPrecisionHarness.load(model(), CONTEXT, false);
            }
            // The host runs its own sequential loop: the batched-prefill switches are the device's.
            String batched = System.clearProperty("jitllm.withPrefillDecode");
            String width = System.clearProperty("jitllm.prefillBatchSize");
            try {
                trace =
                        KvPrecisionHarness.run(
                                cpuModel, false, DataType.F32, 1, prompt, STEPS, null);
            } finally {
                System.setProperty("jitllm.withPrefillDecode", batched);
                System.setProperty("jitllm.prefillBatchSize", width);
            }
            HOST.put(tokens, trace);
        }
        return trace;
    }

    /** Runs {@code check}, recording its failure instead of ending the test. */
    private void record(Runnable check) {
        try {
            check.run();
        } catch (AssertionError e) {
            System.out.println("[narrow] FAILED: " + e.getMessage());
            failures.add(e.getMessage());
        }
    }

    /** {@code got}'s rows bit for bit {@code expected}'s. */
    private void assertIdentical(
            String what, KvPrecisionHarness.Trace expected, KvPrecisionHarness.Trace got) {
        record(
                () -> {
                    assertEquals(what + ", rows", expected.rows().size(), got.rows().size());
                    for (int i = 0; i < expected.rows().size(); i++) {
                        assertArrayEquals(
                                what + ", row " + i, expected.rows().get(i), got.rows().get(i), 0f);
                    }
                });
        System.out.println("[narrow] " + what + ": compared bit for bit");
    }

    /**
     * A first chunk of {@code tokens} on the narrow family, bit for bit the rows of one of the two
     * plans it shares its arithmetic with: a plan {@value #NARROW} wide, which runs the same layers
     * on as many rows, or else the plan {@value #WIDTH} wide with no narrow family, whose split-K
     * choices the narrow family's JIT GEMMs take ({@code SplitKGemm.narrowed}). The FP32-cache rows
     * pick the plan; the FP16-cache rows must match the same one, except where its first chunk
     * takes cuDNN attention, which the narrow family never does; there the FP16 rows are held to
     * the FP32 ones within the FP16-cache contract instead.
     *
     * <p>Example: Granite F16 on the CUDA JIT kernels splits its projections as the plan 512 wide
     * does, so its rows are that plan's; on MLX they are the plan 128 wide's.
     */
    private void assertFirstChunk(Model model, String libraries, int tokens) {
        List<Integer> prompt = prefilling(model, tokens);
        KvPrecisionHarness.Trace plan32 = run(model, NARROW, DataType.F32, prompt, null);
        int[] forced = plan32.tokens();
        KvPrecisionHarness.Trace narrow32 = run(model, true, DataType.F32, prompt, forced);
        boolean wide = !identical(plan32, narrow32);
        String against = wide ? "the plan " + WIDTH + " wide" : "a plan " + NARROW + " wide";
        assertIdentical(
                label(libraries, tokens) + ", FP32 cache, against " + against,
                wide ? run(model, false, DataType.F32, prompt, forced) : plan32,
                narrow32);
        KvPrecisionHarness.Trace narrow16 = run(model, true, DataType.F16, prompt, forced);
        if (libraries.contains("cuDNN")) {
            assertFp16Cache(label(libraries, tokens), narrow32, narrow16);
        } else {
            assertIdentical(
                    label(libraries, tokens) + ", FP16 cache, against " + against,
                    wide
                            ? run(model, false, DataType.F16, prompt, forced)
                            : run(model, NARROW, DataType.F16, prompt, forced),
                    narrow16);
        }
    }

    /** Whether {@code a}'s rows are bit for bit {@code b}'s. */
    private static boolean identical(KvPrecisionHarness.Trace a, KvPrecisionHarness.Trace b) {
        if (a.rows().size() != b.rows().size()) {
            return false;
        }
        for (int i = 0; i < a.rows().size(); i++) {
            if (!Arrays.equals(a.rows().get(i), b.rows().get(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * A full chunk and then {@code tokens - 512} on the narrow family, measured as the full-width
     * plan is: the FP32-cache rows against the host within {@link #bounds()}, the FP16-cache rows
     * against the FP32-cache ones within the FP16-cache contract. The narrow family is held to each
     * where the full width meets it; where the full width does not, both are reported. An {@link
     * #ACCEPTED} cell takes its ceiling in place of the relative L2 bound it misses.
     */
    private void assertLaterChunk(Model model, String libraries, int tokens) throws Exception {
        List<Integer> prompt = prefilling(model, tokens);
        KvPrecisionHarness.Trace host = host(tokens, prompt);
        String what = label(libraries, tokens);
        Bounds b = bounds();
        Accepted hostCell = accepted(libraries, tokens, "host", b.relL2());
        Bounds narrowBounds =
                hostCell == null
                        ? b
                        : new Bounds(
                                b.atolPerRms(),
                                b.rtol(),
                                b.ceilingPerRms(),
                                hostCell.ceiling(),
                                b.minCosine(),
                                b.violationFraction(),
                                b.decisionGap());
        Accepted cacheCell = accepted(libraries, tokens, "cache", 0.01);
        double narrowCacheL2 = cacheCell == null ? 0.01 : cacheCell.ceiling();
        KvPrecisionHarness.Trace wide32 = run(model, false, DataType.F32, prompt, host.tokens());
        KvPrecisionHarness.Trace wide16 = run(model, false, DataType.F16, prompt, host.tokens());
        KvPrecisionHarness.Trace narrow32 = run(model, true, DataType.F32, prompt, host.tokens());
        KvPrecisionHarness.Trace narrow16 = run(model, true, DataType.F16, prompt, host.tokens());
        boolean wideHost =
                passes(
                        () ->
                                assertRowsWithin(
                                        what + ", full width, FP32 cache against the host",
                                        host.rows(),
                                        wide32.rows(),
                                        bounds()));
        Runnable narrowHost =
                () ->
                        assertRowsWithin(
                                what + ", narrow, FP32 cache against the host",
                                host.rows(),
                                narrow32.rows(),
                                narrowBounds);
        KvPrecisionHarness.Comparison wideKv = KvPrecisionHarness.compare(wide32, wide16);
        KvPrecisionHarness.Comparison narrowKv = KvPrecisionHarness.compare(narrow32, narrow16);
        System.out.println("[narrow] " + what + ", full width, FP16 against FP32 cache: " + wideKv);
        System.out.println("[narrow] " + what + ", narrow, FP16 against FP32 cache: " + narrowKv);
        boolean wideKvPasses =
                passes(() -> wideKv.assertWithin(what + ", full width", 0.9999, 0.01, 0.95));
        Runnable narrowKvCheck =
                () ->
                        narrowKv.assertWithin(
                                what + ", narrow, FP16 against FP32 cache",
                                0.9999,
                                narrowCacheL2,
                                0.95);
        if (wideHost) {
            record(narrowHost);
        } else {
            System.out.println(
                    "[narrow] baseline: the full width misses the host bounds; narrow: "
                            + (passes(narrowHost) ? "within" : "outside"));
        }
        if (wideKvPasses) {
            record(narrowKvCheck);
        } else {
            System.out.println(
                    "[narrow] baseline: the full width misses the FP16-cache contract; narrow: "
                            + (passes(narrowKvCheck) ? "within" : "outside"));
        }
    }

    /**
     * The {@link #ACCEPTED} cell of this model for {@code check}, or null; its bound must be the
     * one the cell is measured against here.
     */
    private Accepted accepted(String libraries, int tokens, String check, double bound) {
        for (Accepted a : ACCEPTED) {
            if (a.model().equals(model().getFileName().toString())
                    && a.libraries().equals(libraries)
                    && a.tokens() == tokens
                    && a.check().equals(check)) {
                assertEquals("the bound " + a + " was accepted against", a.bound(), bound, 0);
                System.out.printf(
                        "[narrow] accepted cell, %s: relative L2 held to %s in place of %s"
                                + " (measured %s)%n",
                        label(libraries, tokens) + ", " + check,
                        a.ceiling(),
                        a.bound(),
                        a.measured());
                return a;
            }
        }
        return null;
    }

    /** Whether {@code check} passes, its failure printed. */
    private static boolean passes(Runnable check) {
        try {
            check.run();
            return true;
        } catch (AssertionError e) {
            System.out.println("[narrow]   " + e.getMessage());
            return false;
        }
    }

    /**
     * The FP16-cache rows against the FP32-cache ones, as {@code GpuFp16KvPrecision} holds them.
     */
    private void assertFp16Cache(
            String what, KvPrecisionHarness.Trace fp32, KvPrecisionHarness.Trace fp16) {
        KvPrecisionHarness.Comparison kv = KvPrecisionHarness.compare(fp32, fp16);
        System.out.println("[narrow] " + what + ", narrow, FP16 against FP32 cache: " + kv);
        record(
                () ->
                        kv.assertWithin(
                                what + ", narrow, FP16 against FP32 cache", 0.9999, 0.01, 0.95));
    }

    /** The rows of a prefill {@code tokens} long the same with and without the narrow family. */
    private void assertFullWidthEither(Model model, String libraries, int tokens) {
        List<Integer> prompt = prefilling(model, tokens);
        KvPrecisionHarness.Trace off = run(model, false, DataType.F16, prompt, null);
        KvPrecisionHarness.Trace on = run(model, true, DataType.F16, prompt, off.tokens());
        assertIdentical(label(libraries, tokens) + ", FP16 cache, full width either way", off, on);
    }

    private String label(String libraries, int tokens) {
        return model().getFileName() + ", " + libraries + ", prefill of " + tokens + " tokens";
    }

    @Test
    public void nativeLibraries() throws Exception {
        narrowPrefills(true);
    }

    @Test
    public void jitKernels() throws Exception {
        narrowPrefills(false);
    }

    private void narrowPrefills(boolean nativeLibraries) throws Exception {
        Model model = gpu();
        System.setProperty(
                ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY, String.valueOf(nativeLibraries));
        String libraries = assumeNarrowFamily(model, nativeLibraries);
        assertFirstChunk(model, libraries, 40);
        assertFirstChunk(model, libraries, NARROW);
        assertFullWidthEither(model, libraries, NARROW + 1);
        assertLaterChunk(model, libraries, WIDTH + 48);
        assertLaterChunk(model, libraries, WIDTH + NARROW);
        assertFullWidthEither(model, libraries, WIDTH + 200);
        assertTrue(failures.size() + " failed: " + failures, failures.isEmpty());
    }

    @Test
    public void resetRestoresTheSequenceAcrossFamilies() throws Exception {
        Model model = gpu();
        State state = State.withPrefillBatchSize(WIDTH, model::createNewState);
        int seed = state.latestToken;
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            int rows = ((BatchPrefillDecodePlan) plan).narrowPrefillRows();
            assumeTrue("this plan declined the narrow family", rows > 0);
            assertEquals("the narrow family's rows", NARROW, rows);
            List<float[]> fresh = generate(model, state, plan, prefilling(model, WIDTH + 48));
            state.latestToken = seed;
            List<float[]> other = generate(model, state, plan, prefilling(model, 40));
            assertTrue("the short prompt ran", !other.isEmpty());
            plan.resetSequenceState();
            state.latestToken = seed;
            List<float[]> again = generate(model, state, plan, prefilling(model, WIDTH + 48));
            assertEquals("rows after the reset", fresh.size(), again.size());
            for (int i = 0; i < fresh.size(); i++) {
                assertArrayEquals("row " + i + " after the reset", fresh.get(i), again.get(i), 0f);
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    /**
     * Each chunk on the family its size selects, counted through the plan: a chunk of up to 128
     * tokens on the narrow family at any start position, a longer one on the primary, or past
     * position 0 on the fallback family where the plan has one.
     */
    @Test
    public void chunksRunOnTheFamilyTheirSizeSelects() throws Exception {
        Model model = gpu();
        State state = State.withPrefillBatchSize(WIDTH, model::createNewState);
        int seed = state.latestToken;
        BatchPrefillDecodePlan plan =
                (BatchPrefillDecodePlan) TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            assumeTrue("this plan declined the narrow family", plan.narrowPrefillRows() > 0);
            boolean fallback = plan.hasBatchPrefillFallback();
            Dispatch dispatch = new Dispatch(plan);
            BatchPrefillDecodePlan counted =
                    (BatchPrefillDecodePlan)
                            Proxy.newProxyInstance(
                                    BatchPrefillDecodePlan.class.getClassLoader(),
                                    new Class<?>[] {BatchPrefillDecodePlan.class},
                                    dispatch);
            int[][] cases = {
                // prefill tokens, then the chunks expected on primary, fallback, narrow
                {40, 0, 0, 1},
                {NARROW, 0, 0, 1},
                {NARROW + 1, 1, 0, 0},
                {WIDTH + 48, 1, 0, 1},
                {WIDTH + NARROW, 1, 0, 1},
                {WIDTH + 200, fallback ? 1 : 2, fallback ? 1 : 0, 0},
            };
            for (int[] c : cases) {
                plan.resetSequenceState();
                state.latestToken = seed;
                dispatch.clear();
                generate(model, state, counted, prefilling(model, c[0]));
                assertArrayEquals(
                        "chunks on primary, fallback and narrow for a prefill of " + c[0],
                        new int[] {c[1], c[2], c[3]},
                        dispatch.counts());
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    /** Counts which prefill family each chunk ran on, passing every call through to the plan. */
    private static final class Dispatch implements InvocationHandler {
        private final BatchPrefillDecodePlan plan;
        private int primary;
        private int fallback;
        private int narrow;

        Dispatch(BatchPrefillDecodePlan plan) {
            this.plan = plan;
        }

        void clear() {
            primary = fallback = narrow = 0;
        }

        int[] counts() {
            return new int[] {primary, fallback, narrow};
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            switch (method.getName()) {
                case "tornadoVMForwardBatchPrefill" -> primary++;
                case "tornadoVMForwardBatchPrefillFallback" -> fallback++;
                case "tornadoVMForwardBatchPrefillNarrow" -> narrow++;
                default -> {}
            }
            try {
                return method.invoke(plan, args);
            } catch (InvocationTargetException e) {
                throw e.getCause();
            }
        }
    }

    /**
     * The native libraries the plan runs, after skipping where it builds no narrow family (a choice
     * made for precision, {@code PrefillFamily.decline}) or where native libraries were asked for
     * and the device has none, which would only repeat the JIT case.
     */
    private String assumeNarrowFamily(Model model, boolean nativeLibraries) {
        State state = State.withPrefillBatchSize(WIDTH, model::createNewState);
        TornadoVMMasterPlan plan;
        try {
            plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        } catch (UnsupportedOperationException refused) {
            // A family with no native-library path on this backend refuses them by name.
            assumeTrue("no native libraries here: " + refused.getMessage(), !nativeLibraries);
            throw refused;
        }
        try {
            String libraries = plan.executionInfo().nativeLibraries();
            assumeTrue("no native libraries here", !nativeLibraries || !"none".equals(libraries));
            if (!nativeLibraries) {
                assertEquals("JIT kernels only", "none", libraries);
            }
            int rows = ((BatchPrefillDecodePlan) plan).narrowPrefillRows();
            System.out.printf(
                    "  %s: native libraries %s, narrow rows %d%n",
                    model().getFileName(), libraries, rows);
            assumeTrue("this plan declined the narrow family", rows > 0);
            assertEquals("the narrow family's rows", NARROW, rows);
            return libraries;
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    private static List<float[]> generate(
            Model model, State state, TornadoVMMasterPlan plan, List<Integer> prompt) {
        List<float[]> rows = new ArrayList<>();
        Sampler capturing =
                logits -> {
                    float[] row = new float[logits.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = logits.get(i);
                    }
                    rows.add(row);
                    return Sampler.TENSOR_ARGMAX.sampleToken(logits);
                };
        int skippedSeed = PromptIngestion.of(state, prompt, 0).firstIndex();
        int budget = prompt.size() + STEPS - skippedSeed;
        model.generateTokensGPU(state, 0, prompt, Set.of(), budget, capturing, false, null, plan);
        return rows;
    }
}

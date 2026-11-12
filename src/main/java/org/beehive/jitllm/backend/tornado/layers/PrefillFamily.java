package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.inference.state.State;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

// @formatter:off
/**
 * The family of batched-prefill layer graphs a layer class is building: the plan's own, the
 * primary, or the narrow family, the same layers built again over fewer rows for a chunk with no
 * more tokens than that (a short prompt, or the last part of a long one).
 *
 * <p>The plan sets the family on the workspace ({@code TornadoWorkspace.prefillFamily}) around each
 * build, as it sets the weight prepacking, and a layer class reads it there ({@link #of}). Building
 * the narrow family:
 *
 * <ol>
 *   <li>The primary's graphs are plain task graphs, made by {@link #taskGraph}, which keeps them by
 *       name.
 *   <li>A narrow graph names itself {@link #name}{@code (batchPrefillLayer_3)} = {@code
 *       batchPrefillNarrowLayer_3} and has a twin, the primary graph it was made for: the same
 *       layers, the same tasks, so the twin holds the device buffer of every array it uses.
 *       TornadoVM allocates an array in the first graph whose tasks use it, not in the first that
 *       declares it, so the twin, not the graph that uploads an array, is where to bind it from.
 *   <li>Every array the narrow graph declares, however it declares it, is bound from the twin:
 *       weights, scratch, caches, recurrent state, the per-chunk arrays. The plan uploads those
 *       per-chunk ones into the twin's buffers before a narrow chunk ({@link #refreshes}); the
 *       decode graphs read them there, the block table among them.
 *   <li>The only arrays a narrow graph allocates are those it names {@link #own}: buffers sized to
 *       its rows for a library that takes an array only at its exact extent (MLX). Any other array
 *       the twin does not declare fails the build, rather than becoming a second buffer.
 *   <li>The layer class takes the primary's helper objects through {@link #primary}, typed, and
 *       narrows those that size their work by rows ({@code SplitKGemm.narrowed}).
 * </ol>
 *
 * <p>Example: a plan 512 wide builds a narrow family 128 wide. A 100-token prompt, or the last 100
 * tokens of a 612-token one, then runs one tile of rows instead of four.
 */
// @formatter:on
public final class PrefillFamily {

    /** Outside a batched-prefill build: plain graphs, no narrow family. */
    private static final PrefillFamily PLAIN = new PrefillFamily(null, null, 0);

    /** The primary family; null for the primary itself. */
    private final PrefillFamily primaryFamily;

    /** The primary's layer object, for the narrow family's typed sharing; null for the primary. */
    private final BatchPrefillTransformerLayerTaskGraphs primaryLayers;

    /** The narrow family's rows; 0 for the primary. */
    private final int rows;

    /** The primary's graphs, by name. */
    private final Map<String, TaskGraph> graphs = new HashMap<>();

    /** The primary's graphs, by their snapshots. */
    private final Map<ImmutableTaskGraph, String> snapshots = new IdentityHashMap<>();

    /** The narrow family's per-chunk arrays, by the twin that holds them. */
    private final Map<String, List<Object>> refresh = new LinkedHashMap<>();

    private final Set<Object> own = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<Object> bound = Collections.newSetFromMap(new IdentityHashMap<>());
    private int narrowGraphs;

    private PrefillFamily(
            PrefillFamily primaryFamily,
            BatchPrefillTransformerLayerTaskGraphs primaryLayers,
            int rows) {
        this.primaryFamily = primaryFamily;
        this.primaryLayers = primaryLayers;
        this.rows = rows;
    }

    /** A plan's primary family. */
    public static PrefillFamily primary() {
        return new PrefillFamily(null, null, 0);
    }

    /** The narrow family of this primary, {@code rows} wide, over {@code layers}, its layers. */
    public PrefillFamily narrow(int rows, BatchPrefillTransformerLayerTaskGraphs layers) {
        if (primaryFamily != null || this == PLAIN) {
            throw new IllegalStateException("a narrow family is made from a plan's primary");
        }
        return new PrefillFamily(this, layers, rows);
    }

    /**
     * {@code layers}, built as this family: the family is on {@code workspace} for the build, as
     * the weight prepacking is. A narrow build checks that each of its graphs came from {@link
     * #taskGraph}, and throws {@link Declined} where a layer class declines.
     */
    public <T extends BatchPrefillTransformerLayerTaskGraphs> T build(
            TornadoWorkspace workspace, Supplier<T> layers) {
        workspace.prefillFamily = this;
        try {
            T built = layers.get();
            if (isNarrow() && narrowGraphs != built.getLayerImmutableTaskGraphs().size()) {
                throw new IllegalStateException(
                        built.getClass().getSimpleName()
                                + " builds a prefill graph without PrefillFamily.taskGraph");
            }
            return built;
        } finally {
            workspace.prefillFamily = null;
        }
    }

    /** The family {@code state}'s plan is building, or plain graphs outside a build. */
    public static PrefillFamily of(State state) {
        PrefillFamily family = state.workspace.prefillFamily;
        return family != null ? family : PLAIN;
    }

    /** Whether {@code graph} is a narrow family's: it binds its arrays from its twin. */
    public static boolean isNarrow(TaskGraph graph) {
        return graph instanceof NarrowGraph;
    }

    /** Whether this is a narrow family. */
    public boolean isNarrow() {
        return primaryFamily != null;
    }

    /**
     * The primary's layer object, typed, for the objects a narrow family takes from it; null in the
     * primary.
     *
     * @throws IllegalStateException if the primary is of another class
     */
    public <T> T primary(Class<T> type) {
        if (primaryLayers == null) {
            return null;
        }
        if (!type.isInstance(primaryLayers)) {
            throw new IllegalStateException(
                    type.getSimpleName()
                            + " narrows a primary of "
                            + primaryLayers.getClass().getSimpleName());
        }
        return type.cast(primaryLayers);
    }

    /** The name in this family of a primary graph: {@code batchPrefillNarrowLayer_3}. */
    public String name(String primaryName) {
        return isNarrow() ? primaryName.replace("Layer_", "NarrowLayer_") : primaryName;
    }

    /** {@link GraphNames#batchPrefillLayer} in this family: {@code batchPrefillNarrowLayer_3}. */
    public String batchPrefillLayer(int index) {
        return name(GraphNames.batchPrefillLayer(index));
    }

    /** {@link GraphNames#batchLayer} in this family: {@code batchNarrowLayer_3}. */
    public String batchLayer(int index) {
        return name(GraphNames.batchLayer(index));
    }

    /** {@link #taskGraph(String, String)} with the twin of the same name. */
    public TaskGraph taskGraph(String primaryName) {
        return taskGraph(primaryName, primaryName);
    }

    /**
     * The graph made for primary graph {@code primaryName}; in a narrow family its arrays are bound
     * from {@code twin}, a primary graph running the same tasks. Example: Qwen3 F16 under cuDNN
     * twins its narrow graphs with the fallback family's, which run its attention.
     */
    public TaskGraph taskGraph(String primaryName, String twin) {
        if (this == PLAIN) {
            return new TaskGraph(primaryName);
        }
        if (!isNarrow()) {
            TaskGraph graph = new PrimaryGraph(primaryName, this);
            graphs.put(primaryName, graph);
            return graph;
        }
        if (primaryName.contains("NarrowLayer_")) {
            throw new IllegalArgumentException(
                    primaryName + " is a narrow graph's name; pass the primary graph's");
        }
        TaskGraph twinGraph = primaryFamily.graphs.get(twin);
        if (twinGraph == null) {
            throw new IllegalStateException("no primary graph " + twin);
        }
        narrowGraphs++;
        return new NarrowGraph(name(primaryName), this, twin, twinGraph.getArgumentsLookup());
    }

    /**
     * {@code array}, a buffer of the narrow family's own: sized to its rows for a library that
     * takes an array only at its exact extent. The primary's arrays pass through.
     */
    public <T> T own(T array) {
        if (isNarrow()) {
            own.add(array);
        }
        return array;
    }

    /**
     * Ends a narrow build where a layer cannot run fewer rows with the precision it has at the
     * plan's width: the plan then has no narrow family. Throw the result.
     *
     * <p>Example: MLX's quantized GEMM, at 128 rows by 2048 columns, splits its depth and sums the
     * parts in FP16, which it does not at the plan's 512 rows.
     */
    public static Declined decline(String reason) {
        return new Declined(reason);
    }

    /** See {@link #decline}. */
    public static final class Declined extends RuntimeException {
        private Declined(String reason) {
            super(reason, null, false, false);
        }
    }

    /** The narrow graphs built, to check that the layer class built every one through this. */
    public int narrowGraphs() {
        return narrowGraphs;
    }

    /**
     * The per-chunk arrays the narrow family reads from its twins, by the index of each twin in
     * {@code primary}, the primary graphs as the plan lists them: a narrow chunk uploads them there
     * first.
     */
    public Map<Integer, Object[]> refreshes(List<ImmutableTaskGraph> primary) {
        Map<Integer, Object[]> byIndex = new LinkedHashMap<>();
        int found = 0;
        for (int g = 0; g < primary.size(); g++) {
            List<Object> arrays = refresh.get(primaryFamily.snapshots.get(primary.get(g)));
            if (arrays != null) {
                byIndex.put(g, arrays.toArray());
                found++;
            }
        }
        if (found != refresh.size()) {
            throw new IllegalStateException(
                    "a per-chunk array's twin is not a primary layer graph");
        }
        return byIndex;
    }

    /** The primary's arrays the narrow family binds. */
    public int boundArrays() {
        return bound.size();
    }

    /** The arrays the narrow family allocates itself ({@link #own}). */
    public int ownArrays() {
        return own.size();
    }

    /** Their bytes. */
    public long ownBytes() {
        long bytes = 0;
        for (Object o : own) {
            if (o instanceof TornadoNativeArray array) {
                bytes += array.getNumBytesOfSegment();
            }
        }
        return bytes;
    }

    /** A primary graph: plain, its snapshot remembered so a twin's position can be found. */
    private static final class PrimaryGraph extends TaskGraph {

        private final PrefillFamily family;

        PrimaryGraph(String name, PrefillFamily family) {
            super(name);
            this.family = family;
        }

        @Override
        public ImmutableTaskGraph snapshot() {
            ImmutableTaskGraph snapshot = super.snapshot();
            family.snapshots.put(snapshot, getTaskGraphName());
            return snapshot;
        }
    }

    /** A narrow graph: binds every array its twin declares from the twin; see the class comment. */
    private static final class NarrowGraph extends TaskGraph {

        private final PrefillFamily family;
        private final String twin;
        private final Set<Object> twinDeclares;

        NarrowGraph(String name, PrefillFamily family, String twin, Set<Object> twinDeclares) {
            super(name);
            this.family = family;
            this.twin = twin;
            this.twinDeclares = twinDeclares;
        }

        @Override
        public TaskGraph transferToDevice(int mode, Object... objects) {
            List<Object> mine = bindFromTwin(objects, mode == DataTransferMode.EVERY_EXECUTION);
            if (!mine.isEmpty()) {
                super.transferToDevice(mode, mine.toArray());
            }
            return this;
        }

        @Override
        public TaskGraph consumeFromDevice(String source, Object... objects) {
            List<Object> mine = bindFromTwin(objects, false);
            if (!mine.isEmpty()) {
                super.consumeFromDevice(source, mine.toArray());
            }
            return this;
        }

        /** Binds the twin's arrays from it; returns this family's own, which pass as declared. */
        private List<Object> bindFromTwin(Object[] objects, boolean perChunk) {
            List<Object> fromTwin = new ArrayList<>();
            List<Object> mine = new ArrayList<>();
            for (Object o : objects) {
                if (family.own.contains(o)) {
                    mine.add(o);
                } else if (twinDeclares.contains(o)) {
                    fromTwin.add(o);
                } else {
                    throw new IllegalStateException(
                            getTaskGraphName()
                                    + " declares "
                                    + o.getClass().getSimpleName()
                                    + " that its twin "
                                    + twin
                                    + " does not: a narrow graph allocates only its own arrays");
                }
            }
            if (!fromTwin.isEmpty()) {
                family.bound.addAll(fromTwin);
                if (perChunk) {
                    family.refresh.computeIfAbsent(twin, k -> new ArrayList<>()).addAll(fromTwin);
                }
                super.consumeFromDevice(twin, fromTwin.toArray());
            }
            return mine;
        }
    }
}

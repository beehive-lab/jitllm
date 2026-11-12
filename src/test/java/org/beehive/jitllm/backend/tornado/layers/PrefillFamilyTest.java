package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.List;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * What a layer class gets from {@link PrefillFamily} in each build: the family on the workspace,
 * names, the primary's layer object and the decline. The binding of arrays needs a device and is
 * covered by the narrow-chunk accelerator tests.
 *
 * <p>Class A: no device, no model file.
 */
public class PrefillFamilyTest {

    /** A layer class that builds no graphs and records what it was given. */
    private static class Layers implements BatchPrefillTransformerLayerTaskGraphs {
        final PrefillFamily family;
        final Layers primary;
        final String name;
        final String qwen35Name;
        final List<ImmutableTaskGraph> graphs;

        Layers(TornadoWorkspace workspace) {
            this(workspace, List.of());
        }

        Layers(TornadoWorkspace workspace, List<ImmutableTaskGraph> graphs) {
            family = workspace.prefillFamily;
            primary = family.primary(Layers.class);
            name = family.batchPrefillLayer(3);
            qwen35Name = family.batchLayer(3);
            this.graphs = graphs;
        }

        @Override
        public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
            return graphs;
        }

        @Override
        public void updateGridScheduler(GridScheduler scheduler) {}

        @Override
        public String getLastLayerTaskGraphID() {
            return name;
        }
    }

    /** Another layer class, which asks for a primary of its own class. */
    private static final class OtherLayers extends Layers {
        final OtherLayers other;

        OtherLayers(TornadoWorkspace workspace) {
            super(workspace);
            other = family.primary(OtherLayers.class);
        }
    }

    @Test
    public void thePrimaryHasThePlainNames() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        Layers primary = PrefillFamily.primary().build(workspace, () -> new Layers(workspace));
        assertFalse(primary.family.isNarrow());
        assertNull(primary.primary);
        assertEquals("batchPrefillLayer_3", primary.name);
        assertEquals("batchLayer_3", primary.qwen35Name);
        assertNull("the build ends with the family off the workspace", workspace.prefillFamily);
    }

    @Test
    public void theNarrowFamilyTakesThePrimaryTyped() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        Layers narrow = family.narrow(128, primary).build(workspace, () -> new Layers(workspace));
        assertTrue(narrow.family.isNarrow());
        assertSame(primary, narrow.primary);
        assertEquals("batchPrefillNarrowLayer_3", narrow.name);
        assertEquals("batchNarrowLayer_3", narrow.qwen35Name);
        assertNull(workspace.prefillFamily);
    }

    @Test
    public void aPrimaryOfAnotherClassFails() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        PrefillFamily narrow = family.narrow(128, primary);
        assertThrows(
                IllegalStateException.class,
                () -> narrow.build(workspace, () -> new OtherLayers(workspace)));
        assertNull(workspace.prefillFamily);
    }

    @Test
    public void aNarrowGraphNotMadeThroughTheFamilyFails() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        // One graph the layer class says it built, none through the family.
        List<ImmutableTaskGraph> one = Collections.singletonList(null);
        assertThrows(
                IllegalStateException.class,
                () ->
                        family.narrow(128, primary)
                                .build(workspace, () -> new Layers(workspace, one)));
    }

    @Test
    public void aDeclinedBuildHasNoNarrowFamily() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        PrefillFamily narrow = family.narrow(128, primary);
        PrefillFamily.Declined declined =
                assertThrows(
                        PrefillFamily.Declined.class,
                        () ->
                                narrow.build(
                                        workspace,
                                        () -> {
                                            throw PrefillFamily.decline(
                                                    "fewer rows lose precision here");
                                        }));
        assertEquals("fewer rows lose precision here", declined.getMessage());
        assertEquals(0, narrow.ownArrays());
        assertNull(workspace.prefillFamily);
    }

    @Test
    public void aNarrowGraphIsNamedFromItsPrimarysName() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        PrefillFamily narrow = family.narrow(128, primary);
        // A name already narrowed would be narrowed again, and no grid would match its tasks.
        assertThrows(
                IllegalArgumentException.class,
                () -> narrow.taskGraph("batchPrefillNarrowLayer_0", "batchPrefillLayer_0"));
    }

    @Test
    public void onlyAPrimaryHasANarrowFamily() {
        TornadoWorkspace workspace = new TornadoWorkspace();
        PrefillFamily family = PrefillFamily.primary();
        Layers primary = family.build(workspace, () -> new Layers(workspace));
        PrefillFamily narrow = family.narrow(128, primary);
        assertThrows(IllegalStateException.class, () -> narrow.narrow(64, primary));
    }
}

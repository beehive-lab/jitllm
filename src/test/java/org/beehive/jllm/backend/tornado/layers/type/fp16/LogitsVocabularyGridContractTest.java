package org.beehive.jllm.backend.tornado.layers.type.fp16;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;
import uk.ac.manchester.tornado.api.WorkerGrid1D;

// @formatter:off
/**
 * The vocabulary projection's worker grid must pair with the kernel that was installed, including
 * for the subclasses that install their own.
 *
 * <p><b>The defect this exists for.</b> {@code LogitsFP16Layer.updateGridScheduler} derived the
 * vocabulary grid from the device capability. {@code LogitsGraniteFP16Layer} and {@code
 * Gemma4LogitsFP16Layer} override {@code setupLogitsTaskGraph} to install their own {@code
 * vocab_proj} — Granite scales by {@code logitScale}, Gemma 4 soft-caps — and inherit that grid. On
 * a device where the capability was granted they got a 32-lane grid for a kernel that reduces
 * through shared memory: 99.43% of Granite F16 logits wrong at up to 2.4e+08× tolerance, with
 * nothing thrown anywhere.
 *
 * <p><b>No fixture.</b> That defect was caught by a model-backed CPU-parity gate, which needs a
 * GGUF that is not committed and therefore skips wherever it is absent — which is how it survived
 * one full accelerator run. This test needs no model, no device and no GPU, so it runs in ordinary
 * CI on every machine, and it is deliberately not an {@code AccelTest}.
 *
 * <p><b>What it proves.</b> Two things, which together are the contract:
 *
 * <ol>
 *   <li>The pairing itself: {@code vocabularyWorker(v, false)} — the answer for any subclass that
 *       supplies its own kernel — is the shared-memory grid, and {@code vocabularyWorker(v, true)}
 *       is one 32-lane workgroup per row.
 *   <li>That a subclass cannot reach the {@code true} branch by accident. The flag the base reads
 *       is private to {@link LogitsFP16Layer} and is written only inside the branch that installs
 *       the shuffle-reducing kernel, so a subclass that overrides {@code setupLogitsTaskGraph}
 *       cannot set it — that is a language guarantee, and this asserts the field still has the
 *       shape the guarantee depends on.
 * </ol>
 *
 * <p>It also enumerates the subclasses that override {@code setupLogitsTaskGraph}, so adding a
 * third one fails here and makes its author confirm the pairing rather than inherit it silently.
 */
// @formatter:on
public class LogitsVocabularyGridContractTest {

    /** Mirrors LogitsFP16Layer's own constants; asserted against the real grid below. */
    private static final int SHUFFLE_LOCAL = 32;

    private static final int VOCABULARY = 151936;

    /**
     * The subclasses that build their own {@code vocab_proj}, and therefore must keep the
     * shared-memory grid. Adding one means deciding which kernel it installs.
     */
    private static final Set<String> OWN_VOCABULARY_PROJECTION =
            Set.of("LogitsGraniteFP16Layer", "Gemma4LogitsFP16Layer");

    @Test
    public void theShuffleReducingKernelGetsOneThirtyTwoLaneWorkgroupPerRow() {
        WorkerGrid1D grid = LogitsFP16Layer.vocabularyWorker(VOCABULARY, true);
        assertEquals("local work", SHUFFLE_LOCAL, grid.getLocalWork()[0]);
        assertEquals(
                "one workgroup per vocabulary row",
                (long) VOCABULARY * SHUFFLE_LOCAL,
                grid.getGlobalWork()[0]);
    }

    @Test
    public void aCustomSharedMemoryProjectionKeepsTheSharedMemoryGrid() {
        WorkerGrid1D grid = LogitsFP16Layer.vocabularyWorker(VOCABULARY, false);
        long local = grid.getLocalWork()[0];
        assertTrue(
                "a shared-memory vocabulary kernel needs more than one warp per row; it was given"
                        + " "
                        + local
                        + ", which is the shuffle-reducing kernel's grid",
                local > SHUFFLE_LOCAL);
        assertEquals(
                "global work must stay one workgroup per vocabulary row",
                (long) VOCABULARY * local,
                grid.getGlobalWork()[0]);
    }

    @Test
    public void theFlagASubclassMustNotBeAbleToSetIsPrivateToTheBase() throws Exception {
        Field flag = null;
        for (Field f : LogitsFP16Layer.class.getDeclaredFields()) {
            if (f.getType() == boolean.class && f.getName().toLowerCase().contains("shuffle")) {
                flag = f;
            }
        }
        assertTrue(
                "LogitsFP16Layer no longer has the boolean that records which vocabulary kernel"
                        + " setupLogitsTaskGraph installed. If the grid is derived some other way now,"
                        + " this test needs to assert that other way instead of being deleted.",
                flag != null);
        assertTrue(
                "the flag must be private, or a subclass could set it and pair a 32-lane grid with"
                        + " its own shared-memory kernel: "
                        + flag,
                Modifier.isPrivate(flag.getModifiers()));
        assertTrue(
                "the flag must be an instance field: " + flag,
                !Modifier.isStatic(flag.getModifiers()));
    }

    @Test
    public void onlyTheKnownSubclassesBuildTheirOwnVocabularyProjection() throws Exception {
        List<String> found = new ArrayList<>();
        for (Class<?> c : subclassesOfLogitsFp16Layer()) {
            for (var m : c.getDeclaredMethods()) {
                if (m.getName().equals("setupLogitsTaskGraph")) {
                    found.add(c.getSimpleName());
                }
            }
        }
        assertEquals(
                "a class overriding setupLogitsTaskGraph installs its own vocab_proj and therefore"
                        + " inherits the base's worker grid. Confirm that its kernel reduces through"
                        + " shared memory, then add it here.",
                new TreeSet<>(OWN_VOCABULARY_PROJECTION),
                new TreeSet<>(found));
    }

    /** Every loadable class in LogitsFP16Layer's own package that extends it. */
    private static List<Class<?>> subclassesOfLogitsFp16Layer() throws Exception {
        String pkg = LogitsFP16Layer.class.getPackageName();
        URI root =
                LogitsFP16Layer.class.getProtectionDomain().getCodeSource().getLocation().toURI();
        File dir = new File(new File(root), pkg.replace('.', '/'));
        List<Class<?>> out = new ArrayList<>();
        File[] files = dir.listFiles();
        assertTrue("cannot read " + dir + " to enumerate subclasses", files != null);
        for (File f : files) {
            String n = f.getName();
            if (!n.endsWith(".class") || n.contains("$")) {
                continue;
            }
            Class<?> c = Class.forName(pkg + "." + n.substring(0, n.length() - ".class".length()));
            if (c != LogitsFP16Layer.class && LogitsFP16Layer.class.isAssignableFrom(c)) {
                out.add(c);
            }
        }
        return out;
    }
}

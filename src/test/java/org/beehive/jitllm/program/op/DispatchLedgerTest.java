package org.beehive.jllm.program.op;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * <b>Half two, the bound.</b> {@link #countPerFamilyAndDataType} and {@link #overBound} are the
 * enforcement itself, tested against a synthetic census so the bound is seen to fail before it has
 * real code to guard — the same reason Rule 3 was written before the package it protects.
 *
 * <p>Note what the roadmap does <i>not</i> promise: the per-dtype <b>kernel set</b> does not
 * collapse. TornadoVM compiles per concrete native array type and Java has no generics over
 * primitives, so one kernel body cannot serve every representation. What collapses is the
 * per-(model × dtype × mode × MMA) <b>class</b> explosion, which is what this ledger counts.
 */
public class DispatchLedgerTest {

    private static final Path TORNADO = Path.of("src/main/java/org/beehive/jllm/backend/tornado");

    /** The two representation-specific subtrees: layer implementations and plan components. */
    private static final List<Path> DISPATCH_ROOTS =
            List.of(TORNADO.resolve("layers/type"), TORNADO.resolve("plan/components"));

    private static final int K = 2;

    /**
     * What a representation costs today, per representation. <b>Never raise these.</b>
     *
     * <p>{@code F16}: 22 layer classes plus 8 plan components. {@code Q8_0}: 23 plus 9 — one more
     * of each, because Qwen2-MoE ships only in {@code Q8_0}.
     */
    private static final Map<String, Integer> LEDGER = Map.of("fp16", 30, "q8_0", 32);

    /** What one model family costs across both representations today: Llama, 7 and 6. */
    private static final Map<String, Integer> LLAMA_LEDGER = Map.of("fp16", 7, "q8_0", 6);

    @Test
    public void theLegacyDispatchLedgerHasNotGrown() {
        Map<String, Integer> actual = countBy(name -> true);
        for (Map.Entry<String, Integer> pinned : new TreeMap<>(LEDGER).entrySet()) {
            int found = actual.getOrDefault(pinned.getKey(), 0);
            assertTrue(
                    "dispatch classes for "
                            + pinned.getKey()
                            + " grew to "
                            + found
                            + ", ceiling "
                            + pinned.getValue()
                            + ". This tree is collapsed; nothing may add to it.",
                    found <= pinned.getValue());
        }
    }

    @Test
    public void theLedgerIsNotStale() {
        Map<String, Integer> actual = countBy(name -> true);
        assertEquals(
                "the ledger is stale: lower it in the same commit that removed the classes",
                new TreeMap<>(LEDGER),
                new TreeMap<>(actual));
    }

    /**
     * The motivating number, kept visible: one model family, two representations, thirteen classes.
     * A vocabulary parameterized by representation is what makes that two.
     */
    @Test
    public void oneFamilyStillCostsThirteenClassesAcrossTwoRepresentations() {
        Map<String, Integer> llama = countBy(name -> name.startsWith("Llama"));
        assertEquals(new TreeMap<>(LLAMA_LEDGER), new TreeMap<>(llama));
        int total = llama.values().stream().mapToInt(Integer::intValue).sum();
        assertTrue("Llama's dispatch classes grew to " + total, total <= 13);
    }

    @Test
    public void theBoundFlagsAFamilyThatNeedsThreeClassesForOneRepresentation() {
        Map<String, Integer> census = new TreeMap<>();
        census.put("MAT_VEC/F16", 2);
        census.put("MAT_VEC/Q8_0", 1);
        census.put("ATTENTION/F16", 3);
        List<String> over = overBound(census, K);
        assertEquals(
                "only the pair over the bound may be reported", List.of("ATTENTION/F16"), over);
    }

    /** A census within the bound reports nothing. */
    @Test
    public void theBoundPassesAtExactlyK() {
        Map<String, Integer> census = new TreeMap<>();
        census.put("MAT_VEC/F16", K);
        census.put("SOFTMAX/F32", K);
        assertTrue(overBound(census, K).isEmpty());
    }

    /** The operation dispatch surface obeys the bound. */
    @Test
    public void theOperationDispatchSurfaceObeysTheBound() {
        Map<String, Integer> census =
                countPerFamilyAndDataType(Path.of("src/main/java/org/beehive/jllm/backend"));
        List<String> over = overBound(census, K);
        assertTrue("operation dispatch over the k=" + K + " bound: " + over, over.isEmpty());
    }

    // enforcement

    /** Pairs whose dispatch-class count exceeds {@code k}, in a stable order. */
    static List<String> overBound(Map<String, Integer> census, int k) {
        List<String> over = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : new TreeMap<>(census).entrySet()) {
            if (entry.getValue() > k) {
                over.add(entry.getKey());
            }
        }
        return over;
    }

    /**
     * Counts dispatch classes per {@code family/dataType} under {@code root}, by the directory
     * convention the tree already uses: the representation is the directory, the family is the
     * class-name prefix.
     */
    static Map<String, Integer> countPerFamilyAndDataType(Path root) {
        Map<String, Integer> census = new TreeMap<>();
        if (!Files.isDirectory(root)) {
            return census;
        }
        try (Stream<Path> files = Files.walk(root)) {
            files.filter(p -> p.getFileName().toString().endsWith(".java"))
                    .forEach(
                            p -> {
                                String dataType = representationOf(root, p);
                                if (dataType != null) {
                                    String family = p.getFileName().toString().replace(".java", "");
                                    census.merge(family + "/" + dataType, 1, Integer::sum);
                                }
                            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return census;
    }

    private static String representationOf(Path root, Path file) {
        Path relative = root.relativize(file);
        for (Path segment : relative) {
            String name = segment.toString();
            if (name.equals("fp16") || name.equals("q8_0")) {
                return name;
            }
        }
        return null;
    }

    private Map<String, Integer> countBy(java.util.function.Predicate<String> simpleName) {
        Map<String, Integer> counts = new TreeMap<>();
        for (Path root : DISPATCH_ROOTS) {
            assertTrue("dispatch root missing: " + root, Files.isDirectory(root));
            try (Stream<Path> files = Files.walk(root)) {
                files.filter(p -> p.getFileName().toString().endsWith(".java"))
                        .filter(p -> simpleName.test(p.getFileName().toString()))
                        .forEach(
                                p -> {
                                    String dataType = representationOf(root, p);
                                    if (dataType != null) {
                                        counts.merge(dataType, 1, Integer::sum);
                                    }
                                });
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        return counts;
    }
}

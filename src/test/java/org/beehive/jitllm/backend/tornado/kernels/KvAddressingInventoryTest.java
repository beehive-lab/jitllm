package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.Test;

public class KvAddressingInventoryTest {

    /**
     * Occurrences of the per-sequence contiguous KV base, which has nowhere to put a lease term.
     */
    private static final Pattern LEGACY_CONTIGUOUS =
            Pattern.compile(
                    "\\b[A-Za-z_][A-Za-z0-9_]* \\* contextLength \\* [A-Za-z_][A-Za-z0-9_]*");

    private static final int LEGACY_CEILING = 8;

    private static final List<String> PAGED_TWINS =
            List.of(
                    "ropeRotationWithCacheCopyPrecomputedPaged",
                    "ropeRotationWithCacheCopyPrecomputedFP16Paged",
                    "processHeadsFlashAttentionPaged",
                    "processHeadsFlashAttentionFP16Paged",
                    "processHeadsFlashAttentionFP16ScalarPaged",
                    "processHeadsFlashAttentionSplitKVPaged",
                    "processHeadsFlashAttentionSplitKVFP16Paged",
                    "processHeadsFlashAttentionSplitKVFP16PackedPaged",
                    "processHeadsParallelPaged");

    private static final List<String> BATCH_PREFILL_TWINS =
            List.of(
                    "batchedRopeWithKVCachePaged",
                    "batchedRopeWithKVCachePackedPaged",
                    "batchedRopeWithKVCachePackedFP16Paged",
                    "batchedFlashAttentionPaged",
                    "batchedFlashAttentionFP16OutPaged",
                    "batchedFlashAttentionFP16OutKVFP16Paged",
                    "batchedFlashAttentionFP16OutKVFP16PackedTilePaged");

    /** The family twins, per file: Qwen2/Qwen2-MoE/Qwen3, Granite and Phi3. */
    private static final Map<String, List<String>> FAMILY_TWINS =
            Map.of(
                    "Qwen3PagedKvKernels.java",
                            List.of(
                                    "processHeadsParallelPaged",
                                    "ropeRotationWithCacheCopyPaged",
                                    "ropeRotationWithCacheCopyFP16Paged",
                                    "batchedRopeWithKVCacheQwen3Paged",
                                    "batchedRopeWithKVCacheQwen3PackedPaged",
                                    "batchedRopeWithKVCacheQwen3PackedFP16Paged"),
                    "Qwen2PagedKvKernels.java",
                            List.of(
                                    "processHeadsFlashAttentionPaged",
                                    "batchedRopeWithKVCacheQwen2Paged"),
                    "GranitePagedKvKernels.java",
                            List.of(
                                    "processHeadsFlashAttentionWithGraniteScalePaged",
                                    "processHeadsParallelGranitePaged",
                                    "ropeRotationWithCacheCopyPaged"),
                    "Phi3PagedKvKernels.java", List.of("ropeRotationWithCacheCopyPhi3Paged"));

    /** Every twin, per file. Adding a kernel that touches KV means adding it here. */
    private static final Map<String, List<String>> ALL_TWINS = buildLedger();

    private static Map<String, List<String>> buildLedger() {
        Map<String, List<String>> all = new java.util.LinkedHashMap<>(FAMILY_TWINS);
        all.put("TransformerPagedKvKernels.java", PAGED_TWINS);
        all.put("TransformerPagedKvBatchPrefillKernels.java", BATCH_PREFILL_TWINS);
        return all;
    }

    private static final Path MAIN = Path.of("src", "main", "java");

    /** Block and line comments, so the ledger counts code and not prose about the code. */
    private static final Pattern COMMENTS =
            Pattern.compile("/\\*.*?\\*/|//[^\\n]*", Pattern.DOTALL);

    private static String read(Path p) {
        try {
            return COMMENTS.matcher(Files.readString(p, StandardCharsets.UTF_8)).replaceAll("");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Map<String, Integer> legacyOccurrencesByFile() {
        Map<String, Integer> counts = new TreeMap<>();
        try (Stream<Path> files = Files.walk(MAIN)) {
            files.filter(p -> p.toString().endsWith(".java"))
                    .forEach(
                            p -> {
                                Matcher m = LEGACY_CONTIGUOUS.matcher(read(p));
                                int n = 0;
                                while (m.find()) {
                                    n++;
                                }
                                if (n > 0) {
                                    counts.put(MAIN.relativize(p).toString(), n);
                                }
                            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return counts;
    }

    @Test
    public void legacyContiguousKvAddressingOnlyShrinks() {
        assertTrue(
                "this test reads the sources; run it from the project root",
                Files.isDirectory(MAIN));
        Map<String, Integer> byFile = legacyOccurrencesByFile();
        int total = byFile.values().stream().mapToInt(Integer::intValue).sum();

        assertTrue(
                "legacy contiguous KV addressing grew to "
                        + total
                        + " (ceiling "
                        + LEGACY_CEILING
                        + "). If a family migrated, lower the ceiling; if a kernel was added, it should"
                        + " use KvBlockAddress instead. Per file: "
                        + byFile,
                total <= LEGACY_CEILING);
    }

    /** The replacement must not quietly reintroduce what it replaced. */
    @Test
    public void thePagedKernelsContainNoContiguousAddressing() {
        for (String file :
                List.of(
                        "TransformerPagedKvKernels.java",
                        "TransformerPagedKvBatchPrefillKernels.java",
                        "Qwen3PagedKvKernels.java",
                        "Qwen2PagedKvKernels.java",
                        "GranitePagedKvKernels.java",
                        "Phi3PagedKvKernels.java")) {
            String paged = read(MAIN.resolve("org/beehive/jllm/backend/tornado/kernels/" + file));
            Matcher m = LEGACY_CONTIGUOUS.matcher(paged);

            assertTrue(file + " must address KV only through KvBlockAddress", !m.find());
            assertTrue(
                    file + " must actually call the one definition",
                    paged.contains("KvBlockAddress.offset("));
        }
    }

    /** Every migrated kernel exists as a twin, and its legacy original is <b>gone</b>. */
    @Test
    public void everyTwinExistsAndItsLegacyOriginalIsGone() {
        Path kernels = MAIN.resolve("org/beehive/jllm/backend/tornado/kernels");
        for (Map.Entry<String, List<String>> entry : ALL_TWINS.entrySet()) {
            String paged = read(kernels.resolve(entry.getKey()));
            for (String twin : entry.getValue()) {
                assertTrue("missing paged twin " + twin, paged.contains("void " + twin + "("));
                String original = twin.substring(0, twin.length() - "Paged".length());
                assertTrue(
                        "the legacy kernel "
                                + original
                                + " outlived its migration; nothing"
                                + " calls it and nothing writes the layout it reads",
                        !declaredOutsidePagedFiles(kernels, original));
            }
        }
    }

    /** Whether a non-paged kernel file still declares this method. */
    private static boolean declaredOutsidePagedFiles(Path kernels, String method) {
        try (Stream<Path> files = Files.list(kernels)) {
            return files.filter(f -> f.toString().endsWith(".java"))
                    .filter(f -> !f.getFileName().toString().contains("Paged"))
                    .anyMatch(f -> read(f).contains("void " + method + "("));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * {@code processHeadsParallelPaged} calls {@code processHeadTornadoPaged}; that is not a twin.
     */
    private static int countInternalCalls(String paged) {
        return paged.split("processHeadTornadoPaged\\(", -1).length - 1;
    }
}

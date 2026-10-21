package org.beehive.jitllm.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * Every {@code *_rms_reduce} task that reduces through {@code
 * AbstractTransformerLayerTaskGraphs.rmsReduceKernel()} must be paired with a {@code
 * *_rms_finalize} task under the same prefix.
 *
 * <p><b>Why this is a gate and not a style preference.</b> On the NON_NVIDIA path {@code
 * rmsReduceKernel()} selects {@code reductionOneBlockWithLayer}, which splits the sum of squares
 * across workgroups and has workgroup 0 combine the partial sums with no inter-workgroup
 * synchronization. The scale it leaves behind is only correct once a separate {@code
 * reductionFinalNormalization} task recomputes it. Omit that task and the block normalizes by a
 * partial sum: the run still exits 0, still resolves its backend, still reports throughput, and
 * emits fluent-looking token salad.
 *
 * <p>{@code Qwen2Q8_0FFNLayers} shipped in exactly that state — an {@code ffn_rms_finalize} with no
 * {@code attn_rms_finalize} beside it. It was invisible on CUDA and on OpenCL-over-NVIDIA, where
 * {@code DeviceCapability.SINGLE_PASS_RMS} selects the single-workgroup reduction and there is no
 * finalize task to omit, so only the Metal standalone row caught it, and only once that row
 * asserted the output rather than the exit code.
 *
 * <p>The check reads sources rather than task graphs because a task graph needs a device, weights
 * and a model state to exist, and this defect is fully visible in the layout. A layer stack that
 * reduces without {@code rmsReduceKernel()} — {@code Qwen2MoEQ8_0FFNLayers} names {@code
 * reductionOneBlockWithLayerSingleGroup} unconditionally, so no finalize is ever required — is not
 * subject to the rule and is skipped by the same test that enforces it.
 *
 * <p>A layer stack is checked together with the superclasses it inherits from under the same tree,
 * because a shared base can register the reductions' worker grids while each subclass declares the
 * tasks themselves. An abstract base is not checked on its own; it is covered by the concrete
 * stacks that extend it.
 */
public class RmsFinalizePairingTest {

    private static final Path LAYERS =
            Paths.get("src/main/java/org/beehive/jitllm/backend/tornado/layers");

    private static final Pattern REDUCE = Pattern.compile("\"([a-z0-9_]+)_rms_reduce\"");
    private static final Pattern FINALIZE = Pattern.compile("\"([a-z0-9_]+)_rms_finalize\"");
    private static final Pattern REDUCE_CALL = Pattern.compile("rmsReduceKernel\\(\\)\\s*,");
    private static final Pattern EXTENDS = Pattern.compile("\\bextends\\s+(\\w+)");

    @Test
    public void everyMultiWorkgroupRmsReductionIsFinalized() {
        List<Path> sources = layerSources();
        assertFalse("no layer sources found under " + LAYERS.toAbsolutePath(), sources.isEmpty());

        Map<String, Path> byClass = new HashMap<>();
        for (Path source : sources) {
            byClass.put(className(source), source);
        }

        List<String> checked = new ArrayList<>();
        for (Path source : sources) {
            Matcher declaration = declaration(source);
            if (declaration == null || declaration.group(1).contains("abstract")) {
                continue;
            }
            String body = withSuperclasses(source, byClass);
            // Only the stacks that reduce through the shared helper can select the
            // multi-workgroup kernel, and only those owe a finalize task. A call site passes the
            // helper as a task argument; its definition and the comments that name it do not.
            if (!REDUCE_CALL.matcher(body).find()) {
                continue;
            }
            Set<String> reduced = prefixes(REDUCE, body);
            if (reduced.isEmpty()) {
                continue;
            }
            Set<String> finalized = prefixes(FINALIZE, body);
            assertEquals(
                    source.getFileName()
                            + " reduces "
                            + reduced
                            + " but finalizes "
                            + finalized
                            + ". Every block that calls rmsReduceKernel() must emit"
                            + " \"<prefix>_rms_finalize\" under shouldUseFinalNormalization(),"
                            + " or its scale is a partial sum on every non-NVIDIA device.",
                    reduced,
                    finalized);
            checked.add(source.getFileName().toString());
        }

        // A rename of the helper or of the task names would otherwise turn this into a test that
        // passes by checking nothing.
        assertTrue(
                "expected the rule to apply to the layer stacks; it matched " + checked,
                checked.size() >= 10);
        assertTrue(
                "Qwen2Q8_0FFNLayers is the stack this rule was written for and must be covered:"
                        + " matched "
                        + checked,
                checked.contains("Qwen2Q8_0FFNLayers.java"));
    }

    /** The source of {@code source} followed by that of each superclass found under the tree. */
    private static String withSuperclasses(Path source, Map<String, Path> byClass) {
        StringBuilder body = new StringBuilder();
        Set<Path> seen = new LinkedHashSet<>();
        Path current = source;
        while (current != null && seen.add(current)) {
            body.append(read(current)).append('\n');
            Matcher declaration = declaration(current);
            Matcher parent =
                    declaration == null
                            ? null
                            : EXTENDS.matcher(withoutTypeArguments(declaration.group(2)));
            current = parent != null && parent.find() ? byClass.get(parent.group(1)) : null;
        }
        return body.toString();
    }

    /** The declaration of the top-level class named after the file, or {@code null}. */
    private static Matcher declaration(Path source) {
        Pattern pattern =
                Pattern.compile(
                        "((?:\\w+\\s+)*)class\\s+"
                                + Pattern.quote(className(source))
                                + "\\b([^{]*)\\{");
        Matcher matcher = pattern.matcher(read(source));
        return matcher.find() ? matcher : null;
    }

    /** Drops {@code <...>} so a type parameter's bound is not read as the superclass. */
    private static String withoutTypeArguments(String header) {
        String previous;
        do {
            previous = header;
            header = header.replaceAll("<[^<>]*>", "");
        } while (!header.equals(previous));
        return header;
    }

    private static String className(Path source) {
        String name = source.getFileName().toString();
        return name.substring(0, name.length() - ".java".length());
    }

    private static Set<String> prefixes(Pattern pattern, String body) {
        Set<String> found = new LinkedHashSet<>();
        Matcher matcher = pattern.matcher(body);
        while (matcher.find()) {
            found.add(matcher.group(1));
        }
        return found;
    }

    private static List<Path> layerSources() {
        try (Stream<Path> tree = Files.walk(LAYERS)) {
            return tree.filter(p -> p.toString().endsWith(".java")).collect(Collectors.toList());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static String read(Path source) {
        try {
            return new String(Files.readAllBytes(source), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

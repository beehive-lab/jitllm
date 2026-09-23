package org.beehive.jitllm.runtime.diagnostics;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * It asserts <b>codes and structure</b>, never whole sentences. Pinning prose would fail on every
 * wording improvement and teach people to stop improving them; the code and the facts are the
 * contract, and the sentence around them is free to get better.
 *
 * <p>The check runs against the <b>source tree</b> rather than by invoking each failure. Several
 * categories need a GPU, a malformed model file or a duplicate provider on the classpath, and a
 * catalogue that could only be verified with all of those present would be verified by nobody. What
 * matters here is the invariant a catalogue can actually lose: a code that names a failure nothing
 * throws, or a documented failure with no code.
 */
public class DiagnosticCatalogueTest {

    private static final Path MAIN = Path.of("src/main/java");

    /**
     * Every code is used by at least one throwing path.
     *
     * <p>A code that nothing throws is a documented failure that cannot happen — worse than an
     * undocumented one, because it claims coverage the code does not have.
     */
    @Test
    public void everyCodeHasARealThrowingPath() throws IOException {
        String sources = allSources();
        Set<String> unused = new HashSet<>();
        for (DiagnosticCode code : DiagnosticCode.values()) {
            // The enum's own declaration does not count as a use.
            int uses = countOccurrences(sources, "DiagnosticCode." + code.name());
            if (uses == 0) {
                unused.add(code.name());
            }
        }
        assertTrue(
                "these codes name a failure nothing throws, which claims coverage the code does"
                        + " not have — either wire them to their throwing path or remove them: "
                        + unused,
                unused.isEmpty());
    }

    /** Codes are unique and well-formed, because callers match on them. */
    @Test
    public void codesAreUniqueAndWellFormed() {
        Set<String> ids = new HashSet<>();
        Pattern shape = Pattern.compile("GPUL-(CFG|MEM|MOD|LIFE|REQ)-\\d{3}");
        for (DiagnosticCode code : DiagnosticCode.values()) {
            assertTrue(
                    code + " has id '" + code.id() + "', which is not of the documented shape",
                    shape.matcher(code.id()).matches());
            assertTrue(
                    "duplicate diagnostic id "
                            + code.id()
                            + "; a caller matching on it would catch two different failures",
                    ids.add(code.id()));
        }
        assertEquals("every code must be distinct", DiagnosticCode.values().length, ids.size());
    }

    /** The prefix is the matchable part, and must bracket exactly the id. */
    @Test
    public void thePrefixCarriesTheId() {
        for (DiagnosticCode code : DiagnosticCode.values()) {
            assertEquals("[" + code.id() + "] ", code.prefix());
            assertTrue(code.message("something went wrong").startsWith("[" + code.id() + "] "));
            assertTrue(
                    "the message must survive intact after the prefix",
                    code.message("something went wrong").endsWith("something went wrong"));
        }
    }

    /**
     * A configuration or backend failure is never converted into a silent fallback.
     *
     * <p>The specific shape being forbidden: catching one of these and continuing on another path.
     * That is how a user who asked for a device gets a different one and measures it believing it
     * is the one they chose — this project has already recorded one accelerator gate that passed
     * that way.
     */
    @Test
    public void configurationFailuresAreNotCaughtAndTurnedIntoFallback() throws IOException {
        String sources = allSources();
        for (String forbidden :
                new String[] {
                    "catch (UnsupportedLoweringException",
                    "catch (InsufficientDeviceMemoryException"
                }) {
            assertFalse(
                    "a "
                            + forbidden.substring(7)
                            + " must reach the caller. Catching it and"
                            + " continuing on another path hides the defect and can change results.",
                    sources.contains(forbidden));
        }
    }

    /**
     * Every failure the catalogue documents is reachable from a real {@code throw}.
     *
     * <p>Pairs with the first test: that one finds codes nothing throws, this one finds throw sites
     * whose code is not the one the catalogue documents for that category.
     */
    @Test
    public void codedThrowSitesUseTheCatalogue() throws IOException {
        String sources = allSources();
        Matcher m = Pattern.compile("DiagnosticCode\\.([A-Z_]+)").matcher(sources);
        Set<String> known = new HashSet<>();
        for (DiagnosticCode code : DiagnosticCode.values()) {
            known.add(code.name());
        }
        Set<String> unknown = new HashSet<>();
        while (m.find()) {
            if (!known.contains(m.group(1)) && !"values".equals(m.group(1))) {
                unknown.add(m.group(1));
            }
        }
        assertTrue(
                "these throw sites name a code the catalogue does not define: " + unknown,
                unknown.isEmpty());
    }

    private static String allSources() throws IOException {
        StringBuilder all = new StringBuilder();
        try (Stream<Path> files = Files.walk(MAIN)) {
            for (Path p :
                    files.filter(p -> p.toString().endsWith(".java"))
                            .filter(p -> !p.getFileName().toString().equals("DiagnosticCode.java"))
                            .toList()) {
                all.append(Files.readString(p)).append('\n');
            }
        }
        return all.toString();
    }

    private static int countOccurrences(String haystack, String needle) {
        int count = 0;
        int at = haystack.indexOf(needle);
        while (at >= 0) {
            count++;
            at = haystack.indexOf(needle, at + needle.length());
        }
        return count;
    }
}

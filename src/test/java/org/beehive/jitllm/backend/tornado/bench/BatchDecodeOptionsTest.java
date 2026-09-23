package org.beehive.jllm.backend.tornado.bench;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

public class BatchDecodeOptionsTest {

    private static final String[] PROPERTIES = {
        "batch.decode.B",
        "batch.decode.ctx",
        "batch.decode.n",
        "batch.decode.paged",
        "batch.decode.blockSize",
        "batch.decode.blocks",
        "batch.decode.continuous",
        "batch.decode.requests",
        "batch.decode.minN",
        "batch.decode.prefixCache",
        "batch.decode.temp",
        "batch.decode.deviceSample",
        "batch.decode.cudaGraphs"
    };

    /**
     * Set every one of them to a value that would be unmistakable if it were still read, then check
     * the defaults are still the defaults. A removal is only real if something fails when it is
     * undone, and this is that something.
     */
    @Test
    public void theObsoletePropertiesNoLongerAffectAnything() {
        withCleanProperties(
                () -> {
                    System.setProperty("batch.decode.B", "7");
                    System.setProperty("batch.decode.ctx", "99");
                    System.setProperty("batch.decode.n", "5");
                    System.setProperty("batch.decode.paged", "true");
                    System.setProperty("batch.decode.blockSize", "64");
                    System.setProperty("batch.decode.blocks", "12345");
                    System.setProperty("batch.decode.continuous", "true");
                    System.setProperty("batch.decode.requests", "777");
                    System.setProperty("batch.decode.minN", "3");
                    System.setProperty("batch.decode.prefixCache", "true");
                    System.setProperty("batch.decode.temp", "0.9");
                    System.setProperty("batch.decode.deviceSample", "false");
                    System.setProperty("batch.decode.cudaGraphs", "false");
                    System.setProperty("batch.decode.refill", "false");

                    BatchDecodeOptions defaults = BatchDecodeOptions.of(32);
                    assertEquals("batch.decode.B must not be read", 32, defaults.batchSize());
                    assertEquals("nor ctx", 512, defaults.decodeContext());
                    assertEquals("nor n", 64, defaults.decodeTokens());
                    assertFalse("nor paged", defaults.paged());
                    assertEquals("nor blockSize", 16, defaults.blockSize());
                    assertEquals("nor blocks", 0, defaults.blocks());
                    assertFalse("nor continuous", defaults.continuous());
                    assertEquals("nor requests", 128, defaults.requests());
                    assertEquals("nor minN", 32, defaults.minDecodeTokens());
                    assertFalse("nor prefixCache", defaults.prefixCache());
                    assertEquals("nor temp", 0.0f, defaults.temperature(), 0f);
                    assertTrue("nor deviceSample", defaults.deviceSample());
                    assertTrue("nor cudaGraphs", defaults.cudaGraphs());
                });
    }

    /** And there is no reader left to call. */
    @Test
    public void noPropertyReaderSurvivesOnTheHarness() {
        for (var method : BatchedDecodeEngine.class.getDeclaredMethods()) {
            assertFalse(
                    "BatchedDecodeEngine."
                            + method.getName()
                            + " looks like a property reader;"
                            + " the batch.decode.* mapping was removed",
                    method.getName().toLowerCase().contains("systemproperties"));
        }
    }

    /** Unset {@code blocks} means "derive from the batch and context", not "zero blocks". */
    @Test
    public void theBlockCountIsDerivedWhenUnset() {
        BatchDecodeOptions derived = BatchDecodeOptions.of(8);
        assertEquals(0, derived.blocks());
        assertEquals("8 slots x 32 blocks each", 256, derived.resolvedBlocks(32));

        BatchDecodeOptions explicit =
                new BatchDecodeOptions(
                        8, 512, 64, true, true, 16, 100, false, 32, 32, false, 0.0f, true);
        assertEquals("an explicit count wins", 100, explicit.resolvedBlocks(32));
    }

    /**
     * Two settings that used to be silently ignored are now refused.
     *
     * <p>{@code prefixCache} was ANDed with {@code paged}, and {@code deviceSample} with a zero
     * temperature. A run that asked for either and did not get it looked exactly like a run that
     * never asked.
     */
    @Test
    public void combinationsThatCannotHoldAreRefusedRatherThanIgnored() {
        try {
            new BatchDecodeOptions(8, 512, 64, true, false, 16, 0, false, 32, 32, true, 0.0f, true);
            fail("prefix caching without paged storage must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("paged"));
        }
        try {
            new BatchDecodeOptions(
                    8, 512, 64, true, false, 16, 0, false, 32, 32, false, 0.7f, true);
            fail("device sampling with a temperature must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("greedy"));
        }
    }

    /** The properties are a default source, so two values can differ in one JVM. */
    @Test
    public void twoConfigurationsCanCoexist() {
        BatchDecodeOptions small = BatchDecodeOptions.of(4);
        BatchDecodeOptions large = BatchDecodeOptions.of(64);
        assertEquals(4, small.batchSize());
        assertEquals(64, large.batchSize());
        assertFalse(small.equals(large));
    }

    private static void withCleanProperties(Runnable body) {
        String[] saved = new String[PROPERTIES.length];
        for (int i = 0; i < PROPERTIES.length; i++) {
            saved[i] = System.getProperty(PROPERTIES[i]);
            System.clearProperty(PROPERTIES[i]);
        }
        try {
            body.run();
        } finally {
            for (int i = 0; i < PROPERTIES.length; i++) {
                if (saved[i] == null) {
                    System.clearProperty(PROPERTIES[i]);
                } else {
                    System.setProperty(PROPERTIES[i], saved[i]);
                }
            }
        }
    }
}

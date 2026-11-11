package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.tornado.DeviceMemoryExhaustedException;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.junit.Test;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

/**
 * What a GPU session's construction reports when memory runs out ({@link
 * InsufficientDeviceMemoryException#whileAllocating}, which {@code DelegatingModel.newSession} and
 * the legacy runtime's lazy plan go through): this API's exception with {@code GPUL-MEM-001} and
 * the original failure as cause, worded for the host's direct memory or the device budget; any
 * other failure unchanged.
 */
public class InsufficientDeviceMemoryExceptionTest {

    private static final OutOfMemoryError DIRECT =
            new OutOfMemoryError(
                    "Cannot reserve 117440528 bytes of direct buffer memory (allocated 6328288900,"
                            + " limit 6442450944)");

    @Test
    public void rawDirectMemoryExhaustionNamesTheHostLimit() {
        InsufficientDeviceMemoryException failure = thrownBy(DIRECT);
        assertSame(DIRECT, failure.getCause());
        assertEquals(DiagnosticCode.DEVICE_MEMORY_INSUFFICIENT, failure.code());
        assertTrue(failure.getMessage(), failure.getMessage().startsWith("[GPUL-MEM-001]"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("direct memory"));
        assertTrue(failure.getMessage(), failure.getMessage().contains("MaxDirectMemorySize"));
        assertFalse(failure.getMessage(), failure.getMessage().contains("Raise --gpu-memory"));
        assertNull(failure.plan());
    }

    @Test
    public void rawDeviceExhaustionNamesGpuMemory() {
        TornadoOutOfMemoryException device =
                new TornadoOutOfMemoryException("Unable to allocate 33554448 bytes of memory.");
        InsufficientDeviceMemoryException failure = thrownBy(device);
        assertSame(device, failure.getCause());
        assertTrue(failure.getMessage(), failure.getMessage().contains("Raise --gpu-memory"));
    }

    @Test
    public void theBackendsTranslationKeepsItsMessageAndBecomesTheCause() {
        DeviceMemoryExhaustedException backend =
                assertThrows(
                        DeviceMemoryExhaustedException.class,
                        () ->
                                DeviceMemoryExhaustedException.whileAllocating(
                                        () -> {
                                            throw DIRECT;
                                        }));
        InsufficientDeviceMemoryException failure = thrownBy(backend);
        assertSame(backend, failure.getCause());
        assertEquals(backend.getMessage(), failure.getMessage());
    }

    @Test
    public void heapExhaustionAndOtherFailuresPassUnchanged() {
        OutOfMemoryError heap = new OutOfMemoryError("Java heap space");
        assertSame(
                heap,
                assertThrows(
                        OutOfMemoryError.class,
                        () ->
                                InsufficientDeviceMemoryException.whileAllocating(
                                        () -> {
                                            throw heap;
                                        })));
        IllegalStateException other = new IllegalStateException("not memory");
        assertSame(
                other,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                InsufficientDeviceMemoryException.whileAllocating(
                                        () -> {
                                            throw other;
                                        })));
    }

    private static InsufficientDeviceMemoryException thrownBy(Throwable failure) {
        return assertThrows(
                InsufficientDeviceMemoryException.class,
                () ->
                        InsufficientDeviceMemoryException.whileAllocating(
                                () -> {
                                    if (failure instanceof Error error) {
                                        throw error;
                                    }
                                    throw (RuntimeException) failure;
                                }));
    }
}

package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

/**
 * Memory running out on the accelerator path is told apart from the host's direct memory, each with
 * its own remedy, and a heap {@code OutOfMemoryError} is left alone.
 *
 * <p>Example: {@code Cannot reserve 117440528 bytes of direct buffer memory} names {@code
 * -XX:MaxDirectMemorySize}, not {@code --gpu-memory}.
 */
public class DeviceMemoryExhaustedExceptionTest {

    private static final OutOfMemoryError DIRECT =
            new OutOfMemoryError(
                    "Cannot reserve 117440528 bytes of direct buffer memory (allocated 6328288900,"
                            + " limit 6442450944)");

    @Test
    public void hostDirectMemoryNamesTheJvmLimitNotTheDeviceBudget() {
        String message = DeviceMemoryExhaustedException.message("building the plan", DIRECT);
        assertTrue(message, message.startsWith("[GPUL-MEM-001]"));
        assertTrue(message, message.contains("The host ran out of direct memory"));
        assertTrue(message, message.contains("-XX:MaxDirectMemorySize"));
        assertTrue(message, message.contains("--gpu-memory does not change it"));
        assertFalse(message, message.contains("Raise --gpu-memory"));
    }

    @Test
    public void theDeviceBudgetNamesGpuMemory() {
        RuntimeException device =
                new TornadoOutOfMemoryException("Unable to allocate 33554448 bytes of memory.");
        String message = DeviceMemoryExhaustedException.message("building the plan", device);
        assertTrue(message, message.startsWith("[GPUL-MEM-001]"));
        assertTrue(message, message.contains("The device ran out of memory"));
        assertTrue(message, message.contains("Raise --gpu-memory"));
    }

    @Test
    public void recognisedFailuresAreWrappedWithTheirCause() {
        DeviceMemoryExhaustedException host =
                assertThrows(
                        DeviceMemoryExhaustedException.class,
                        () ->
                                DeviceMemoryExhaustedException.whileAllocating(
                                        () -> {
                                            throw DIRECT;
                                        }));
        assertSame(DIRECT, host.getCause());
        assertTrue(host.getMessage(), host.getMessage().contains("direct memory"));
        assertTrue(
                DeviceMemoryExhaustedException.isDeviceExhaustion(
                        new TornadoOutOfMemoryException(
                                "Unable to allocate 33554448 bytes of memory.")));
        assertTrue(
                DeviceMemoryExhaustedException.isDeviceExhaustion(
                        new IllegalStateException("plan", DIRECT)));
    }

    @Test
    public void heapExhaustionAndOtherFailuresPassUnchanged() {
        OutOfMemoryError heap = new OutOfMemoryError("Java heap space");
        assertFalse(DeviceMemoryExhaustedException.isDeviceExhaustion(heap));
        assertSame(
                heap,
                assertThrows(
                        OutOfMemoryError.class,
                        () ->
                                DeviceMemoryExhaustedException.whileAllocating(
                                        () -> {
                                            throw heap;
                                        })));
        IllegalStateException other = new IllegalStateException("not memory");
        assertSame(
                other,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                DeviceMemoryExhaustedException.whileAllocating(
                                        () -> {
                                            throw other;
                                        })));
    }
}

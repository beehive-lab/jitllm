package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

public class ExecutionPlanReleaseTest {

    @Test
    public void aCheckedFailureToCloseIsReportedAsIllegalState() {
        Exception cause = new Exception("close failed");
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                ExecutionPlanRelease.close(
                                        () -> {
                                            throw cause;
                                        }));
        assertSame(cause, thrown.getCause());
    }

    @Test
    public void closesThePlanAndKeepsTheOriginalFailure() {
        AtomicInteger closes = new AtomicInteger();
        RuntimeException failure = new RuntimeException("copy-in failed");

        ExecutionPlanRelease.closeAfter(failure, closes::incrementAndGet);

        assertEquals(1, closes.get());
        assertArrayEquals(new Throwable[0], failure.getSuppressed());
    }

    @Test
    public void aFailureToCloseIsSuppressedOnTheOriginal() {
        RuntimeException failure = new RuntimeException("copy-in failed");
        IllegalStateException cleanup = new IllegalStateException("close failed");

        ExecutionPlanRelease.closeAfter(
                failure,
                () -> {
                    throw cleanup;
                });

        assertEquals(1, failure.getSuppressed().length);
        assertSame(cleanup, failure.getSuppressed()[0]);
    }
}

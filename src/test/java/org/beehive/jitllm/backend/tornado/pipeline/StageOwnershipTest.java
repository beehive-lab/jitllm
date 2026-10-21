package org.beehive.jitllm.backend.tornado.pipeline;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * {@link StageOwnership}: a pipeline that fails while it is built releases every stage it has and
 * its transport, in the transport's order, and the build's failure is the one thrown.
 *
 * <p>{@link #build} mirrors the pipeline constructor's order — stages built one by one, each plan
 * owned as soon as it exists and then configured, the repack, the warm-up — with a failure injected
 * at one point.
 */
public class StageOwnershipTest {

    private enum FailAt {
        NOWHERE,
        CONFIGURING_SECOND_STAGE,
        REPACK,
        WARM_UP
    }

    /** A transport that, like NCCL's, closes the plans before its communicator. */
    private static final class RecordingTransport implements PipelineTransport {
        final List<String> events;

        RecordingTransport(List<String> events) {
            this.events = events;
        }

        @Override
        public void close(Runnable closePlans) {
            try {
                closePlans.run();
            } finally {
                events.add("communicator");
            }
        }

        @Override
        public void addSend(TaskGraph graph, int stage, String producer, FloatArray x, int to) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void addReceive(TaskGraph graph, int stage, FloatArray x, int fromStage) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void updateGridScheduler(int stage, GridScheduler scheduler) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void execute(TornadoExecutionPlan[] plans) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void execute(TornadoExecutionPlan[] plans, int[][] graphs, boolean cudaGraphs) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void executeChunks(TornadoExecutionPlan[] plans, int chunks, ChunkStep step) {
            throw new UnsupportedOperationException();
        }
    }

    private final List<String> events = new ArrayList<>();

    private StageOwnership build(int stages, FailAt failAt, String failingRelease) {
        StageOwnership ownership = new StageOwnership(new RecordingTransport(events));
        try {
            for (int s = 0; s < stages; s++) {
                String name = "stage" + s;
                ownership.own(
                        release(name + ".free", failingRelease),
                        release(name + ".close", failingRelease));
                if (s == 1 && failAt == FailAt.CONFIGURING_SECOND_STAGE) {
                    throw new IllegalStateException("configure stage1");
                }
            }
            if (failAt == FailAt.REPACK) {
                throw new IllegalStateException("repack");
            }
            if (failAt == FailAt.WARM_UP) {
                throw new OutOfMemoryError("warm-up");
            }
        } catch (RuntimeException e) {
            throw ownership.releaseAfter(e);
        } catch (Error e) {
            throw ownership.releaseAfter(e);
        }
        return ownership;
    }

    private StageOwnership.Release release(String event, String failing) {
        return () -> {
            events.add(event);
            if (event.equals(failing)) {
                throw new IllegalStateException(event + " failed");
            }
        };
    }

    private static final String[] EVERYTHING = {
        "stage0.free",
        "stage0.close",
        "stage1.free",
        "stage1.close",
        "stage2.free",
        "stage2.close",
        "communicator"
    };

    @Test
    public void aStageThatFailsWhileConfiguredIsReleasedWithTheOnesBefore() {
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class,
                        () -> build(3, FailAt.CONFIGURING_SECOND_STAGE, null));
        assertEquals("configure stage1", thrown.getMessage());
        assertArrayEquals(
                new String[] {
                    "stage0.free", "stage0.close", "stage1.free", "stage1.close", "communicator"
                },
                events.toArray());
    }

    @Test
    public void aFailedRepackReleasesEveryStage() {
        IllegalStateException thrown =
                assertThrows(IllegalStateException.class, () -> build(3, FailAt.REPACK, null));
        assertEquals("repack", thrown.getMessage());
        assertArrayEquals(EVERYTHING, events.toArray());
    }

    @Test
    public void aFailedWarmUpReleasesEveryStageAndStaysTheErrorThrown() {
        OutOfMemoryError thrown =
                assertThrows(OutOfMemoryError.class, () -> build(3, FailAt.WARM_UP, null));
        assertEquals("warm-up", thrown.getMessage());
        assertEquals(0, thrown.getSuppressed().length);
        assertArrayEquals(EVERYTHING, events.toArray());
    }

    @Test
    public void anEarlyStageThatFailsToReleaseDoesNotStopTheRest() {
        IllegalStateException thrown =
                assertThrows(
                        IllegalStateException.class, () -> build(3, FailAt.REPACK, "stage0.free"));
        // The build's failure stays primary; the release failure rides along.
        assertEquals("repack", thrown.getMessage());
        assertEquals(1, thrown.getSuppressed().length);
        assertEquals("stage0.free failed", thrown.getSuppressed()[0].getMessage());
        // stage0 is still closed after its free failed, and every later stage and the
        // communicator still released.
        assertArrayEquals(EVERYTHING, events.toArray());
    }

    @Test
    public void closingABuiltPipelineReleasesEverythingAndReportsEveryFailure() {
        StageOwnership ownership = build(3, FailAt.NOWHERE, null);
        assertEquals(0, events.size());
        ownership.release();
        assertArrayEquals(EVERYTHING, events.toArray());

        events.clear();
        StageOwnership failing = new StageOwnership(new RecordingTransport(events));
        IllegalStateException first = new IllegalStateException("first");
        IllegalStateException second = new IllegalStateException("second");
        failing.own(
                () -> events.add("a.free"),
                () -> {
                    throw first;
                });
        failing.own(
                () -> {
                    throw second;
                },
                () -> events.add("b.close"));
        IllegalStateException thrown = assertThrows(IllegalStateException.class, failing::release);
        assertSame(first, thrown);
        assertSame(second, thrown.getSuppressed()[0]);
        assertArrayEquals(new String[] {"a.free", "b.close", "communicator"}, events.toArray());
    }
}

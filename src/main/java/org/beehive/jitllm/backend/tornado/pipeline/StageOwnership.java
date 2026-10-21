package org.beehive.jitllm.backend.tornado.pipeline;

import java.util.ArrayList;
import java.util.List;

// @formatter:off
/**
 * What a model split across devices owns: its transport and each stage's execution plan, from the
 * moment each exists. Released together — when the model is closed, or when building it fails
 * part-way, before the caller has an object to close.
 *
 * <p>Every release is attempted: a plan whose device memory cannot be freed is still closed, and a
 * stage that fails to release does not keep the later ones, or the transport, from releasing. The
 * plans are released at the point the transport names ({@link PipelineTransport#close(Runnable)}).
 * The first failure is thrown with the others suppressed on it; after a failed build, the build's
 * own failure stays the one thrown.
 */
// @formatter:on
public final class StageOwnership {

    /** One release action, which may fail. */
    @FunctionalInterface
    public interface Release {
        void run() throws Exception;
    }

    private record Plan(Release freeDeviceMemory, Release close) {}

    private final PipelineTransport transport;
    private final List<Plan> plans = new ArrayList<>();

    public StageOwnership(PipelineTransport transport) {
        this.transport = transport;
    }

    /**
     * Takes ownership of a stage's plan as soon as it is constructed, before it is configured.
     *
     * @param freeDeviceMemory frees the plan's device buffers
     * @param close closes the plan
     */
    public void own(Release freeDeviceMemory, Release close) {
        plans.add(new Plan(freeDeviceMemory, close));
    }

    /**
     * Releases every plan and the transport.
     *
     * @throws RuntimeException the first release that failed, the others suppressed on it; a
     *     checked failure arrives as the cause of an {@link IllegalStateException}
     */
    public void release() {
        List<Throwable> failures = new ArrayList<>();
        try {
            transport.close(() -> releasePlans(failures));
        } catch (RuntimeException | Error e) {
            failures.add(e);
        }
        if (failures.isEmpty()) {
            return;
        }
        Throwable first = failures.get(0);
        for (Throwable later : failures.subList(1, failures.size())) {
            first.addSuppressed(later);
        }
        if (first instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (first instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException("failed to release a pipeline stage", first);
    }

    /**
     * Releases everything after {@code failure} ended the build, and returns {@code failure} to be
     * thrown, carrying any release failure as suppressed.
     */
    public <T extends Throwable> T releaseAfter(T failure) {
        try {
            release();
        } catch (RuntimeException | Error cleanup) {
            failure.addSuppressed(cleanup);
        }
        return failure;
    }

    private void releasePlans(List<Throwable> failures) {
        for (Plan plan : plans) {
            attempt(plan.freeDeviceMemory(), failures);
            attempt(plan.close(), failures);
        }
    }

    private static void attempt(Release release, List<Throwable> failures) {
        try {
            release.run();
        } catch (Exception | Error e) {
            failures.add(e);
        }
    }
}

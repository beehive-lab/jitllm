/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.beehive.jitllm.backend.tornado;

/**
 * Releases a TornadoVM execution plan. {@code close()} frees the plan's device memory; there is
 * nothing more to release, so calling {@code freeDeviceMemory()} first is not needed.
 */
public final class ExecutionPlanRelease {

    private ExecutionPlanRelease() {}

    /** Closes {@code plan}, so its device memory is freed. */
    public static void close(AutoCloseable plan) {
        try {
            plan.close();
        } catch (Exception e) {
            throw new IllegalStateException("failed to close the TornadoVM execution plan", e);
        }
    }

    /**
     * Closes {@code plan} after {@code failure} stopped its initialization. A failure to close is
     * added to {@code failure} as suppressed, so the original cause is the one reported.
     */
    static void closeAfter(Throwable failure, AutoCloseable plan) {
        try {
            plan.close();
        } catch (Exception | Error cleanup) {
            failure.addSuppressed(cleanup);
        }
    }
}

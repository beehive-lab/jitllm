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
package org.beehive.jitllm.engine;

/**
 * Why a request was refused before it ever ran.
 *
 * <p>Rejection is only for requests that can <b>never</b> run, plus the one capacity case that is a
 * policy rather than a shortfall: a full queue. Ordinary capacity shortfall is <b>not</b> rejection
 * — the request queues, because waiting for capacity is a normal state.
 */
public enum RejectionReason {

    /**
     * The queue is at its bound. The engine never blocks in {@code addRequest}, so backpressure is
     * a rejection the caller can see and retry on, not a hidden wait.
     */
    QUEUE_FULL,

    /**
     * The declared budget exceeds what a slot can ever hold. No amount of waiting helps, so this is
     * terminal and immediate rather than queued.
     */
    CANNOT_EVER_FIT,

    /** The request is not well formed — a non-positive budget, or no tokens to run. */
    MALFORMED,

    /** The engine is shutting down and is not accepting work. */
    SHUTDOWN
}

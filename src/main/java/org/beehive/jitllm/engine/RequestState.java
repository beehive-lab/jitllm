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
 * Where a request is in its life. The whole vocabulary — there is no other state.
 *
 * <p>Allowed transitions, and nothing else [engine-contract.md]:
 *
 * <pre>
 * QUEUED  → RUNNING | REJECTED | CANCELLED
 * RUNNING → COMPLETED | FAILED | CANCELLED
 * </pre>
 *
 * <p>Terminal states never transition.
 */
public enum RequestState {

    /** Submitted, waiting for capacity. Waiting is normal, not an error. */
    QUEUED,

    /** Admitted: holds a slot and a lease, and advances one token per step. */
    RUNNING,

    /** Finished generating — stop token, token budget, or context full. */
    COMPLETED(true),

    /** The caller asked to stop, or closed a non-terminal handle. */
    CANCELLED(true),

    /** Never ran. See {@link RejectionReason}. */
    REJECTED(true),

    /** Ran and broke: a callback threw, the backend failed, or an invariant did not hold. */
    FAILED(true);

    private final boolean terminal;

    RequestState() {
        this(false);
    }

    RequestState(boolean terminal) {
        this.terminal = terminal;
    }

    /** Terminal states release their slot and lease, and never transition again. */
    public boolean isTerminal() {
        return terminal;
    }
}

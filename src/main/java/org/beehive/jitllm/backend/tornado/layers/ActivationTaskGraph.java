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
package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Common interface for a single activation task graph (standard, decode, and batch-prefill
 * variants).
 *
 * <p>Implemented by {@link Activation} and custom activation wrappers used by {@link
 * SingleTokenForwardPlanComponents}.
 */
public interface ActivationTaskGraph {
    ImmutableTaskGraph getImmutableTaskGraph();

    GridScheduler updateGridScheduler(GridScheduler scheduler);
}

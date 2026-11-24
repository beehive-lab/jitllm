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
package org.beehive.jitllm.backend.tornado.ops;

import org.beehive.jitllm.backend.tornado.plan.WeightBindings;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;

/** An {@link Implementation} after allocation: it adds the tasks of its steps to task graphs. */
public interface Instance {

    /** Buffers that all layer graphs share: the first graph uploads them, the others reuse them. */
    default Object[] buffers() {
        return new Object[0];
    }

    /**
     * Adds the tasks of {@code step} to {@code graph}, and records in {@code bindings} the weights
     * those tasks read.
     */
    void emit(TaskGraph graph, Step step, Operands operands, WeightBindings bindings);

    /** Adds the worker grids of the tasks {@link #emit} added. */
    void addGrids(GridScheduler scheduler);
}

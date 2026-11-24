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
package org.beehive.jitllm.arch.fixture.model;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Deliberate violator, used only by {@link org.beehive.jitllm.arch.DependencyRulesSelfTest} to
 * prove the rules actually fail on bad code. Never imported by production code and never part of
 * {@link org.beehive.jitllm.arch.ProductionClasses}.
 *
 * <p>Breaks Rule 1 and Rule 2 (imports TornadoVM outside the backend, from a model package), Rule 5
 * (non-final field) and Rule 11 (references TaskGraph).
 */
public class ViolatingModel {

    /** Rule 5 — mutable field on a model type. */
    public FloatArray logits;

    /** Rules 1, 2 and 11 — a plan type in a signature outside the backend. */
    public TaskGraph leak(TaskGraph graph) {
        return graph;
    }
}

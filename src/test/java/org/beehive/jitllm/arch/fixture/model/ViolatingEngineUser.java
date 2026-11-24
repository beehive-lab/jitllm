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

import org.beehive.jitllm.engine.RequestState;

/**
 * Deliberate violator, used only by {@link org.beehive.jitllm.arch.DependencyRulesSelfTest} to
 * prove Rule 18 actually fails on bad code. Never imported by production code and never part of
 * {@link org.beehive.jitllm.arch.ProductionClasses}.
 *
 * <p>Breaks Rule 18: a type in a model package reaching up into {@code.engine.}. If this were
 * allowed, the simple single-sequence path — the one defined by not having an engine — would
 * acquire a dependency on a scheduler it does not need.
 */
public class ViolatingEngineUser {

    /** Rule 18 — an engine type reached from below the engine. */
    public RequestState leak() {
        return RequestState.QUEUED;
    }
}
